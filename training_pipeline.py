"""
Pretrain the LLM

The most basic training pipeline possible.

The model and the training data streams are abstracted out into separate files.

Consider that the h5 files with tokens are already present (using tiktoken r50k_base)

Inspired from github.com/FareedKhan-dev/train-llm-from-scratch
"""

import os
import numpy as np
import torch
import h5py
import tqdm
import contextlib
import tempfile
import time

from collections.abc import Iterator
from model.transformer import Transformer


# Variables necessary for running the training pipeline
# --- From model
vocab_size = 50304
context_length = 128
n_heads = 8
n_embed = 128
n_blocks = 1
# --- Other
train_path, eval_path = "pile_train.h5", "pile_dev.h5"
model_out_path = "transformer_weights.pt"
batch_size = 8
grad_accum = 8
train_steps = 20000
eval_steps = 1000
eval_iters = 100

learning_rate = 5e-4
optimizer = "adamw"
lr_schedule = "cosine"
min_lr = 5e-05
weight_decay = 0.1
decay_step = 16000
grad_clip = 1.0


seed = 123

def batch_iterator(data_path: str,
                   batch_size: int,
                   context_length: int,
                   device: str = "cpu",
                   seed: int = 0
                   ) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    
    rng = np.random.default_rng(seed)

    with h5py.File(data_path, 'r') as hdf5_file:

        dataset = hdf5_file['tokens']

        # num tokens in dataset
        length = dataset.shape[0]
        
        # total possible sequences that can be made from the data
        possible_examples = (length-1)//context_length

        # numbers from 0 to possible_examples-1
        example_idxs = np.arange(possible_examples)

        # randomize the indices
        rng.shuffle(example_idxs)

        # start epoch
        epoch = 0
        counter = 0

        while True:
            if counter+batch_size > possible_examples and counter > 0:
                rng.shuffle(example_idxs)
                counter = 0
                epoch +=1

            
            # true indices for the start token of each batch
            indices = example_idxs[counter:counter+batch_size]*context_length

            samples = torch.tensor(np.array([dataset[idx: idx+context_length+1] for idx in indices]))

            xb = samples[:, :context_length].to(device)
            yb = samples[:, 1:context_length+1].to(device)

            counter+=batch_size

            yield xb, yb


@torch.no_grad()
def estimate_loss(model: Transformer, device: str, steps: int) -> dict[str, float]:
    out = {}
    model.eval()

    for p in [train_path, eval_path]:

        b_iter = batch_iterator(p, batch_size, context_length, device, int(np.random.randint(0, 2**31-1)))

        losses_eval = []

        for _ in range(steps):

            try:
                xb, yb = next(b_iter)
                _, loss = model(xb, yb)
                losses_eval.append(float(loss.item()))
            except StopIteration:
                break
        split = "train" if p==train_path else "eval"

        out[split] = float(np.mean(losses_eval)) if losses_eval else float("nan")

    model.train()
    return out


def save_checkpoint(model, optimizer, device, step, train_loss, dev_loss, losses, save_path):

    payload = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'losses': losses,
            'train_loss': train_loss,
            'dev_loss': dev_loss,
            'step': step,
            'device': device,
            'pytorch_version': torch.__version__,
            'current_lr': optimizer.param_groups[0]["lr"]
            }

    with tempfile.NamedTemporaryFile(
            prefix=f".{os.path.basename(save_path)}.{int(time.time())}.",
            suffix=".tmp",
            delete=False,
            ) as tmp_file:
        tmp_path = tmp_file.name

        try:
            torch.save(payload, tmp_path)
            os.replace(tmp_path, save_path)
        except Exception:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise


def main() -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # I have either cpu or xpu
    device = "xpu" if torch.xpu.is_available() else "cpu"

    # use bf16 for xpu else use float32 for Automatic Mixed Precision
    amp_dtype = torch.bfloat16 if device == "xpu" else torch.float32

    def autocast_ctx():
        if device == "xpu":
            return torch.autocast(device_type=device, dtype=amp_dtype)
        return contextlib.nullcontext()

    print("using device: ", device)

    model = Transformer(n_embed, n_heads, context_length, vocab_size, n_blocks).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    b_iterator = batch_iterator(train_path, batch_size, context_length, device, seed)

    losses: list[float] = []
    start_step = 0

    pbar = tqdm.tqdm(range(start_step, train_steps))

    last_completed_step = 0

    for step in pbar:

        try:

            optimizer.zero_grad(set_to_none=True)
            step_loss = 0.0
            for _ in range(grad_accum):
                xb, yb = next(b_iterator)
                with autocast_ctx():
                    _, loss = model(xb, yb)
                    loss = loss/grad_accum
                loss.backward()
                step_loss+=float(loss.item())

            losses.append(step_loss)

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()

            if step % eval_steps == 0:
                eval_losses = estimate_loss(model, device, eval_iters)

                train_loss = eval_losses["train"]
                eval_loss = eval_losses["eval"]

                print(f"Step: {step}, Train loss: {train_loss}, Eval loss: {eval_loss}")

            if step == decay_step:
                for group in optimizer.param_groups:
                    group["lr"] = min_lr

            last_completed_step = step
        
        except StopIteration:
            print("Training data iterator finished")
            break

    eval_loss = estimate_loss(model, device, 200)
    train_loss = eval_loss["train"]
    dev_loss = eval_loss["eval"]

    save_checkpoint(model, optimizer, device, last_completed_step, train_loss, dev_loss, losses, model_out_path)


if __name__=="__main__":
    main()
