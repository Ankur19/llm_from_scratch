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

from collections.abc import Iterator


# Variables necessary for running the training pipeline
# --- From model
vocab_size = 50304
context_length = 128
# --- Other
train_path, eval_path = "data/train.h5", "data/eval.h5"
batch_size = 8
grad_accum = 12
train_steps = 50000
eval_steps = 1000
eval_iters = 100

learning_rate = 0.0003
optimizer = "adamw"
lr_schedule = "cosine"
min_lr = 3e-05
weight_decay = 0.1
grad_clip = 1.0


seed = 123

def batch_iterator(data_path: str,
                   batch_size: int,
                   context_length: int,
                   device: str = "cpu",
                   seed: int
                   ) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    
    rng = np.random.default_rng(seed)

    with h5py.File(data_path, 'r') as hdf5_file:

        dataset = hdf5_file['tokens']

        # num tokens in dataset
        length = dataset.size[0]
        
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
                epochs +=1

            
            # true indices for the start token of each batch
            indices = example_idxs[counter:counter+batch_size]*context_length

            samples = torch.tensor(np.array([dataset[idx: idx+context_length+1] for idx in indices]))

            xb = samples[:context_length].to(device)
            yb = samples[1:context_length+1].to(device)

            counter+=batch_size

            yeild xb, yb


@torch.no_grad()
def estimate_loss(model, dev_path, steps) -> dict[str, float]


def main() -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # I have either cpu or xpu
    device = "xpu" if torch.xpu.is_available() else "cpu"

    # use bf16 for xpu else use float32 for Automatic Mixed Precision
    amp_dtype = torch.bfloat16 if device == "xpu" else torch.float32

    def autocast_ctx():
        if device = "xpu":
            return torch.autocast(device_type=device, dtype=amp_dtype)
        return contextlib.nullcontext()

    print("using device: ", device)

    model = "<PLACEHOLDER>"

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    b_iterator = batch_iterator(train_path, batch_size, context_length, "cpu", seed)

    losses: list[float] = []
    start_step = 0

    pbar = tqdm(range(start_step, train_steps))

    for step in pbar:

        try:

            optimizer.zero_grad(set_to_none=True)
            step_loss = 0.0
            for _ in range(grad_accum):
                xb, yb = next(b_iterator)
                with autocast_ctx():
                    _, loss = model(xb, yb)
                    loss.backward()
                    loss = loss/grad_accum
                step_loss+=float(loss.item())

            losses.append(step_loss)

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

            if step % eval_steps == 0:

