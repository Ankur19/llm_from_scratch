"""
Pretrain the LLM

The most basic training pipeline possible.

The model and the training data streams are abstracted out into separate files
"""

import os
import numpy as np
import torch
import h5py


# Variables necessary for running the training pipeline
# --- From model
vocab_size = 50304
context_length = 128
# --- Other
train_path, eval_path = "data/train.h5", "data/eval.h5"
batch_size = 8
grad_accum_steps = 12

seed = 123

def main() -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    model = "<PLACEHOLDER>"


