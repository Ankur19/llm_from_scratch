import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Head(nn.Module):

    tril: torch.Tensor

    def __init__(self, n_embed: int, head_size: int, context_length: int):
        super().__init__()

        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)

        self.register_buffer('tril', torch.tril(torch.ones(context_size, context_size)), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        B, T, C = x.shape
        head_size = self.key.out_features

        k = self.key(x)
        q = self.query(x)

        scale_factor = 1 / math.sqrt(head_size)

        attn_weights = (q @ k.transpose(-2, -1)) * scale_factor

        attn_weights = attn_weights.masked_fill(self.tril[:T, :T] == 0, float("-inf"))

        attn_weights = F.softmax(attn_weights, dim=-1)

        v = self.value(x)

        return attn_weights @ v


















