import tensor
import tensor.nn as nn

from models.mlp import MLP
from models.attention import MultiHeadAttention
from models.embedding import Embedding

class Block(nn.Module):

    def __init__(self, n_heads: int, n_embed: int, context_length: int) -> None:

        super().__init__()
        self.ln1 = nn.LayerNorm(n_embed)
        self.attn = MultiHeadAttention(n_embed, n_heads, context_length)
        self.ln2 = nn.LayerNorm(n_embed)
        self.mlp = MLP(n_embed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = x + self.attn(self.ln1(x))

        x2 = x1 + self.mlp(self.ln2(x1))

        return x2


