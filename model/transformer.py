import tensor
import tensor.nn as nn
import tensor.nn.functional as F

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


class Transformer(nn.Module):

    def __init__(self, n_embed: int, n_heads: int, context_length: int, vocab_size: int, n_blocks: int) -> None:

        super().__init__()

        self.context_length = context_length
        self.n_blocks = n_blocks

        self.embedding = Embedding(vocab_size, context_length, n_embed)

        self.attn_blocks = nn.ModuleList([Block(n_heads, n_embed, context_length) for _ in range(n_blocks)])

        self.ln = nn.LayerNorm(n_embed)

        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        x = self.embedding(idx)

        for block in self.attn_blocks:
            x = block(x)
        x = self.ln(x)

        x = self.lm_head(x)

        loss = None

        if targets is not None:
            B, T, C = x.shape

            flat_logits = x.reshape(B * T, C)

            targets = targets.reshape(B * T).long()

            loss = F.cross_entropy(flat_logits, targets)

        return x, loss

