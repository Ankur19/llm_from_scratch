import torch
import torch.nn as nn

class Embedding(nn.Module):

    pos_idxs: torch.Tensor

    def __init__(self, vocab_size: int, context_length: int, n_embed: int) -> None:

        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, n_embed)
        self.position_embedding = nn.Embedding(context_length, n_embed)
        self.register_buffer('pos_idxs', torch.arange(context_length))

    def forward(self, idx: torch.Tensor) -> torch.Tensor:

        _, T = idx.size

        tok_embed = self.token_embedding(idx)
        pos_embed = self.position_embedding(self.pos_idxs[:T])

        return tok_embed + pos_embed

if __name__ == "__main__":
    emb = Embedding(vocab_size=100, context_length=16, n_embed=32)
    x = torch.randint(0, 100, (4, 10))
    out = emb(x)
    print(out.shape)  # torch.Size([4, 10, 32])


