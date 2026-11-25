import torch
import torch.nn as nn

class MLP(nn.Module):

    def __init__(self, n_embed: int) -> None:
        super().__init__()

        self.scale_up = nn.Linear(n_embed, 4 * n_embed)
        self.scale_down = nn.Linear(4 * n_embed, n_embed)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.scale_up(x)

        x = self.activation(x)
        
        x = self.scale_down(x)

        return x

if __name__=="__main__":
    n_embed = 4
    batch_size = 4
    context_length = 32

    x = torch.randn(batch_size, context_length, n_embed)

    mlp = MLP(n_embed)

    out = mlp(x)
    
    assert out.shape, x.shape

