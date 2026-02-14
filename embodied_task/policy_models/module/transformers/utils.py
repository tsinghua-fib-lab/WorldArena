import torch
from torch import nn


class SquaredReLU(nn.Module):
    """Squared ReLU activation function"""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.pow(torch.relu(x), 2)


def feed_forward_layer(dim: int, mult: int = 4, activation: str = 'gelu'):
    """Feed forward layer with given activation function"""

    activations = dict(gelu=nn.GELU, sqrelu=SquaredReLU, relu=nn.ReLU)
    assert activation in activations, f'activation can only be one of {activations.keys()}'

    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias=False),
        activations[activation](),
        nn.Linear(inner_dim, dim, bias=False),
    )

# RMSNorm -- Better, simpler alternative to LayerNorm
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-8) -> None:
        super().__init__()
        self.scale, self.eps = dim**-0.5, eps
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.norm(x, dim=-1, keepdim=True) * self.scale
        return x / norm.clamp(min=self.eps) * self.g


# SwishGLU -- A Gated Linear Unit (GLU) with the Swish activation; always better than GELU MLP!
class SwishGLU(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.act, self.project = nn.SiLU(), nn.Linear(in_dim, 2 * out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projected, gate = self.project(x).tensor_split(2, dim=-1)
        return projected * self.act(gate)

