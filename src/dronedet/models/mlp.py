"""Plan-specified tabular multilayer perceptron."""
from __future__ import annotations

import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, input_dim: int = 84, *, hidden_layers: int = 2,
                 width: int = 64, dropout: float = 0.2):
        super().__init__()
        if input_dim < 1 or hidden_layers < 1 or width < 1 or not 0 <= dropout < 1:
            raise ValueError("invalid MLP dimensions or dropout")
        layers: list[nn.Module] = []
        previous = input_dim
        for _ in range(hidden_layers):
            layers.extend((nn.Linear(previous, width), nn.ReLU(), nn.Dropout(dropout)))
            previous = width
        layers.append(nn.Linear(previous, 1))
        self.network = nn.Sequential(*layers)
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.width = width
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)


def build_mlp(input_dim: int = 84, **kwargs) -> MLP:
    model = MLP(input_dim=input_dim, **kwargs)
    return model


def parameter_count(model: nn.Module) -> int:
    return sum(int(p.numel()) for p in model.parameters() if p.requires_grad)


def mlp_parameter_count(input_dim: int = 84, *, hidden_layers: int = 2, width: int = 64) -> int:
    total = 0
    previous = input_dim
    for _ in range(hidden_layers):
        total += previous * width + width
        previous = width
    return total + previous + 1


def assert_default_parameter_count() -> int:
    count = parameter_count(build_mlp())
    if count != 9665:
        raise AssertionError(f"default MLP parameter count {count} != 9665")
    return count
