"""Shape-driven CNN architectures for the detection-model stages."""
from __future__ import annotations

import torch
from torch import nn


class DroneCNN(nn.Module):
    def __init__(self, input_shape: tuple[int, int, int] = (1, 64, 28), *,
                 channels: tuple[int, ...] = (16, 32, 64),
                 kernel_size: int = 3, pool: tuple[int, int] = (2, 2),
                 head: str = "global_average", dropout: float = 0.3):
        super().__init__()
        if len(input_shape) != 3 or len(channels) < 1 or kernel_size % 2 == 0:
            raise ValueError("CNN needs (channels,height,width), at least one block and an odd kernel")
        if head not in {"global_average", "flatten"} or not 0 <= dropout < 1:
            raise ValueError("invalid CNN head or dropout")
        padding = kernel_size // 2
        blocks: list[nn.Module] = []
        in_channels = input_shape[0]
        for i, out_channels in enumerate(channels):
            block_kernel = kernel_size if i == 0 else 3
            block_padding = block_kernel // 2
            blocks.extend((nn.Conv2d(in_channels, out_channels, block_kernel,
                                     padding=block_padding),
                           nn.BatchNorm2d(out_channels), nn.ReLU()))
            if i < len(channels) - 1:
                blocks.append(nn.MaxPool2d(pool))
            in_channels = out_channels
        self.features = nn.Sequential(*blocks)
        self.head_kind = head
        self.dropout = nn.Dropout(dropout)
        with torch.no_grad():
            sample = torch.zeros((1, *input_shape), dtype=torch.float32)
            feature_shape = self.features(sample).shape[1:]
        if head == "global_average":
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            head_features = channels[-1]
        else:
            self.pool = nn.Identity()
            head_features = int(torch.tensor(feature_shape).prod().item())
        self.classifier = nn.Linear(head_features, 1)
        self.input_shape = input_shape
        self.channels = channels
        self.kernel_size = kernel_size
        self.pool_shape = pool

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.dropout(x)
        return self.classifier(x).squeeze(-1)


def build_cnn(input_shape: tuple[int, int, int] = (1, 64, 28), **kwargs) -> DroneCNN:
    return DroneCNN(input_shape=input_shape, **kwargs)


def parameter_count(model: nn.Module) -> int:
    return sum(int(p.numel()) for p in model.parameters() if p.requires_grad)


def assert_parameter_counts(input_shape: tuple[int, int, int]) -> dict[str, int]:
    counts = {
        "default": parameter_count(build_cnn(input_shape)),
        "widths_8_16_32": parameter_count(build_cnn(input_shape, channels=(8, 16, 32))),
        "widths_32_64_128": parameter_count(build_cnn(input_shape, channels=(32, 64, 128))),
        "depth_2": parameter_count(build_cnn(input_shape, channels=(16, 32))),
        "depth_4": parameter_count(build_cnn(input_shape, channels=(16, 32, 64, 64))),
        "kernel_5": parameter_count(build_cnn(input_shape, kernel_size=5)),
    }
    expected = {"default": 23585, "widths_8_16_32": 6033,
                "widths_32_64_128": 93249, "depth_2": 4929,
                "depth_4": 60641, "kernel_5": 23841}
    for name, value in expected.items():
        if counts[name] != value:
            raise AssertionError(f"CNN {name} parameter count {counts[name]} != {value}")
    if input_shape == (1, 64, 28):
        flatten = parameter_count(build_cnn(input_shape, head="flatten"))
        if flatten != 30689:
            raise AssertionError(f"CNN flatten parameter count {flatten} != 30689")
        counts["flatten"] = flatten
    else:
        counts["flatten"] = parameter_count(build_cnn(input_shape, head="flatten"))
    return counts
