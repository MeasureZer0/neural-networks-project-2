import torch
import torch.nn as nn


class LinearBaseline(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 1) -> None:
        super().__init__()
        self.head = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)
