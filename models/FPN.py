from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet50_Weights, resnet50


class LateralBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.bn(self.conv(x)))


class OutputBlock(nn.Module):
    def __init__(self, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.bn(self.conv(x)))


class FPN(nn.Module):
    def __init__(
        self,
        out_channels: int = 256,
        pretrained: bool = True,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()

        self.out_channels = out_channels

        # example from paper with resnet50
        weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = resnet50(weights=weights)

        self.c1 = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool
        )
        self.c2 = backbone.layer1
        self.c3 = backbone.layer2
        self.c4 = backbone.layer3
        self.c5 = backbone.layer4

        if freeze_backbone:
            for param in [
                *self.c1.parameters(),
                *self.c2.parameters(),
                *self.c3.parameters(),
                *self.c4.parameters(),
                *self.c5.parameters(),
            ]:
                param.requires_grad = False

        self.lat5 = LateralBlock(2048, out_channels)
        self.lat4 = LateralBlock(1024, out_channels)
        self.lat3 = LateralBlock(512, out_channels)
        self.lat2 = LateralBlock(256, out_channels)

        self.out5 = OutputBlock(out_channels)
        self.out4 = OutputBlock(out_channels)
        self.out3 = OutputBlock(out_channels)
        self.out2 = OutputBlock(out_channels)

        self._init_weights()

    def _init_weights(self) -> None:
        for module in [
            self.lat5,
            self.lat4,
            self.lat3,
            self.lat2,
            self.out5,
            self.out4,
            self.out3,
            self.out2,
        ]:
            for m in module.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(
                        m.weight, mode="fan_out", nonlinearity="relu"
                    )
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1.0)
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        c1 = self.c1(x)
        c2 = self.c2(c1)
        c3 = self.c3(c2)
        c4 = self.c4(c3)
        c5 = self.c5(c4)

        m5 = self.lat5(c5)
        m4 = self.lat4(c4)
        m3 = self.lat3(c3)
        m2 = self.lat2(c2)

        m4 = m4 + F.interpolate(m5, size=m4.shape[-2:], mode="nearest")
        m3 = m3 + F.interpolate(m4, size=m3.shape[-2:], mode="nearest")
        m2 = m2 + F.interpolate(m3, size=m2.shape[-2:], mode="nearest")

        p5 = self.out5(m5)
        p4 = self.out4(m4)
        p3 = self.out3(m3)
        p2 = self.out2(m2)

        out: dict[str, torch.Tensor] = {
            "P2": p2,
            "P3": p3,
            "P4": p4,
            "P5": p5,
        }

        return out
