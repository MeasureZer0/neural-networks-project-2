from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet50_Weights, resnet50


class LateralConnection(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class OutputBlock(nn.Module):
    def __init__(self, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class FPN(nn.Module):
    def __init__(self, out_channels: int = 256, pretrained: bool = True) -> None:
        super().__init__()

        weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = resnet50(weights=weights)

        self.c1 = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool
        )
        self.c2 = backbone.layer1
        self.c3 = backbone.layer2
        self.c4 = backbone.layer3
        self.c5 = backbone.layer4

        self.lat5 = LateralConnection(2048, out_channels)
        self.lat4 = LateralConnection(1024, out_channels)
        self.lat3 = LateralConnection(512, out_channels)
        self.lat2 = LateralConnection(256, out_channels)

        self.out5 = OutputBlock(out_channels)
        self.out4 = OutputBlock(out_channels)
        self.out3 = OutputBlock(out_channels)
        self.out2 = OutputBlock(out_channels)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        # Bottom-up
        c1 = self.c1(x)
        c2 = self.c2(c1)
        c3 = self.c3(c2)
        c4 = self.c4(c3)
        c5 = self.c5(c4)

        p5 = self.lat5(c5)

        p4 = self.lat4(c4) + F.interpolate(p5, size=c4.shape[-2:], mode="nearest")
        p3 = self.lat3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode="nearest")
        p2 = self.lat2(c2) + F.interpolate(p3, size=c2.shape[-2:], mode="nearest")

        return {
            "P2": self.out2(p2),
            "P3": self.out3(p3),
            "P4": self.out4(p4),
            "P5": self.out5(p5),
        }


class SegmentationHead(nn.Module):
    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(32, in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(32, in_channels),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def forward(
        self, features: dict[str, torch.Tensor], input_shape: tuple[int, int]
    ) -> torch.Tensor:
        p2 = features["P2"]

        p3_up = F.interpolate(
            features["P3"], size=p2.shape[-2:], mode="bilinear", align_corners=False
        )
        p4_up = F.interpolate(
            features["P4"], size=p2.shape[-2:], mode="bilinear", align_corners=False
        )
        p5_up = F.interpolate(
            features["P5"], size=p2.shape[-2:], mode="bilinear", align_corners=False
        )

        merged = p2 + p3_up + p4_up + p5_up

        x = self.conv_block(merged)
        logits = self.classifier(x)

        return F.interpolate(
            logits, size=input_shape, mode="bilinear", align_corners=False
        )


class FPNSegmentation(nn.Module):
    def __init__(
        self,
        num_classes: int = 5,
        out_channels: int = 256,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.fpn = FPN(out_channels, pretrained=pretrained)
        self.head = SegmentationHead(out_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_shape = x.shape[-2:]
        features = self.fpn(x)
        return self.head(features, input_shape)
