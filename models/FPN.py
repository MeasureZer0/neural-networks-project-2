from __future__ import annotations

import torch
import torch.nn as nn
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
        self.c2 = backbone.layer1  # 256
        self.c3 = backbone.layer2  # 512
        self.c4 = backbone.layer3  # 1024
        self.c5 = backbone.layer4  # 2048

        self.lat5 = LateralConnection(2048, out_channels)
        self.lat4 = LateralConnection(1024, out_channels)
        self.lat3 = LateralConnection(512, out_channels)
        self.lat2 = LateralConnection(256, out_channels)

        self.up5to4 = nn.ConvTranspose2d(
            out_channels, out_channels, kernel_size=4, stride=2, padding=1
        )
        self.up4to3 = nn.ConvTranspose2d(
            out_channels, out_channels, kernel_size=4, stride=2, padding=1
        )
        self.up3to2 = nn.ConvTranspose2d(
            out_channels, out_channels, kernel_size=4, stride=2, padding=1
        )

        self.out5 = OutputBlock(out_channels)
        self.out4 = OutputBlock(out_channels)
        self.out3 = OutputBlock(out_channels)
        self.out2 = OutputBlock(out_channels)

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

        m4 = m4 + self.up5to4(m5)
        m3 = m3 + self.up4to3(m4)
        m2 = m2 + self.up3to2(m3)

        return {
            "P2": self.out2(m2),
            "P3": self.out3(m3),
            "P4": self.out4(m4),
            "P5": self.out5(m5),
        }


class SegmentationHead(nn.Module):
    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__()

        self.upP3 = nn.ConvTranspose2d(
            in_channels, in_channels, kernel_size=4, stride=2, padding=1
        )
        self.upP4 = nn.ConvTranspose2d(
            in_channels, in_channels, kernel_size=8, stride=4, padding=2
        )
        self.upP5 = nn.ConvTranspose2d(
            in_channels, in_channels, kernel_size=16, stride=8, padding=4
        )

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Conv2d(in_channels, num_classes, kernel_size=1)

        self.final_up = nn.ConvTranspose2d(
            num_classes, num_classes, kernel_size=8, stride=4, padding=2
        )

    def forward(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        p2 = features["P2"]
        p3_up = self.upP3(features["P3"])
        p4_up = self.upP4(features["P4"])
        p5_up = self.upP5(features["P5"])

        merged = p2 + p3_up + p4_up + p5_up

        x = self.conv_block(merged)
        logits = self.classifier(x)

        return self.final_up(logits)


class FPNSegmentation(nn.Module):
    def __init__(self, num_classes: int = 5, out_channels: int = 256) -> None:
        super().__init__()
        self.fpn = FPN(out_channels)
        self.head = SegmentationHead(out_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.fpn(x)
        return self.head(features)
