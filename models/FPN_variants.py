import torch
import torch.nn as nn
import torch.nn.functional as F

from models.FPN import FPN, SegmentationHead


class FPNNoLateral_Module(FPN):
    """
    Ablation Study: Remove lateral connections.
    """

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        c1 = self.c1(x)
        c2 = self.c2(c1)
        c3 = self.c3(c2)
        c4 = self.c4(c3)
        c5 = self.c5(c4)

        p5 = self.lat5(c5)
        p4 = F.interpolate(p5, size=c4.shape[-2:], mode="nearest")
        p3 = F.interpolate(p4, size=c3.shape[-2:], mode="nearest")
        p2 = F.interpolate(p3, size=c2.shape[-2:], mode="nearest")

        return {
            "P2": self.out2(p2),
            "P3": self.out3(p3),
            "P4": self.out4(p4),
            "P5": self.out5(p5),
        }


class FPNNoLateral(nn.Module):
    def __init__(self, num_classes: int = 5, out_channels: int = 256) -> None:
        super().__init__()
        self.fpn = FPNNoLateral_Module(out_channels)
        self.head = SegmentationHead(out_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.fpn(x), x.shape[-2:])


class FPNSingleScale(nn.Module):
    """
    Ablation Study: Segmentation head uses only P2 features.
    """

    def __init__(self, num_classes: int = 5, out_channels: int = 256) -> None:
        super().__init__()
        self.fpn = FPN(out_channels)
        self.head = SegmentationHead(out_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_shape = x.shape[-2:]
        features = self.fpn(x)
        p2 = features["P2"]
        logits = self.head.classifier(self.head.conv_block(p2))
        return F.interpolate(
            logits, size=input_shape, mode="bilinear", align_corners=False
        )


class FPNSumMerge(nn.Module):
    """
    Ablation Study: model using summation for merging.
    """

    def __init__(self, num_classes: int = 5, out_channels: int = 256) -> None:
        super().__init__()
        self.fpn = FPN(out_channels)
        self.head = SegmentationHead(out_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_shape = x.shape[-2:]
        return self.head(self.fpn(x), input_shape)


class FPNConcatMerge(nn.Module):
    """
    Ablation Study: Concatenation instead of Summation in the head.
    """

    def __init__(self, num_classes: int = 5, out_channels: int = 256) -> None:
        super().__init__()
        self.fpn = FPN(out_channels)
        self.head = SegmentationHead(out_channels, num_classes)
        self.head.conv_block[0] = nn.Conv2d(
            out_channels * 4, out_channels, kernel_size=3, padding=1, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_shape = x.shape[-2:]
        f = self.fpn(x)
        p2, p3, p4, p5 = f["P2"], f["P3"], f["P4"], f["P5"]

        p3_u = F.interpolate(
            p3, size=p2.shape[-2:], mode="bilinear", align_corners=False
        )
        p4_u = F.interpolate(
            p4, size=p2.shape[-2:], mode="bilinear", align_corners=False
        )
        p5_u = F.interpolate(
            p5, size=p2.shape[-2:], mode="bilinear", align_corners=False
        )

        merged = torch.cat([p2, p3_u, p4_u, p5_u], dim=1)
        x_head = self.head.conv_block(merged)
        logits = self.head.classifier(x_head)
        return F.interpolate(
            logits, size=input_shape, mode="bilinear", align_corners=False
        )


class FPNShallowHead(nn.Module):
    """
    Ablation Study: Simplified head.
    """

    def __init__(self, num_classes: int = 5, out_channels: int = 256) -> None:
        super().__init__()
        self.fpn = FPN(out_channels)
        self.head = SegmentationHead(out_channels, num_classes)
        self.head.conv_block = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.fpn(x), x.shape[-2:])


class FPNDeepHead(nn.Module):
    """
    Ablation Study: Deeper segmentation head.
    """

    def __init__(self, num_classes: int = 5, out_channels: int = 256) -> None:
        super().__init__()
        self.fpn = FPN(out_channels)
        self.head = SegmentationHead(out_channels, num_classes)

        layers = []
        for _ in range(4):
            layers.extend(
                [
                    nn.Conv2d(
                        out_channels, out_channels, kernel_size=3, padding=1, bias=False
                    ),
                    nn.GroupNorm(32, out_channels),
                    nn.ReLU(inplace=True),
                ]
            )
        self.head.conv_block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.fpn(x), x.shape[-2:])


class FPNNoP5_Module(FPN):
    """
    Ablation Study: Remove the highest pyramid level.
    """

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        c1 = self.c1(x)
        c2 = self.c2(c1)
        c3 = self.c3(c2)
        c4 = self.c4(c3)
        p4 = self.lat4(c4)
        p3 = self.lat3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode="nearest")
        p2 = self.lat2(c2) + F.interpolate(p3, size=c2.shape[-2:], mode="nearest")

        return {
            "P2": self.out2(p2),
            "P3": self.out3(p3),
            "P4": self.out4(p4),
            "P5": torch.zeros(p4.size()),
        }


class FPNNoP5(nn.Module):
    def __init__(self, num_classes: int = 5, out_channels: int = 256) -> None:
        super().__init__()
        self.fpn = FPNNoP5_Module(out_channels)
        self.head = SegmentationHead(out_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.fpn(x), x.shape[-2:])
