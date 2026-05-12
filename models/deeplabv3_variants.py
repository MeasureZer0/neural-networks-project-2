import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50

from models.deeplabv3_model import ASPP, DeepLabV3


class DeepLabV3NoASPP(DeepLabV3):
    """
    Ablation Study: Replace the entire ASPP with a single 1x1 conv + global pooling.
    """

    def __init__(
        self,
        num_classes: int = 5,
        pretrained: bool = True,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__(num_classes, pretrained, freeze_backbone)
        # Replace ASPP with simple projection
        self.aspp = nn.Sequential(
            nn.Conv2d(2048, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]
        feat = self.backbone.conv1(x)
        feat = self.backbone.bn1(feat)
        feat = self.backbone.relu(feat)
        feat = self.backbone.maxpool(feat)
        feat = self.backbone.layer1(feat)
        feat = self.backbone.layer2(feat)
        feat = self.backbone.layer3(feat)
        feat = self.backbone.layer4(feat)

        x = self.aspp(feat)
        x = self.classifier(x)
        return F.interpolate(x, size=input_size, mode="bilinear", align_corners=False)


class DeepLabV3NoGlobalPool(DeepLabV3):
    """
    Ablation Study: Remove the Global Average Pooling branch from ASPP.
    """

    def __init__(
        self,
        num_classes: int = 5,
        pretrained: bool = True,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__(num_classes, pretrained, freeze_backbone)
        self.aspp.project[0] = nn.Conv2d(256 * 4, 256, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]
        feat = self.backbone.conv1(x)
        feat = self.backbone.bn1(feat)
        feat = self.backbone.relu(feat)
        feat = self.backbone.maxpool(feat)
        feat = self.backbone.layer1(feat)
        feat = self.backbone.layer2(feat)
        feat = self.backbone.layer3(feat)
        feat = self.backbone.layer4(feat)

        x1 = self.aspp.branch1(feat)
        x2 = self.aspp.branch2(feat)
        x3 = self.aspp.branch3(feat)
        x4 = self.aspp.branch4(feat)

        x_aspp = torch.cat([x1, x2, x3, x4], dim=1)
        x_aspp = self.aspp.project(x_aspp)

        x_out = self.classifier(x_aspp)
        return F.interpolate(
            x_out, size=input_size, mode="bilinear", align_corners=False
        )


class DeepLabV3NarrowASPP(DeepLabV3):
    """
    Ablation Study: Reduce ASPP internal channels from 256 to 128.
    """

    def __init__(
        self,
        num_classes: int = 5,
        pretrained: bool = True,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__(num_classes, pretrained, freeze_backbone)
        self.aspp = ASPP(in_channels=2048, out_channels=128)
        self.classifier[0] = nn.Conv2d(128, 128, 3, padding=1, bias=False)
        self.classifier[1] = nn.BatchNorm2d(128)
        self.classifier[3] = nn.Conv2d(128, num_classes, 1)


class DeepLabV3SmallDilations(DeepLabV3):
    """
    Ablation Study: Smaller ASPP dilations [3, 6, 9] instead of [6, 12, 18].
    """

    def __init__(
        self,
        num_classes: int = 5,
        pretrained: bool = True,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__(num_classes, pretrained, freeze_backbone)
        self.aspp.branch2[0].dilation = (3, 3)
        self.aspp.branch2[0].padding = (3, 3)
        self.aspp.branch3[0].dilation = (6, 6)
        self.aspp.branch3[0].padding = (6, 6)
        self.aspp.branch4[0].dilation = (9, 9)
        self.aspp.branch4[0].padding = (9, 9)


class DeepLabV3LargeDilations(DeepLabV3):
    """
    Ablation Study: Larger ASPP dilations [12, 24, 36].
    """

    def __init__(
        self,
        num_classes: int = 5,
        pretrained: bool = True,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__(num_classes, pretrained, freeze_backbone)
        self.aspp.branch2[0].dilation = (12, 12)
        self.aspp.branch2[0].padding = (12, 12)
        self.aspp.branch3[0].dilation = (24, 24)
        self.aspp.branch3[0].padding = (24, 24)
        self.aspp.branch4[0].dilation = (36, 36)
        self.aspp.branch4[0].padding = (36, 36)


class DeepLabV3ShallowHead(DeepLabV3):
    """
    Ablation Study: Simplified segmentation head.
    """

    def __init__(
        self,
        num_classes: int = 5,
        pretrained: bool = True,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__(num_classes, pretrained, freeze_backbone)
        # Reduce classifier to a single 1x1 conv
        self.classifier = nn.Conv2d(256, num_classes, 1)


class DeepLabV3Dropout(DeepLabV3):
    """
    Ablation Study: Add additional Dropout(p=0.5) before the final classifier.
    """

    def __init__(
        self,
        num_classes: int = 5,
        pretrained: bool = True,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__(num_classes, pretrained, freeze_backbone)
        # Insert Dropout2d before the last layer
        new_classifier = nn.Sequential(
            self.classifier[0],
            self.classifier[1],
            self.classifier[2],
            nn.Dropout2d(p=0.5),
            self.classifier[3],
        )
        self.classifier = new_classifier


class DeepLabV3Cascaded(nn.Module):
    """
    Ablation Study: Cascaded Atrous Convolutions.
    """

    def __init__(
        self,
        num_classes: int = 5,
        pretrained: bool = True,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()
        self.backbone = resnet50(
            weights="IMAGENET1K_V1" if pretrained else None,
            replace_stride_with_dilation=[False, True, True],
        )

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            for m in self.backbone.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    m.requires_grad_(False)

        self.classifier = nn.Sequential(
            nn.Conv2d(2048, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, num_classes, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]

        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = self.classifier(x)
        return F.interpolate(x, size=input_size, mode="bilinear", align_corners=False)
