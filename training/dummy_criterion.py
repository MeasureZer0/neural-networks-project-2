# pyright: reportPrivateImportUsage=false
import torch
from torch import nn


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        pred = torch.sigmoid(logits)

        pred = pred.reshape(pred.shape[0], -1)
        targets = targets.reshape(targets.shape[0], -1).float()

        intersection = (pred * targets).sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (
            pred.sum(dim=1) + targets.sum(dim=1) + self.smooth
        )
        return (1.0 - dice).mean()
