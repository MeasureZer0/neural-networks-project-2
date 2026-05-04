import torch
import torch.nn as nn
import torch.nn.functional as F


class CELoss(nn.Module):
    def __init__(self) -> None:
        super(CELoss, self).__init__()
        self.ce = nn.CrossEntropyLoss()

    def forward(
        self,
        logits: torch.Tensor,  # [Batch, Classes, H, W]
        targets: torch.Tensor,  # [Batch, H, W]
    ) -> torch.Tensor:
        return self.ce(logits, targets)


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0) -> None:
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(
        self,
        logits: torch.Tensor,  # [Batch, Classes, H, W]
        targets: torch.Tensor,  # [Batch, H, W]
    ) -> torch.Tensor:

        probs = F.softmax(logits, dim=1)
        num_classes = logits.shape[1]

        targets_one_hot = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()

        dims = (0, 2, 3)
        intersection = torch.sum(probs * targets_one_hot, dims)
        cardinality = torch.sum(probs + targets_one_hot, dims)

        dice_score = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1 - dice_score.mean()


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0) -> None:
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(
        self,
        logits: torch.Tensor,  # [Batch, Classes, H, W]
        targets: torch.Tensor,  # [Batch, H, W]
    ) -> torch.Tensor:

        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        return focal_loss.mean()
