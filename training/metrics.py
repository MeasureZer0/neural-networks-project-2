from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader


class SegmentationMetrics:
    def __init__(
        self,
        num_classes: int,
        background_index: Optional[int] = None,
        device: str = "cpu",
    ) -> None:
        self.num_classes = num_classes
        self.background_index = background_index
        self.device = device

        # Global confusion matrix: shape [C, C]
        self.confusion = torch.zeros(
            (num_classes, num_classes), dtype=torch.int64, device=device
        )

    def _prepare_labels(self, logits: Tensor, labels: Tensor) -> tuple[Tensor, Tensor]:
        """
        logits: [B, C, H, W]
        labels: [B, H, W] (int) or [B, C, H, W] (one-hot)
        """

        # Convert logits to predicted class indices
        preds = torch.argmax(logits, dim=1)  # [B, H, W]

        # Convert one-hot labels to integer labels if needed
        if labels.ndim == 4:
            labels = torch.argmax(labels, dim=1)

        return preds, labels

    @torch.no_grad()
    def update(self, logits: Tensor, labels: Tensor) -> None:
        preds, labels = self._prepare_labels(logits, labels)

        # Flatten
        preds = preds.view(-1)
        labels = labels.view(-1)

        # Compute confusion matrix for this batch

        indices = self.num_classes * labels + preds
        batch_confusion = torch.bincount(
            indices, minlength=self.num_classes**2
        ).reshape(self.num_classes, self.num_classes)

        self.confusion += batch_confusion

    def per_class_iou(self) -> Tensor:
        TP = torch.diag(self.confusion)
        FP = self.confusion.sum(dim=0) - TP
        FN = self.confusion.sum(dim=1) - TP
        denom = TP + FP + FN
        iou = TP / torch.clamp(denom, min=1)
        return iou

    def mean_iou(self, exclude_background: bool = True) -> float:
        iou = self.per_class_iou()
        if exclude_background and self.background_index is not None:
            iou = torch.cat(
                [iou[: self.background_index], iou[self.background_index + 1 :]]
            )
        return float(iou.mean().item())

    def per_class_dice(self) -> Tensor:
        TP = torch.diag(self.confusion)
        FP = self.confusion.sum(dim=0) - TP
        FN = self.confusion.sum(dim=1) - TP
        dice = 2 * TP / torch.clamp(2 * TP + FP + FN, min=1)
        return dice

    def mean_dice(self, exclude_background: bool = True) -> float:
        dice = self.per_class_dice()
        if exclude_background and self.background_index is not None:
            dice = torch.cat(
                [dice[: self.background_index], dice[self.background_index + 1 :]]
            )
        return float(dice.mean().item())

    def pixel_accuracy(self) -> float:
        correct = torch.diag(self.confusion).sum()
        total = self.confusion.sum()
        return float((correct / torch.clamp(total, min=1)).item())


@torch.no_grad()
def full_segmentation_eval(
    model: Module,
    dataloader: DataLoader,
    device: str,
    num_classes: int,
    background_index: Optional[int] = None,
    use_fp16: bool = False,
) -> Tuple[
    Dict[str, float],  # overall results
    Dict[str, List[float]],  # per-class results
]:

    model.eval()
    metrics = SegmentationMetrics(
        num_classes=num_classes, background_index=background_index, device=device
    )

    for batch in dataloader:
        images = batch["images"].to(device)
        labels = batch["labels"].to(device)

        with torch.cuda.amp.autocast(enabled=use_fp16):
            logits = model(images)

        metrics.update(logits, labels)

    overall_results: Dict[str, float] = {
        "overall_pixel_accuracy": metrics.pixel_accuracy(),
        "mean_iou": metrics.mean_iou(),
        "mean_dice": metrics.mean_dice(),
    }

    per_class_results: Dict[str, List[float]] = {
        "iou": metrics.per_class_iou().cpu().tolist(),
        "dice": metrics.per_class_dice().cpu().tolist(),
    }

    return overall_results, per_class_results
