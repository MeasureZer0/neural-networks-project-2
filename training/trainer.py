import os
from copy import deepcopy
from itertools import cycle
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp.grad_scaler import GradScaler
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from training.checkpointing import save_checkpoint
from training.configs.baseline import BaselineConfig
from training.metrics import SegmentationMetrics


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        criterion: nn.Module,
        scheduler: Optional[LRScheduler],
        device: str,
        config: BaselineConfig,
        start_epoch: int = 0,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.device = device
        self.config = config
        self.start_epoch = start_epoch

        self.precision = getattr(config, "precision", "fp32")

        precision_map = {
            "fp32": torch.float32,  # type: ignore[assignment]
            "fp16": torch.float16,  # type: ignore[assignment]
            "bf16": torch.bfloat16,  # type: ignore[assignment]
        }

        self.dtype = precision_map.get(self.precision, torch.float32)  # type: ignore[assignment]

        use_scaler = self.precision == "fp16" and device == "cuda"
        self.scaler = GradScaler(enabled=use_scaler)

        self.use_channels_last = (
            getattr(config, "channels_last", False) and device == "cuda"
        )
        if self.use_channels_last:
            self.model = self.model.to(memory_format=torch.channels_last)  # type: ignore[call-overload]

        self.semi_supervised = getattr(config, "semi_supervised", None)
        self.use_ssl = getattr(self.semi_supervised, "enabled", False)
        self.teacher: nn.Module | None = None
        self.global_step = 0
        if self.use_ssl:
            self.teacher = deepcopy(self.model)
            self.teacher.eval()
            for param in self.teacher.parameters():
                param.requires_grad_(False)

        if getattr(config, "compile_model", False) and device == "cuda":
            self.model = torch.compile(self.model, mode="reduce-overhead")

        self.checkpoint_dir = getattr(config, "checkpoint_dir", "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.num_classes: int = getattr(config, "num_classes", 5)

        self.use_wandb = getattr(config, "use_wandb", False)
        if self.use_wandb:
            import wandb

            self.wandb = wandb
            if self.wandb.run is None:
                self.wandb.init(
                    project=getattr(config, "wandb_project", "semantic-segmentation"),
                    entity=getattr(config, "wandb_entity", None),
                    name=getattr(config, "name", None),
                    group=getattr(config, "wandb_group", None),
                    tags=getattr(config, "wandb_tags", None) or [],
                    config=vars(config),
                )

    def _batch_to_device(
        self, batch: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor | list[str]]:
        return {
            k: v.to(self.device, non_blocking=True) if torch.is_tensor(v) else v
            for k, v in batch.items()
        }

    def _current_lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]

    @torch.no_grad()
    def update_ema(self) -> None:
        if self.teacher is None:
            return

        max_decay = getattr(self.semi_supervised, "ema_decay", 0.996)
        decay = min(1 - 1 / (self.global_step + 1), max_decay)

        for teacher_param, student_param in zip(
            self.teacher.parameters(), self.model.parameters(), strict=False
        ):
            teacher_param.data.mul_(decay).add_(student_param.data, alpha=1 - decay)

        for teacher_buffer, student_buffer in zip(
            self.teacher.buffers(), self.model.buffers(), strict=False
        ):
            teacher_buffer.copy_(student_buffer)

    def masked_cross_entropy(
        self,
        student_logits: torch.Tensor,
        pseudo_mask: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        loss_u_all = F.cross_entropy(student_logits, pseudo_mask, reduction="none")
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=self.device)
        return loss_u_all[valid_mask].mean()

    def unsupervised_loss(
        self, unlabeled_batch: dict[str, torch.Tensor | list[str]]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.teacher is None:
            raise RuntimeError("SSL teacher is not initialized.")

        image_weak = unlabeled_batch["image_weak"]
        if not torch.is_tensor(image_weak):
            raise TypeError("image_weak must be a tensor.")
        image_weak = (
            image_weak.to(memory_format=torch.channels_last)
            if self.use_channels_last
            else image_weak
        )  # type: ignore[assignment]

        with torch.no_grad():
            teacher_logits = self.teacher(image_weak)
            teacher_probs = torch.softmax(teacher_logits, dim=1)
            pseudo_conf, pseudo_mask = teacher_probs.max(dim=1)
            threshold = getattr(self.semi_supervised, "threshold", 0.95)
            valid_mask = pseudo_conf >= threshold

        strong_keys = ["image_strong"]
        if getattr(self.semi_supervised, "use_dual_strong_views", False):
            strong_keys = ["image_strong_1", "image_strong_2"]

        losses = []
        for key in strong_keys:
            image_strong = unlabeled_batch[key]
            if not torch.is_tensor(image_strong):
                raise TypeError(f"{key} must be a tensor.")
            if self.use_channels_last:
                image_strong = image_strong.to(memory_format=torch.channels_last)  # type: ignore[assignment]
            student_logits = self.model(image_strong)
            losses.append(
                self.masked_cross_entropy(student_logits, pseudo_mask, valid_mask)
            )

        loss_u = torch.stack(losses).mean()
        return loss_u, valid_mask, pseudo_mask

    def train_epoch(
        self,
        dataloader: DataLoader,
        epoch: int,
        unlabeled_loader: DataLoader | None = None,
    ) -> dict[str, float]:
        self.model.train()  # type: ignore[assignment]
        if self.teacher is not None:
            self.teacher.eval()
        total_loss = 0.0
        total_loss_sup = 0.0
        total_loss_unsup = 0.0
        total_valid_ratio = 0.0
        pseudo_hist: torch.Tensor | None = None
        n_batches = len(dataloader)
        unlabeled_iter = (
            cycle(unlabeled_loader) if unlabeled_loader is not None else None
        )
        pbar = tqdm(enumerate(dataloader), total=n_batches, desc=f"Epoch {epoch}")
        for batch_idx, batch in pbar:
            batch = self._batch_to_device(batch)
            images, masks = batch["image"], batch["mask"]
            if not torch.is_tensor(images) or not torch.is_tensor(masks):
                raise TypeError("Supervised batch must contain tensor image and mask.")
            if self.use_channels_last:
                images = images.to(memory_format=torch.channels_last)  # type: ignore[assignment]

            # PyTorch optimization: Set grad to None instead of zero for efficiency
            self.optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=self.device, dtype=self.dtype):
                outputs = self.model(images)
                loss_sup = self.criterion(outputs, masks)
                loss_unsup = torch.tensor(0.0, device=self.device)
                valid_ratio = torch.tensor(0.0, device=self.device)

                if self.use_ssl and unlabeled_iter is not None:
                    unlabeled_batch = self._batch_to_device(next(unlabeled_iter))
                    loss_unsup, valid_mask, pseudo_mask = self.unsupervised_loss(
                        unlabeled_batch
                    )
                    valid_ratio = valid_mask.float().mean()
                    hist = torch.bincount(
                        pseudo_mask[valid_mask].flatten(),
                        minlength=outputs.shape[1],
                    ).detach()
                    pseudo_hist = hist if pseudo_hist is None else pseudo_hist + hist

                lambda_u = getattr(self.semi_supervised, "lambda_u", 1.0)
                warmup_epochs = getattr(self.semi_supervised, "unsup_warmup_epochs", 0)
                if warmup_epochs > 0:
                    lambda_current = lambda_u * min(1.0, epoch / warmup_epochs)
                else:
                    lambda_current = lambda_u
                loss = loss_sup + lambda_current * loss_unsup
            if self.precision == "fp16" and self.device == "cuda":
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
            if self.use_ssl:
                self.update_ema()
            self.global_step += 1

            if self.scheduler is not None:
                self.scheduler.step()
            total_loss += loss.item()
            total_loss_sup += loss_sup.item()
            total_loss_unsup += loss_unsup.item()
            total_valid_ratio += valid_ratio.item()
            pbar.set_postfix({"loss": total_loss / (batch_idx + 1)})
        avg_loss = total_loss / n_batches
        metrics = {
            "loss_total": avg_loss,
            "loss_sup": total_loss_sup / n_batches,
            "loss_unsup": total_loss_unsup / n_batches,
            "unsup_valid_pixel_ratio": total_valid_ratio / n_batches,
        }

        if self.use_wandb:
            log_payload = {
                "train_loss": metrics["loss_total"],
                "train/loss": metrics["loss_total"],
                "train/loss_total": metrics["loss_total"],
                "train/loss_sup": metrics["loss_sup"],
                "train/loss_unsup": metrics["loss_unsup"],
                "train/unsup_valid_pixel_ratio": metrics["unsup_valid_pixel_ratio"],
                "train/lr": self._current_lr(),
                "epoch": epoch,
            }
            if pseudo_hist is not None:
                log_payload["train/pseudo_class_histogram"] = self.wandb.Histogram(
                    pseudo_hist.detach().cpu().numpy()
                )
            self.wandb.log(log_payload)
        return metrics

    @torch.no_grad()
    def validate_epoch(self, dataloader: DataLoader, epoch: int) -> dict[str, float]:
        self.model.eval()  # type: ignore[assignment]
        total_loss = 0.0
        n_batches = len(dataloader)
        metrics = SegmentationMetrics(
            num_classes=self.num_classes,
            background_index=0,
            device=self.device,
        )
        pbar = tqdm(enumerate(dataloader), total=n_batches, desc=f"Validate {epoch}")
        for batch_idx, batch in pbar:
            batch = self._batch_to_device(batch)
            images, masks = batch["image"], batch["mask"]
            if not torch.is_tensor(images) or not torch.is_tensor(masks):
                raise TypeError("Validation batch must contain tensor image and mask.")
            if self.use_channels_last:
                images = images.to(memory_format=torch.channels_last)  # type: ignore[assignment]

            with torch.autocast(device_type=self.device, dtype=self.dtype):
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
            total_loss += loss.item()
            metrics.update(outputs, masks)
            pbar.set_postfix({"val_loss": total_loss / (batch_idx + 1)})

        avg_loss = total_loss / n_batches
        mean_iou = metrics.mean_iou(exclude_background=True)
        mean_dice = metrics.mean_dice(exclude_background=True)
        pixel_acc = metrics.pixel_accuracy()
        per_class_iou = metrics.per_class_iou().cpu().tolist()
        per_class_dice = metrics.per_class_dice().cpu().tolist()

        print(
            f"  val_loss={avg_loss:.4f}  mIoU={mean_iou:.4f}"
            f"  mDice={mean_dice:.4f}  px_acc={pixel_acc:.4f}"
        )

        if self.use_wandb:
            log_dict: dict = {
                "val/loss": avg_loss,
                "val/mean_iou": mean_iou,
                "val/mean_dice": mean_dice,
                "val/pixel_accuracy": pixel_acc,
                "epoch": epoch,
            }
            for i, (iou, dice) in enumerate(
                zip(per_class_iou, per_class_dice, strict=False)
            ):
                log_dict[f"val/iou_class_{i}"] = iou
                log_dict[f"val/dice_class_{i}"] = dice

            self.wandb.log(log_dict)
        val_metrics = {
            "loss": avg_loss,
            "mIoU": mean_iou,
            "mean_dice": mean_dice,
            "pixel_accuracy": pixel_acc,
        }
        for i, iou in enumerate(per_class_iou):
            val_metrics[f"class_iou/{i}"] = iou
        for i, dice in enumerate(per_class_dice):
            val_metrics[f"class_dice/{i}"] = dice
        return val_metrics

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        unlabeled_loader: DataLoader | None = None,
    ) -> None:
        num_epochs = getattr(self.config, "num_epochs", 10)
        best_val_loss = float("inf")
        config_name = getattr(self.config, "name", "baseline_config")

        for epoch in range(self.start_epoch, num_epochs + 1):
            train_metrics = self.train_epoch(train_loader, epoch, unlabeled_loader)
            train_loss = train_metrics["loss_total"]
            val_metrics = self.validate_epoch(val_loader, epoch)
            val_loss = val_metrics["loss"]

            if self.use_ssl:
                print(
                    f"Epoch {epoch}: Train Loss: {train_loss:.4f} | "
                    f"Sup: {train_metrics['loss_sup']:.4f} | "
                    f"Unsup: {train_metrics['loss_unsup']:.4f} | "
                    f"Val Loss: {val_loss:.4f} | "
                    f"Val mIoU: {val_metrics.get('mIoU', 0.0):.4f}"
                )
            else:
                print(
                    f"Epoch {epoch}: Train Loss: {train_loss:.4f} | "
                    f"Val Loss: {val_loss:.4f} | "
                    f"Val mIoU: {val_metrics.get('mIoU', 0.0):.4f}"
                )

            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss

            state = {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),  # type: ignore[assignment]
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict()
                if self.scheduler
                else None,
                "scaler_state_dict": self.scaler.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "config": self.config,
            }

            save_checkpoint(
                state=state,
                checkpoint_dir=self.checkpoint_dir,
                config_name=config_name,
                is_best=is_best,
            )
        if self.use_wandb and self.wandb.run is not None:
            self.wandb.finish()
