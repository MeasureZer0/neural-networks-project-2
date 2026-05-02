import os
from typing import Optional

import torch
import torch.nn as nn
from torch.amp.grad_scaler import GradScaler
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from training.checkpointing import save_checkpoint
from training.configs.baseline import Config


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        criterion: nn.Module,
        scheduler: Optional[LRScheduler],
        device: str,
        config: Config,
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

        if getattr(config, "compile_model", False) and device == "cuda":
            self.model = torch.compile(self.model, mode="reduce-overhead")

        self.checkpoint_dir = getattr(config, "checkpoint_dir", "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.use_wandb = getattr(config, "use_wandb", False)
        if self.use_wandb:
            import wandb

            self.wandb = wandb
            if self.wandb.run is None:
                self.wandb.init(
                    project=getattr(config, "wandb_project", "semantic-segmentation"),
                    config=vars(config),
                )

    def _batch_to_device(
        self, batch: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        return {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}

    def train_epoch(self, dataloader: DataLoader, epoch: int) -> float:
        self.model.train()  # type: ignore[assignment]
        total_loss = 0.0
        n_batches = len(dataloader)
        pbar = tqdm(enumerate(dataloader), total=n_batches, desc=f"Epoch {epoch}")
        for batch_idx, batch in pbar:
            batch = self._batch_to_device(batch)
            images, masks = batch["image"], batch["mask"]

            # PyTorch optimization: Set grad to None instead of zero for efficiency
            self.optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=self.device, dtype=self.dtype):
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
            if self.precision == "fp16" and self.device == "cuda":
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            if self.scheduler is not None:
                self.scheduler.step()
            total_loss += loss.item()
            pbar.set_postfix({"loss": total_loss / (batch_idx + 1)})
        avg_loss = total_loss / n_batches

        if self.use_wandb:
            self.wandb.log({"train_loss": avg_loss}, step=epoch)
        return avg_loss

    @torch.no_grad()
    def validate_epoch(self, dataloader: DataLoader, epoch: int) -> float:
        self.model.eval()  # type: ignore[assignment]
        total_loss = 0.0
        n_batches = len(dataloader)
        pbar = tqdm(enumerate(dataloader), total=n_batches, desc=f"Validate {epoch}")
        for batch_idx, batch in pbar:
            batch = self._batch_to_device(batch)
            images, masks = batch["image"], batch["mask"]

            with torch.autocast(device_type=self.device, dtype=self.dtype):
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
            total_loss += loss.item()
            pbar.set_postfix({"val_loss": total_loss / (batch_idx + 1)})
        avg_loss = total_loss / n_batches

        if self.use_wandb:
            self.wandb.log({"val_loss": avg_loss}, step=epoch)
        return avg_loss

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> None:
        num_epochs = getattr(self.config, "num_epochs", 10)
        best_val_loss = float("inf")
        config_name = getattr(self.config, "name", "baseline_config")

        for epoch in range(self.start_epoch, num_epochs + 1):
            train_loss = self.train_epoch(train_loader, epoch)
            val_loss = self.validate_epoch(val_loader, epoch)

            print(
                f"Epoch {epoch}: Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
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
