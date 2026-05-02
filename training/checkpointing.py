import os
from typing import Any, Optional, Union

import torch
import torch.nn as nn
from torch.amp.grad_scaler import GradScaler
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


def save_checkpoint(
    state: dict[str, Any],
    checkpoint_dir: Union[str, os.PathLike],
    config_name: str = "baseline_config",
    filename: Optional[str] = None,
    is_best: bool = False,
) -> None:
    os.makedirs(checkpoint_dir, exist_ok=True)

    if filename is None:
        epoch = state.get("epoch", 0)
        filename = f"{config_name}_epoch_{epoch}.pth"

    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)

    if is_best:
        best_path = os.path.join(checkpoint_dir, f"{config_name}_best.pth")
        torch.save(state, best_path)
        print(f"New best model saved: {best_path}")


def load_checkpoint(
    checkpoint_path: Union[str, os.PathLike],
    model: nn.Module,
    optimizer: Optional[Optimizer] = None,
    scheduler: Optional[LRScheduler] = None,
    scaler: Optional[GradScaler] = None,
) -> tuple[int, float]:
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    if scaler and "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    return checkpoint["epoch"], checkpoint["val_loss"]
