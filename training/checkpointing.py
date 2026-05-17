import pathlib
import pickle
from os import PathLike, makedirs, path

import torch
import torch.nn as nn
from torch.amp.grad_scaler import GradScaler
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


class _CheckpointCompatUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> object:
        if module in {"pathlib", "pathlib._local"} and name == "WindowsPath":
            return pathlib.PureWindowsPath
        if module in {"pathlib", "pathlib._local"} and name == "PosixPath":
            return pathlib.PurePosixPath
        return super().find_class(module, name)


class _CheckpointCompatPickle:
    Unpickler = _CheckpointCompatUnpickler
    load = pickle.load
    loads = pickle.loads
    dump = pickle.dump
    dumps = pickle.dumps


def normalize_state_dict_keys(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    normalized: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        cleaned = key
        if cleaned.startswith("_orig_mod."):
            cleaned = cleaned.removeprefix("_orig_mod.")
        if cleaned.startswith("module."):
            cleaned = cleaned.removeprefix("module.")
        normalized[cleaned] = value
    return normalized


def load_checkpoint_file(checkpoint_path: str | PathLike[str]) -> dict[str, object]:
    return torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
        pickle_module=_CheckpointCompatPickle,
    )


def save_checkpoint(
    state: dict[str, object],
    checkpoint_dir: str | PathLike[str],
    config_name: str = "baseline_config",
    filename: str | None = None,
    is_best: bool = False,
) -> None:
    makedirs(checkpoint_dir, exist_ok=True)

    if filename is None:
        epoch = state.get("epoch", 0)
        filename = f"{config_name}_epoch_{epoch}.pth"

    filepath = path.join(checkpoint_dir, filename)
    torch.save(state, filepath)

    if is_best:
        best_path = path.join(checkpoint_dir, f"{config_name}_best.pth")
        torch.save(state, best_path)
        print(f"New best model saved: {best_path}")


def load_checkpoint(
    checkpoint_path: str | PathLike[str],
    model: nn.Module,
    optimizer: Optimizer | None = None,
    scheduler: LRScheduler | None = None,
    scaler: GradScaler | None = None,
) -> tuple[int, float]:
    if not path.isfile(checkpoint_path):
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")

    checkpoint = load_checkpoint_file(checkpoint_path)

    model.load_state_dict(normalize_state_dict_keys(checkpoint["model_state_dict"]))

    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    if scaler and "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    return checkpoint["epoch"], checkpoint["val_loss"]
