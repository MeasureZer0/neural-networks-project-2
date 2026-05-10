import math
import os
from pathlib import Path

import optuna
import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from models.deeplabv3_model import DeepLabV3
from models.FPN import FPNSegmentation
from models.UNet import UNet
from torch_datasets.landcover_dataset import LandcoverDataset
from torch_datasets.transforms import TrainTransform, ValTransform
from training.configs.baseline import BaselineConfig
from training.loss import CELoss, DiceLoss, FocalLoss
from training.trainer import Trainer

MODEL_NAME = "fpn"
LOSS_NAME = "dice"
N_TRIALS = 50
N_EPOCHS_TUNE = 5
TIMEOUT = None
STUDY_NAME = None
STORAGE = "sqlite:///optuna.db"

MODELS = {"fpn": FPNSegmentation, "unet": UNet, "deeplabv3": DeepLabV3}
LOSSES = {"dice": DiceLoss, "ce": CELoss, "focal": FocalLoss}

CHECKPOINT_BASE = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "checkpoints", "tune"
)


def cosine_schedule(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
) -> LambdaLR:
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


def objective(
    trial: optuna.Trial,
    base_config: BaselineConfig,
    train_loader: DataLoader,
    val_loader: DataLoader,
) -> float:
    lr = trial.suggest_float("lr", 1e-5, 5e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    beta1 = trial.suggest_float("adam_beta1", 0.85, 0.95)
    beta2 = trial.suggest_float("adam_beta2", 0.95, 0.999)
    use_cosine = trial.suggest_categorical("use_cosine_schedule", [True, False])
    warmup_epochs = trial.suggest_int("warmup_epochs", 0, 3) if use_cosine else 0

    config = BaselineConfig()
    config.__dict__.update(base_config.__dict__)
    config.lr = lr
    config.weight_decay = weight_decay
    config.use_wandb = False
    config.checkpoint_dir = os.path.join(CHECKPOINT_BASE, f"trial_{trial.number}")

    if LOSS_NAME == "focal":
        criterion: torch.nn.Module = FocalLoss(
            alpha=trial.suggest_float("focal_alpha", 0.25, 2.0),
            gamma=trial.suggest_float("focal_gamma", 0.5, 5.0),
        )
    else:
        criterion = LOSSES[LOSS_NAME]()

    model = MODELS[MODEL_NAME]().to(config.device)
    decay = []
    no_decay = []
    for name, param in [*model.named_parameters(), *criterion.named_parameters()]:
        if param.requires_grad:
            (no_decay if param.ndim <= 1 or name.endswith(".bias") else decay).append(
                param
            )

    optimizer = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=lr,
        betas=(beta1, beta2),
        eps=getattr(config, "adam_eps", 1e-6),
    )

    scheduler: LambdaLR | None = None
    if use_cosine:
        steps = len(train_loader) * N_EPOCHS_TUNE
        scheduler = cosine_schedule(optimizer, len(train_loader) * warmup_epochs, steps)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        device=config.device,
        config=config,
        start_epoch=1,
    )

    best_val_loss = float("inf")
    for epoch in range(1, N_EPOCHS_TUNE + 1):
        trainer.train_epoch(train_loader, epoch)
        val_loss = trainer.validate_epoch(val_loader, epoch)
        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        best_val_loss = min(best_val_loss, val_loss)

    return best_val_loss


def make_loader(
    config: BaselineConfig,
    split_file: Path,
    shuffle: bool = False,
    drop_last: bool = False,
) -> DataLoader:
    return DataLoader(
        LandcoverDataset(
            image_dir=config.data_dir,
            split_file=split_file,
            transform=TrainTransform() if shuffle else ValTransform(),
        ),
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=config.device == "cuda",
        persistent_workers=config.num_workers > 0,
        drop_last=drop_last,
    )


def main() -> None:
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    config = BaselineConfig()

    train_loader = make_loader(
        config, config.train_split_file, shuffle=True, drop_last=True
    )
    val_loader = make_loader(config, config.val_split_file, shuffle=False)

    study = optuna.create_study(
        study_name=STUDY_NAME or f"{MODEL_NAME}_{LOSS_NAME}",
        direction="minimize",
        storage=STORAGE,
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2),
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    study.optimize(
        lambda trial: objective(trial, config, train_loader, val_loader),
        n_trials=N_TRIALS,
        timeout=TIMEOUT,
    )
