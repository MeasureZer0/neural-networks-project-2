import importlib
import math
import os
from pathlib import Path

import optuna  # type: ignore[no-redef]
import torch
import wandb
from optuna.integration.wandb import WeightsAndBiasesCallback  # type: ignore[no-redef]
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from models.deeplabv3_model import DeepLabV3
from models.FPN import FPNSegmentation
from models.UNet import UNet
from torch_datasets.landcover_dataset import LandcoverDataset
from torch_datasets.transforms import ValTransform, train_transform_from_config
from training.configs.baseline import BaselineConfig
from training.loss import CELoss, DiceLoss, FocalLoss
from training.metrics import SegmentationMetrics
from training.trainer import Trainer


def build_model(model_name: str, config: BaselineConfig) -> torch.nn.Module:
    num_classes = config.num_classes
    if model_name == "fpn":
        return FPNSegmentation(num_classes=num_classes)
    elif model_name == "unet":
        return UNet(in_channels=3, out_channels=num_classes)
    elif model_name == "deeplabv3":
        return DeepLabV3(
            num_classes=num_classes,
            pretrained=config.pretrained,
            freeze_backbone=config.freeze_backbone,
        )
    else:
        raise ValueError(f"Unknown model: {model_name!r}")


LOSSES: dict[str, type] = {"dice": DiceLoss, "ce": CELoss, "focal": FocalLoss}

CHECKPOINT_BASE = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "checkpoints", "tune"
)


def get_config(config_name: str) -> BaselineConfig:
    try:
        module = importlib.import_module(f"training.configs.{config_name}")
        if hasattr(module, "Config"):
            cls = module.Config
            if isinstance(cls, type) and issubclass(cls, BaselineConfig):
                return cls()
            if isinstance(cls, BaselineConfig):
                return cls
    except (ImportError, AttributeError):
        pass
    return BaselineConfig()


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
    model_name: str,
    loss_name: str,
    n_epochs_tune: int,
    use_wandb: bool = False,
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
    config.use_wandb = False  # logging handled here, not in Trainer
    config.checkpoint_dir = os.path.join(CHECKPOINT_BASE, f"trial_{trial.number}")

    if loss_name == "focal":
        criterion: torch.nn.Module = FocalLoss(
            alpha=trial.suggest_float("focal_alpha", 0.25, 2.0),
            gamma=trial.suggest_float("focal_gamma", 0.5, 5.0),
        )
    else:
        criterion = LOSSES[loss_name]()

    model = build_model(model_name, config).to(config.device)

    decay, no_decay = [], []
    for name, param in [
        *model.named_parameters(),
        *criterion.named_parameters(),
    ]:
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)

    optimizer = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=lr,
        betas=(beta1, beta2),
        eps=getattr(config, "adam_eps", 1e-6),
        fused=config.device == "cuda",
    )

    scheduler: LambdaLR | None = None
    if use_cosine:
        total_steps = len(train_loader) * n_epochs_tune
        warmup_steps = len(train_loader) * warmup_epochs
        scheduler = cosine_schedule(optimizer, warmup_steps, total_steps)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        device=config.device,
        config=config,
        start_epoch=1,
    )

    num_classes: int = config.num_classes
    best_val_loss = float("inf")

    mean_iou = 0.0
    mean_dice = 0.0
    for epoch in range(1, n_epochs_tune + 1):
        train_loss = trainer.train_epoch(train_loader, epoch)

        model.eval()
        total_val_loss = 0.0
        seg_metrics = SegmentationMetrics(
            num_classes=num_classes, background_index=0, device=config.device
        )

        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(config.device, non_blocking=True)
                masks = batch["mask"].to(config.device, non_blocking=True)

                dtype = trainer.dtype
                with torch.autocast(device_type=config.device, dtype=dtype):
                    logits = model(images)
                    loss = criterion(logits, masks)

                total_val_loss += loss.item()
                seg_metrics.update(logits, masks)

        val_loss = total_val_loss / len(val_loader)
        mean_iou = seg_metrics.mean_iou(exclude_background=True)
        mean_dice = seg_metrics.mean_dice(exclude_background=True)
        pixel_acc = seg_metrics.pixel_accuracy()

        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        if use_wandb and wandb.run is not None:
            log_dict: dict = {
                "trial": trial.number,
                "epoch": epoch,
                "train/loss": train_loss,
                "val/loss": val_loss,
                "val/mean_iou": mean_iou,
                "val/mean_dice": mean_dice,
                "val/pixel_accuracy": pixel_acc,
                "lr": optimizer.param_groups[0]["lr"],
                # log hyper-params as trial attrs so they appear in W&B table
                "hparams/lr": lr,
                "hparams/weight_decay": weight_decay,
                "hparams/beta1": beta1,
                "hparams/beta2": beta2,
                "hparams/use_cosine": use_cosine,
                "hparams/warmup_epochs": warmup_epochs,
            }
            per_iou = seg_metrics.per_class_iou().cpu().tolist()
            per_dice = seg_metrics.per_class_dice().cpu().tolist()
            for i, (iou, dice) in enumerate(zip(per_iou, per_dice, strict=False)):
                log_dict[f"val/iou_class_{i}"] = iou
                log_dict[f"val/dice_class_{i}"] = dice

            wandb.log(log_dict)

        best_val_loss = min(best_val_loss, val_loss)

    trial.set_user_attr("best_val_loss", best_val_loss)
    trial.set_user_attr("best_mean_iou", mean_iou)  # last epoch; good enough for HPO
    trial.set_user_attr("best_mean_dice", mean_dice)

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
            transform=train_transform_from_config(config)
            if shuffle
            else ValTransform(),
        ),
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=config.device == "cuda",
        persistent_workers=config.num_workers > 0,
        drop_last=drop_last,
    )


def main(
    model_name: str = "fpn",
    loss_name: str = "dice",
    config_name: str = "baseline",
    n_trials: int = 50,
    n_epochs_tune: int = 5,
    timeout: int | None = None,
    study_name: str | None = None,
    storage: str | None = "sqlite:///optuna.db",
    use_wandb: bool = False,
) -> None:
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    config = get_config(config_name)

    run_name = study_name or f"{model_name}_{loss_name}_tune"
    if use_wandb:
        wandb.init(
            project=getattr(config, "wandb_project", "semantic-segmentation"),
            entity=getattr(config, "wandb_entity", None),
            name=run_name,
            # Wszystkie trialy HPO trafiają do grupy "hpo" — oddzielnej od ablacji
            group="hpo",
            tags=["hpo", model_name, loss_name],
            config={
                "model": model_name,
                "loss": loss_name,
                "n_trials": n_trials,
                "n_epochs_tune": n_epochs_tune,
            },
        )

    callbacks = []
    if use_wandb:
        # WeightsAndBiasesCallback logs trial-level summary to a W&B Table
        callbacks.append(
            WeightsAndBiasesCallback(
                metric_name="val_loss",
                wandb_kwargs={"reinit": False},
            )
        )

    train_loader = make_loader(
        config, config.train_split_file, shuffle=True, drop_last=True
    )
    val_loader = make_loader(config, config.val_split_file, shuffle=False)

    study = optuna.create_study(
        study_name=run_name,
        direction="minimize",
        storage=storage,
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2),
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    study.optimize(
        lambda trial: objective(
            trial,
            config,
            train_loader,
            val_loader,
            model_name,
            loss_name,
            n_epochs_tune,
            use_wandb=use_wandb,
        ),
        n_trials=n_trials,
        timeout=timeout,
        callbacks=callbacks,
    )

    print(f"\nBest trial #{study.best_trial.number}")
    print(f"  val_loss : {study.best_value:.6f}")
    print("  params   :")
    for k, v in study.best_params.items():
        print(f"    {k}: {v}")

    if use_wandb and wandb.run is not None:
        wandb.log(
            {
                "best/val_loss": study.best_value,
                **{f"best/{k}": v for k, v in study.best_params.items()},
            }
        )
        wandb.finish()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Optuna HPO for segmentation models.")
    parser.add_argument(
        "--model", type=str, default="fpn", choices=["fpn", "unet", "deeplabv3"]
    )
    parser.add_argument(
        "--loss", type=str, default="dice", choices=["dice", "ce", "focal"]
    )
    parser.add_argument("--config", type=str, default="baseline")
    parser.add_argument("--n_trials", type=int, default=40)
    parser.add_argument("--n_epochs_tune", type=int, default=10)
    parser.add_argument(
        "--timeout", type=int, default=None, help="Stop after N seconds"
    )
    parser.add_argument("--study_name", type=str, default=None)
    parser.add_argument("--storage", type=str, default="sqlite:///optuna.db")
    parser.add_argument("--wandb", action="store_true", dest="use_wandb")
    args = parser.parse_args()

    main(
        model_name=args.model,
        loss_name=args.loss,
        config_name=args.config,
        n_trials=args.n_trials,
        n_epochs_tune=args.n_epochs_tune,
        timeout=args.timeout,
        study_name=args.study_name,
        storage=args.storage,
        use_wandb=args.use_wandb,
    )
