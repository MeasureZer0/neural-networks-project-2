# pyright: reportPrivateImportUsage=false
import argparse
import importlib
import math

import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from models.UNet import UNet
from torch_datasets.landcover_dataset import LandcoverDataset
from torch_datasets.transforms import TrainTransform, ValTransform
from training.checkpointing import load_checkpoint
from training.configs.baseline import BaselineConfig
from training.trainer import Trainer


def get_config(config_name: str) -> BaselineConfig:
    try:
        module_name = f"training.configs.{config_name}"
        module = importlib.import_module(module_name)
        if hasattr(module, "Config"):
            config_cls = module.Config
            if isinstance(config_cls, type) and issubclass(config_cls, BaselineConfig):
                return config_cls()
            elif isinstance(config_cls, BaselineConfig):
                return config_cls
    except (ImportError, AttributeError) as e:
        print(f"Error loading config {config_name}: {e}")
        print("Using default BaselineConfig")

    return BaselineConfig()


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
) -> LambdaLR:
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a semantic segmentation model.")
    parser.add_argument(
        "--config",
        type=str,
        default="baseline",
        help="Name of the config to use (default: baseline)",
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="Path to checkpoint to resume from"
    )
    args = parser.parse_args()

    # Enable TF32 for faster computation on Tensor Cores
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    # Enable cuDNN auto-tuner for convolutional networks
    torch.backends.cudnn.benchmark = True

    config = get_config(args.config)
    print(f"Using config: {config}")

    model = UNet().to(config.device)

    criterion = torch.nn.CrossEntropyLoss()

    decay, no_decay = [], []
    for _, name, param in [
        *((model, n, p) for n, p in model.named_parameters()),
        *((criterion, n, p) for n, p in criterion.named_parameters()),
    ]:
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)

    optimizer = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": config.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=config.lr,
        fused=True,
        betas=(
            getattr(config, "adam_beta1", 0.9),
            getattr(config, "adam_beta2", 0.98),
        ),
        eps=getattr(config, "adam_eps", 1e-6),
    )

    train_dataset = LandcoverDataset(
        image_dir=config.data_dir,
        split_file=config.train_split_file,
        transform=TrainTransform(),
    )
    val_dataset = LandcoverDataset(
        image_dir=config.data_dir,
        split_file=config.val_split_file,
        transform=ValTransform(),
    )

    pin_memory = config.device == "cuda"
    persistent_workers = config.num_workers > 0

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

    scheduler = None
    if getattr(config, "use_cosine_schedule", False):
        steps_per_epoch = len(train_loader)
        total_steps = steps_per_epoch * config.num_epochs
        warmup_steps = steps_per_epoch * getattr(config, "warmup_epochs", 1)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )
        print(
            f"Scheduler: cosine with {getattr(config, 'warmup_epochs', 5)} warmup epochs "
            f"({warmup_steps} steps) / {total_steps} total steps"
        )

    start_epoch = 1
    if args.resume is not None:
        start_epoch, _ = load_checkpoint(
            checkpoint_path=args.resume,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        model.to(config.device)
        start_epoch += 1
        print(f"Resumed from checkpoint {args.resume}, starting at epoch {start_epoch}")

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        device=config.device,
        config=config,
        start_epoch=start_epoch,
    )
    trainer.fit(train_loader, val_loader)


if __name__ == "__main__":
    main()
