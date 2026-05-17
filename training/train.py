# pyright: reportPrivateImportUsage=false
import argparse
import importlib
import inspect
import math

import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from models.deeplabv3_model import DeepLabV3
from models.deeplabv3_variants import (
    DeepLabV3Cascaded,
    DeepLabV3Dropout,
    DeepLabV3LargeDilations,
    DeepLabV3NarrowASPP,
    DeepLabV3NoASPP,
    DeepLabV3NoGlobalPool,
    DeepLabV3SmallDilations,
)
from models.deeplabv3_variants import DeepLabV3ShallowHead as DLV3ShallowHead
from models.FPN import FPNSegmentation
from models.FPN_variants import (
    FPNConcatMerge,
    FPNDeepHead,
    FPNNoLateral,
    FPNNoP5,
    FPNShallowHead,
    FPNSingleScale,
    FPNSumMerge,
)
from models.UNet import UNet
from models.UNet_variants import (
    UNetDeepBottleneck,
    UNetNarrow,
    UNetNoBN,
    UNetNoSkip,
    UNetResidual,
    UNetShallow,
    UNetWide,
)
from torch_datasets.landcover_dataset import (
    LandcoverDataset,
    UnlabeledLandcoverDataset,
)
from torch_datasets.transforms import (
    SSLTransform,
    ValTransform,
    train_transform_from_config,
)
from training.checkpointing import load_checkpoint
from training.configs.baseline import BaselineConfig
from training.loss import CEDiceLoss, CELoss, DiceLoss, FocalLoss
from training.trainer import Trainer

MODELS = ("fpn", "unet", "deeplabv3")
LOSSES = ("dice", "ce", "focal")


def get_config(config_name: str, variant_name: str | None = None) -> BaselineConfig:
    try:
        module_name = f"training.configs.{config_name}"
        module = importlib.import_module(module_name)

        if variant_name:
            possible_names = [
                variant_name,
                f"{variant_name}Config",
                f"{config_name.capitalize()}{variant_name.capitalize()}",
            ]
            for name in possible_names:
                cls = getattr(module, name, None)
                if (
                    inspect.isclass(cls)
                    and issubclass(cls, BaselineConfig)
                    and cls is not BaselineConfig
                ):
                    return cls()
        if hasattr(module, "Config"):
            c = module.Config
            return c() if inspect.isclass(c) else c

        for name, obj in inspect.getmembers(module, inspect.isclass):
            if (
                issubclass(obj, BaselineConfig)
                and obj is not BaselineConfig
                and not name.startswith("_")
            ):
                return obj()

    except (ImportError, AttributeError) as e:
        print(f"Error loading config {config_name}: {e}")

    print(
        f"No specific config found in {config_name}, returning default BaselineConfig"
    )
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


def ssl_crop_scale_from_config(config: BaselineConfig) -> tuple[float, float] | None:
    crop_scale_min = getattr(config, "crop_scale_min", 0.7)
    crop_scale_max = getattr(config, "crop_scale_max", 1.0)
    if crop_scale_min is None or crop_scale_max is None:
        return None
    return (float(crop_scale_min), float(crop_scale_max))


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
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        choices=MODELS,
        help=(
            "Model architecture to use. Overrides the value from config if provided. "
            f"Choices: {MODELS}"
        ),
    )
    parser.add_argument(
        "--loss",
        type=str,
        default=None,
        choices=LOSSES,
        help=(
            "Loss function to use. Overrides the value from config if provided. "
            f"Choices: {LOSSES}"
        ),
    )
    parser.add_argument(
        "--variant", type=str, default=None, help="Ablation variant name"
    )
    args = parser.parse_args()

    # Enable TF32 for faster computation on Tensor Cores
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    # Enable cuDNN auto-tuner for convolutional networks
    torch.backends.cudnn.benchmark = True

    config = get_config(args.config, args.variant)
    print(f"Using config: {config}")

    model_name: str = args.model or getattr(config, "model", None) or "fpn"

    variant = getattr(config, f"{model_name}_variant", None)

    if model_name == "unet":
        unet_map = {
            "basic": UNet,
            "no_skip": UNetNoSkip,
            "shallow": UNetShallow,
            "wide": UNetWide,
            "narrow": UNetNarrow,
            "no_bn": UNetNoBN,
            "residual": UNetResidual,
            "deep_bottleneck": UNetDeepBottleneck,
        }
        model_cls = unet_map.get(variant if variant else "basic", UNet)
        model = model_cls(
            in_channels=getattr(config, "in_channels", 3),
            out_channels=config.num_classes,
        )

    elif model_name == "fpn":
        fpn_map = {
            "basic": FPNSegmentation,
            "no_lateral": FPNNoLateral,
            "single_scale": FPNSingleScale,
            "sum_merge": FPNSumMerge,
            "concat_merge": FPNConcatMerge,
            "shallow_head": FPNShallowHead,
            "deep_head": FPNDeepHead,
            "no_p5": FPNNoP5,
        }
        model_cls = fpn_map.get(variant if variant else "basic", FPNSegmentation)
        model = model_cls(
            num_classes=config.num_classes,
            out_channels=getattr(config, "out_channels", 256),
        )

    elif model_name == "deeplabv3":
        dlv3_map = {
            "basic": DeepLabV3,
            "no_aspp": DeepLabV3NoASPP,
            "no_global_pool": DeepLabV3NoGlobalPool,
            "narrow_aspp": DeepLabV3NarrowASPP,
            "small_dil": DeepLabV3SmallDilations,
            "large_dil": DeepLabV3LargeDilations,
            "shallow_head": DLV3ShallowHead,
            "dropout": DeepLabV3Dropout,
            "cascaded": DeepLabV3Cascaded,
        }
        model_cls = dlv3_map.get(variant if variant else "basic", DeepLabV3)
        model = model_cls(
            num_classes=config.num_classes,
            pretrained=config.pretrained,
            freeze_backbone=config.freeze_backbone,
        )
    else:
        raise ValueError(f"Unknown model: {model_name!r}. Choose from {MODELS}.")

    model = model.to(config.device)
    print(f"Model: {model_name}")

    loss_name: str = args.loss or getattr(config, "loss", None) or "dice"

    if loss_name == "dice":
        criterion: torch.nn.Module = DiceLoss()
    elif loss_name == "ce":
        criterion = CELoss()
    elif loss_name == "focal":
        criterion = FocalLoss()
    elif loss_name == "ce_dice":
        ce_weight = getattr(config, "ce_weight", 1.0)
        dice_weight = getattr(config, "dice_weight", 1.0)
        criterion = CEDiceLoss(ce_weight=ce_weight, dice_weight=dice_weight)
    else:
        raise ValueError(f"Unknown loss: {loss_name!r}. Choose from {LOSSES}.")
    print(f"Loss: {loss_name}")

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
        transform=train_transform_from_config(config),
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

    unlabeled_loader = None
    semi_supervised = getattr(config, "semi_supervised", None)
    if getattr(semi_supervised, "enabled", False):
        ssl_transform = SSLTransform(
            size=config.img_size,
            crop_scale=ssl_crop_scale_from_config(config),
            use_dual_strong_views=getattr(
                semi_supervised, "use_dual_strong_views", False
            ),
        )
        unlabeled_datasets = []

        unlabeled_split_file = getattr(semi_supervised, "unlabeled_split_file", None)
        if unlabeled_split_file is not None:
            unlabeled_datasets.append(
                UnlabeledLandcoverDataset.from_split_file(
                    split_file=unlabeled_split_file,
                    image_dir=config.data_dir,
                    transform=ssl_transform,
                )
            )

        extra_unlabeled_split_file = getattr(
            semi_supervised, "extra_unlabeled_split_file", None
        )
        if extra_unlabeled_split_file is not None:
            unlabeled_datasets.append(
                UnlabeledLandcoverDataset.from_split_file(
                    split_file=extra_unlabeled_split_file,
                    image_dir=None,
                    transform=ssl_transform,
                )
            )

        if not unlabeled_datasets:
            raise ValueError(
                "When semi-supervised training is enabled, set at least one of "
                "semi_supervised.unlabeled_split_file or "
                "semi_supervised.extra_unlabeled_split_file."
            )

        if len(unlabeled_datasets) == 1:
            unlabeled_dataset = unlabeled_datasets[0]
        else:
            unlabeled_dataset = torch.utils.data.ConcatDataset(unlabeled_datasets)

        unlabeled_loader = DataLoader(
            unlabeled_dataset,
            batch_size=config.batch_size
            * getattr(semi_supervised, "unlabeled_batch_ratio", 1),
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            drop_last=True,
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

    resume_checkpoint = None
    start_epoch = 1
    if args.resume is not None:
        resume_checkpoint = load_checkpoint(
            checkpoint_path=args.resume,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        start_epoch = int(resume_checkpoint["epoch"])
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
    if resume_checkpoint is not None:
        trainer.restore_checkpoint_state(resume_checkpoint)
    trainer.fit(train_loader, val_loader, unlabeled_loader=unlabeled_loader)


if __name__ == "__main__":
    main()
