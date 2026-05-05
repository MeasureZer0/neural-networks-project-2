# pyright: reportPrivateImportUsage=false
from typing import Dict

import torch
import torch.nn as nn
from fvcore.nn import FlopCountAnalysis


def model_stats(model: nn.Module, use_fp16: bool = False) -> Dict[str, float]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    bytes_per_param = 2 if use_fp16 else 4
    param_mb = total * bytes_per_param / 1024**2
    return {
        "total_param_M": total / 1e6,
        "trainable_param_M": trainable / 1e6,
        "frozen_param_M": frozen / 1e6,
        "param_size_MB": param_mb,
        "precision": 16.0 if use_fp16 else 32.0,
    }


def flop_stats(model: nn.Module, sample_input: torch.Tensor) -> Dict[str, float]:
    model.eval()
    flops = FlopCountAnalysis(model, sample_input[:1])
    flops.unsupported_ops_warnings(False)
    flops.uncalled_modules_warnings(False)
    return {"flops_G": flops.total() / 1e9}


def vram_stats(
    model: nn.Module,
    device: torch.device,
    sample_input: torch.Tensor,
    use_fp16: bool = False,
) -> Dict[str, float]:
    if device.type != "cuda":
        return {}

    stats = {}
    if use_fp16:
        model = model.half()

    torch.cuda.reset_peak_memory_stats()
    model.to(device)
    stats["vram_model_MB"] = torch.cuda.memory_allocated() / 1024**2

    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        with torch.amp.autocast(device_type=device.type, enabled=use_fp16):
            _ = model(sample_input[:1].to(device))
    stats["vram_forward_peak_MB"] = torch.cuda.max_memory_allocated() / 1024**2
    return stats


def print_stats(name: str, stats: Dict[str, float]) -> None:
    print(f"\n{'=' * 55}")
    print(f"  {name}")
    print(f"{'=' * 55}")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key:<30} {value:.2f}")
        else:
            print(f"  {key:<30} {value}")


CONFIGS = [
    {
        "name": "FPN (ResNet50 backbone)",
        "model_type": "fpn",
        "num_classes": 5,
    },
    {
        "name": "UNet (from scratch)",
        "model_type": "unet",
        "num_classes": 5,
    },
    {
        "name": "DeepLabV3 - frozen backbone",
        "model_type": "deeplabv3",
        "num_classes": 5,
        "freeze_backbone": True,
    },
    {
        "name": "DeepLabV3 - full fine-tune",
        "model_type": "deeplabv3",
        "num_classes": 5,
        "freeze_backbone": False,
    },
]


def build_model(cfg: dict) -> nn.Module:
    model_type = cfg["model_type"]
    num_classes = cfg["num_classes"]

    if model_type == "fpn":
        from models.FPN import FPNSegmentation

        return FPNSegmentation(num_classes=num_classes)
    elif model_type == "unet":
        from models.UNet import UNet

        return UNet(out_channels=num_classes)
    elif model_type == "deeplabv3":
        from models.deeplabv3_model import DeepLabV3

        return DeepLabV3(
            num_classes=num_classes,
            freeze_backbone=cfg.get("freeze_backbone", True),
        )
    else:
        raise ValueError(f"Unknown model type: {model_type!r}")


def profile_segmentation_models(
    sample_input: torch.Tensor,
    device: torch.device,
) -> None:
    for use_fp16 in [False, True]:
        precision = "FP16" if use_fp16 else "FP32"

        print(f"\n\n{'#' * 55}")
        print(f"  SEGMENTATION MODELS — {precision}")
        print(f"{'#' * 55}")

        for cfg in CONFIGS:
            model = build_model(cfg)
            model.eval()

            stats: Dict[str, float] = {}
            stats.update(model_stats(model, use_fp16=use_fp16))

            try:
                stats.update(flop_stats(model, sample_input))
            except Exception as e:
                print(f"  [FLOPs Error] Skipping FLOPs for {cfg['name']}: {e}")
                stats["flops_G"] = 0.0

            stats.update(vram_stats(model, device, sample_input, use_fp16=use_fp16))
            print_stats(f"{cfg['name']} ({precision})", stats)

            model.cpu()
            if device.type == "cuda":
                torch.cuda.empty_cache()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    BATCH_SIZE = 4
    IMG_SIZE = 512
    sample_input = torch.randn(BATCH_SIZE, 3, IMG_SIZE, IMG_SIZE)

    profile_segmentation_models(sample_input, device)
