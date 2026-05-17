from __future__ import annotations

import base64
import io
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

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
from training.configs.baseline import BaselineConfig
from training.metrics import SegmentationMetrics
from visualisation.demo_app.landcover import LANDCOVER_CLASS_NAMES

MEAN = torch.tensor([0.3651488, 0.39352093, 0.3404547], dtype=torch.float32).view(
    3, 1, 1
)
STD = torch.tensor([0.10747509, 0.09497052, 0.07975048], dtype=torch.float32).view(
    3, 1, 1
)
CLASS_COLORS = np.array(
    [
        [34, 40, 49],
        [234, 179, 8],
        [22, 163, 74],
        [37, 99, 235],
        [239, 68, 68],
    ],
    dtype=np.uint8,
)


@dataclass
class LoadedModel:
    checkpoint_path: Path
    checkpoint: dict[str, Any]
    config: BaselineConfig
    model: nn.Module
    device: str


def resolve_device(requested: str) -> str:
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but no CUDA device is available.")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested, but it is not available.")
    return requested


def _class_names(num_classes: int) -> list[str]:
    if num_classes <= len(LANDCOVER_CLASS_NAMES):
        return LANDCOVER_CLASS_NAMES[:num_classes]
    return [f"Class {index}" for index in range(num_classes)]


def build_model_from_config(config: BaselineConfig) -> nn.Module:
    model_name = getattr(config, "model", "fpn")
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
        return model_cls(
            in_channels=getattr(config, "in_channels", 3),
            out_channels=config.num_classes,
        )

    if model_name == "fpn":
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
        return model_cls(
            num_classes=config.num_classes,
            out_channels=getattr(config, "out_channels", 256),
            pretrained=False,
        )

    if model_name == "deeplabv3":
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
        return model_cls(
            num_classes=config.num_classes,
            pretrained=False,
            freeze_backbone=getattr(config, "freeze_backbone", False),
        )

    raise ValueError(f"Unsupported model type: {model_name!r}")


def normalize_state_dict_keys(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    normalized: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        cleaned = key
        if cleaned.startswith("_orig_mod."):
            cleaned = cleaned.removeprefix("_orig_mod.")
        if cleaned.startswith("module."):
            cleaned = cleaned.removeprefix("module.")
        normalized[cleaned] = value
    return normalized


def load_model(checkpoint_path: str | Path, requested_device: str = "auto") -> LoadedModel:
    path = Path(checkpoint_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = checkpoint.get("config", BaselineConfig())
    if not isinstance(config, BaselineConfig):
        raise TypeError("Checkpoint config is not a BaselineConfig instance.")

    device = resolve_device(requested_device)
    model = build_model_from_config(config)
    model.load_state_dict(normalize_state_dict_keys(checkpoint["model_state_dict"]))
    model.to(device)
    model.eval()

    return LoadedModel(
        checkpoint_path=path,
        checkpoint=checkpoint,
        config=config,
        model=model,
        device=device,
    )


def preprocess_image(image: Image.Image, size: int) -> torch.Tensor:
    resized = image.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
    image_array = np.asarray(resized, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(image_array).permute(2, 0, 1)
    return (tensor - MEAN) / STD


def load_mask(mask_bytes: bytes, target_size: tuple[int, int]) -> torch.Tensor:
    image = Image.open(io.BytesIO(mask_bytes))
    mask = image.convert("L").resize(target_size, Image.Resampling.NEAREST)
    mask_array = np.asarray(mask, dtype=np.int64)
    return torch.from_numpy(mask_array)


def mask_to_rgb(mask: np.ndarray, num_classes: int) -> np.ndarray:
    palette = CLASS_COLORS[:num_classes]
    if len(palette) < num_classes:
        raise ValueError("Not enough class colors configured for this checkpoint.")
    return palette[mask]


def blend_overlay(image: np.ndarray, mask_rgb: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    return np.clip(image * (1.0 - alpha) + mask_rgb * alpha, 0, 255).astype(np.uint8)


def image_to_base64(image: np.ndarray) -> str:
    pil_image = Image.fromarray(image)
    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def summarize_prediction(prediction: np.ndarray, confidence: np.ndarray) -> dict[str, Any]:
    pixel_count = prediction.size
    labels, counts = np.unique(prediction, return_counts=True)
    breakdown = []
    names = _class_names(int(prediction.max()) + 1)
    for label, count in zip(labels.tolist(), counts.tolist(), strict=False):
        class_mask = prediction == label
        breakdown.append(
            {
                "index": label,
                "name": names[label],
                "pixels": count,
                "coverage_pct": (count / pixel_count) * 100.0,
                "mean_confidence": float(confidence[class_mask].mean()),
            }
        )

    dominant = max(breakdown, key=lambda row: row["pixels"]) if breakdown else None
    return {
        "mean_confidence": float(confidence.mean()),
        "dominant_class": dominant,
        "class_breakdown": breakdown,
    }


def compute_metrics(
    logits: torch.Tensor,
    ground_truth: torch.Tensor,
    num_classes: int,
) -> dict[str, Any]:
    metrics = SegmentationMetrics(num_classes=num_classes, background_index=0, device="cpu")
    metrics.update(logits.cpu(), ground_truth.unsqueeze(0).cpu())

    per_class_iou = metrics.per_class_iou().tolist()
    per_class_dice = metrics.per_class_dice().tolist()
    class_rows = []
    for index, (iou, dice) in enumerate(zip(per_class_iou, per_class_dice, strict=False)):
        class_rows.append(
            {
                "index": index,
                "name": _class_names(num_classes)[index],
                "iou": float(iou),
                "dice": float(dice),
            }
        )

    return {
        "pixel_accuracy": metrics.pixel_accuracy(),
        "mean_iou": metrics.mean_iou(exclude_background=True),
        "mean_dice": metrics.mean_dice(exclude_background=True),
        "per_class": class_rows,
    }


def run_inference(
    loaded_model: LoadedModel,
    image_bytes: bytes,
    mask_bytes: bytes | None = None,
    source_label: str | None = None,
) -> dict[str, Any]:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    original_size = image.size
    input_size = int(getattr(loaded_model.config, "img_size", 512))

    input_tensor = preprocess_image(image, input_size).unsqueeze(0).to(loaded_model.device)

    start_time = time.perf_counter()
    with torch.inference_mode():
        logits = loaded_model.model(input_tensor)
        logits = F.interpolate(
            logits,
            size=(original_size[1], original_size[0]),
            mode="bilinear",
            align_corners=False,
        )
        probabilities = torch.softmax(logits, dim=1)
        confidence, prediction = probabilities.max(dim=1)
    inference_ms = (time.perf_counter() - start_time) * 1000.0

    prediction_np = prediction.squeeze(0).cpu().numpy().astype(np.uint8)
    confidence_np = confidence.squeeze(0).cpu().numpy()
    image_np = np.asarray(image, dtype=np.uint8)
    mask_rgb = mask_to_rgb(prediction_np, loaded_model.config.num_classes)
    overlay = blend_overlay(image_np, mask_rgb)

    metrics = None
    ground_truth_b64 = None
    if mask_bytes is not None and mask_bytes:
        ground_truth = load_mask(mask_bytes, original_size)
        metrics = compute_metrics(logits, ground_truth, loaded_model.config.num_classes)
        ground_truth_rgb = mask_to_rgb(
            ground_truth.numpy().astype(np.uint8), loaded_model.config.num_classes
        )
        ground_truth_b64 = image_to_base64(ground_truth_rgb)

    checkpoint = loaded_model.checkpoint
    return {
        "checkpoint": {
            "path": str(loaded_model.checkpoint_path),
            "config_name": type(loaded_model.config).__name__,
            "model_name": getattr(loaded_model.config, "model", "unknown"),
            "img_size": input_size,
            "epoch": checkpoint.get("epoch"),
            "train_loss": checkpoint.get("train_loss"),
            "val_loss": checkpoint.get("val_loss"),
            "device": loaded_model.device,
        },
        "source": source_label,
        "summary": summarize_prediction(prediction_np, confidence_np),
        "metrics": metrics,
        "inference_ms": inference_ms,
        "images": {
            "input": image_to_base64(image_np),
            "prediction": image_to_base64(mask_rgb),
            "overlay": image_to_base64(overlay),
            "ground_truth": ground_truth_b64,
        },
    }
