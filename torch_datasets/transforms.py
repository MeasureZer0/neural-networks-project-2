from typing import Protocol

import albumentations as A
import torch
from albumentations.pytorch import ToTensorV2

_MEAN = (0.3651488, 0.39352093, 0.3404547)
_STD = (0.10747509, 0.09497052, 0.07975048)


class _TrainTransformConfig(Protocol):
    img_size: int
    crop_scale_min: float
    crop_scale_max: float
    hflip_p: float
    vflip_p: float
    rotate90_p: float
    color_jitter: bool
    jitter_brightness: float
    jitter_contrast: float
    jitter_saturation: float
    jitter_hue: float


class TrainTransform:
    def __init__(
        self,
        size: int = 512,
        crop_scale: tuple[float, float] | None = (0.7, 1.0),
        hflip_p: float = 0.5,
        vflip_p: float = 0.5,
        rotate90_p: float = 0.5,
        color_jitter: bool = True,
        jitter_params: tuple[float, float, float, float] = (0.4, 0.4, 0.2, 0.1),
    ) -> None:
        transforms = []

        if crop_scale is not None:
            transforms.append(
                A.RandomResizedCrop(
                    size=(size, size),
                    scale=crop_scale,
                    ratio=(1.0, 1.0),
                )
            )
        else:
            transforms.append(A.Resize(size, size))

        transforms.extend(
            [
                A.HorizontalFlip(p=hflip_p),
                A.VerticalFlip(p=vflip_p),
                A.RandomRotate90(p=rotate90_p),
            ]
        )

        if color_jitter:
            transforms.append(
                A.ColorJitter(
                    brightness=jitter_params[0],
                    contrast=jitter_params[1],
                    saturation=jitter_params[2],
                    hue=jitter_params[3],
                )
            )

        transforms.extend([A.Normalize(mean=_MEAN, std=_STD), ToTensorV2()])

        self.transform = A.Compose(transforms)

    def __call__(
        self, image: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        augmented = self.transform(image=image, mask=mask)
        return augmented["image"], augmented["mask"]


def train_transform_from_config(config: _TrainTransformConfig) -> TrainTransform:
    crop_scale_min = getattr(config, "crop_scale_min", 0.7)
    crop_scale_max = getattr(config, "crop_scale_max", 1.0)
    crop_scale = None
    if crop_scale_min is not None and crop_scale_max is not None:
        crop_scale = (float(crop_scale_min), float(crop_scale_max))

    return TrainTransform(
        size=getattr(config, "img_size", 512),
        crop_scale=crop_scale,
        hflip_p=getattr(config, "hflip_p", 0.5),
        vflip_p=getattr(config, "vflip_p", 0.5),
        rotate90_p=getattr(config, "rotate90_p", 0.5),
        color_jitter=getattr(config, "color_jitter", True),
        jitter_params=(
            getattr(config, "jitter_brightness", 0.4),
            getattr(config, "jitter_contrast", 0.4),
            getattr(config, "jitter_saturation", 0.2),
            getattr(config, "jitter_hue", 0.1),
        ),
    )


class ValTransform:
    def __init__(self, size: int = 512) -> None:
        self.transform = A.Compose(
            [
                A.Resize(size, size),
                A.Normalize(mean=_MEAN, std=_STD),
                ToTensorV2(),
            ]
        )

    def __call__(
        self, image: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        augmented = self.transform(image=image, mask=mask)
        return augmented["image"], augmented["mask"]


class SSLTransform:
    def __init__(
        self,
        size: int = 512,
        crop_scale: tuple[float, float] | None = (0.7, 1.0),
        use_dual_strong_views: bool = False,
    ) -> None:
        spatial_transforms = []
        if crop_scale is not None:
            spatial_transforms.append(
                A.RandomResizedCrop(
                    size=(size, size),
                    scale=crop_scale,
                    ratio=(1.0, 1.0),
                )
            )
        else:
            spatial_transforms.append(A.Resize(size, size))

        spatial_transforms.extend(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
            ]
        )

        self.spatial_transform = A.ReplayCompose(spatial_transforms)
        self.weak_transform = A.Compose([A.Normalize(mean=_MEAN, std=_STD), ToTensorV2()])
        self.strong_transform = A.Compose(
            [
                A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
                A.GaussianBlur(blur_limit=(3, 7), p=0.5),
                A.GaussNoise(p=0.5),
                A.Normalize(mean=_MEAN, std=_STD),
                ToTensorV2(),
            ]
        )
        self.use_dual_strong_views = use_dual_strong_views

    def __call__(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        spatial = self.spatial_transform(image=image)
        weak_image = spatial["image"]
        replay = spatial["replay"]

        strong_spatial = A.ReplayCompose.replay(replay, image=image)["image"]
        result = {
            "image_weak": self.weak_transform(image=weak_image)["image"],
            "image_strong": self.strong_transform(image=strong_spatial)["image"],
        }

        if self.use_dual_strong_views:
            strong_spatial_2 = A.ReplayCompose.replay(replay, image=image)["image"]
            result["image_strong_1"] = result["image_strong"]
            result["image_strong_2"] = self.strong_transform(image=strong_spatial_2)[
                "image"
            ]

        return result
