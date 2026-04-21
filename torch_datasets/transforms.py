import albumentations as A
import torch
from albumentations.pytorch import ToTensorV2

_MEAN = (0.37451365, 0.40184889, 0.35060243)
_STD = (0.1159367, 0.09916778, 0.08508567)


class TrainTransform:
    def __init__(
        self,
        size: int = 512,
        crop_scale: tuple[float, float] | None = (0.7, 1.0),
        hflip_p: float = 0.5,
        vflip_p: float = 0.5,
        rotate90_p: float = 0.5,
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
                A.ColorJitter(
                    brightness=jitter_params[0],
                    contrast=jitter_params[1],
                    saturation=jitter_params[2],
                    hue=jitter_params[3],
                ),
                A.Normalize(mean=_MEAN, std=_STD),
                ToTensorV2(),
            ]
        )

        self.transform = A.Compose(transforms)

    def __call__(
        self, image: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        augmented = self.transform(image=image, mask=mask)
        return augmented["image"], augmented["mask"]


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
