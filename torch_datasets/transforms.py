import random
from typing import Callable

import torch
import torchvision.transforms.functional as F
from torchvision.transforms import ColorJitter, Normalize, RandomResizedCrop, Resize

PairTransform = Callable[
    [torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]
]


class _RandomResizedCrop:
    def __init__(self, size: int, scale: tuple[float, float]) -> None:
        self.size = size
        self.scale = scale

    def __call__(self, image: torch.Tensor, mask: torch.Tensor):
        i, j, h, w = RandomResizedCrop.get_params(
            image, scale=list(self.scale), ratio=[1.0, 1.0]
        )
        image = F.resized_crop(image, i, j, h, w, [self.size, self.size])
        mask = F.resized_crop(
            mask,
            i,
            j,
            h,
            w,
            [self.size, self.size],
            interpolation=F.InterpolationMode.NEAREST,
        )
        return image, mask


class _Flips:
    def __init__(self, hflip_p: float, vflip_p: float) -> None:
        self.hflip_p = hflip_p
        self.vflip_p = vflip_p

    def __call__(self, image: torch.Tensor, mask: torch.Tensor):
        if random.random() < self.hflip_p:
            image, mask = F.hflip(image), F.hflip(mask)
        if random.random() < self.vflip_p:
            image, mask = F.vflip(image), F.vflip(mask)
        return image, mask


class _Rotate90:
    def __init__(self, p: float) -> None:
        self.p = p

    def __call__(self, image: torch.Tensor, mask: torch.Tensor):
        if random.random() < self.p:
            k = random.choice([1, 2, 3])
            image = torch.rot90(image, k, dims=[1, 2])
            mask = torch.rot90(mask, k, dims=[1, 2])
        return image, mask


class _ColorJitter:
    def __init__(self, params: tuple[float, float, float, float]) -> None:
        self.cj = ColorJitter(*params)

    def __call__(self, image: torch.Tensor, mask: torch.Tensor):
        return self.cj(image), mask


class _Resize:
    def __init__(self, size: int) -> None:
        self.size = size

    def __call__(self, image: torch.Tensor, mask: torch.Tensor):
        image = Resize((self.size, self.size))(image)
        mask = Resize(
            (self.size, self.size), interpolation=F.InterpolationMode.NEAREST
        )(mask)
        return image, mask


class _Normalize:
    def __call__(self, image: torch.Tensor, mask: torch.Tensor):
        norm = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        return norm(image), mask


class TrainTransform:
    def __init__(
        self,
        size: int = 512,
        crop_scale: tuple[float, float] | None = (0.7, 1.0),
        hflip_p: float | None = 0.5,
        vflip_p: float | None = 0.5,
        rotate90_p: float | None = 0.5,
        jitter_params: tuple[float, float, float, float] | None = (
            0.4,
            0.4,
            0.2,
            0.1,
        ),
    ) -> None:
        self.transforms_list: list[PairTransform] = []

        if crop_scale is not None:
            self.transforms_list.append(_RandomResizedCrop(size, crop_scale))

        if hflip_p is not None or vflip_p is not None:
            self.transforms_list.append(_Flips(hflip_p or 0.0, vflip_p or 0.0))

        if rotate90_p is not None:
            self.transforms_list.append(_Rotate90(rotate90_p))

        if jitter_params is not None:
            self.transforms_list.append(_ColorJitter(jitter_params))

        if crop_scale is None:
            self.transforms_list.append(_Resize(size))

        self.transforms_list.append(_Normalize())

    def __call__(
        self, image: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        for t in self.transforms_list:
            image, mask = t(image, mask)
        return image, mask


class ValTransform:
    def __init__(self, size: int = 512) -> None:
        self.size = size
        self.norm = Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

    def __call__(
        self, image: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        image = Resize((self.size, self.size))(image)
        mask = Resize(
            (self.size, self.size), interpolation=F.InterpolationMode.NEAREST
        )(mask)
        return self.norm(image), mask
