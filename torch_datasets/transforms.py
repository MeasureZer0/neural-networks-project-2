import random
from typing import Callable

import torch
import torchvision.transforms.functional as F
from torchvision.transforms import (
    ColorJitter,
    Normalize,
    RandomResizedCrop,
    Resize,
)

# Alias for a paired image+mask transform
PairTransform = Callable[
    [torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]
]


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
        self.size = size
        self.transforms_list: list[PairTransform] = []

        self.normalize = Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

        if crop_scale is not None:
            self.transforms_list.append(self._random_resized_crop(crop_scale))

        if hflip_p is not None or vflip_p is not None:
            self.transforms_list.append(self._flips(hflip_p or 0.0, vflip_p or 0.0))

        if rotate90_p is not None:
            self.transforms_list.append(self._rotate90(rotate90_p))

        if jitter_params is not None:
            self.transforms_list.append(self._color_jitter(jitter_params))

        if crop_scale is None:
            self.transforms_list.append(self._resize())

        self.transforms_list.append(self._normalize())

    def _random_resized_crop(self, crop_scale: tuple[float, float]) -> PairTransform:
        size = self.size

        def transform(
            image: torch.Tensor, mask: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            i, j, h, w = RandomResizedCrop.get_params(
                image, scale=list(crop_scale), ratio=[1.0, 1.0]
            )
            image = F.resized_crop(image, i, j, h, w, [size, size])
            mask = F.resized_crop(
                mask,
                i,
                j,
                h,
                w,
                [size, size],
                interpolation=F.InterpolationMode.NEAREST,
            )
            return image, mask

        return transform

    def _flips(self, hflip_p: float, vflip_p: float) -> PairTransform:
        def transform(
            image: torch.Tensor, mask: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            if random.random() < hflip_p:
                image = F.hflip(image)
                mask = F.hflip(mask)
            if random.random() < vflip_p:
                image = F.vflip(image)
                mask = F.vflip(mask)
            return image, mask

        return transform

    def _rotate90(self, p: float) -> PairTransform:
        def transform(
            image: torch.Tensor, mask: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            if random.random() < p:
                k = random.choice([1, 2, 3])  # 90, 180, 270 degrees
                image = torch.rot90(image, k, dims=[1, 2])
                mask = torch.rot90(mask, k, dims=[0, 1])
            return image, mask

        return transform

    def _color_jitter(self, params: tuple[float, float, float, float]) -> PairTransform:
        cj = ColorJitter(*params)

        def transform(
            image: torch.Tensor, mask: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            return cj(image), mask

        return transform

    def _resize(self) -> PairTransform:
        size = self.size

        def transform(
            image: torch.Tensor, mask: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            image = Resize((size, size))(image)
            mask = Resize((size, size), interpolation=F.InterpolationMode.NEAREST)(mask)
            return image, mask

        return transform

    def _normalize(self) -> PairTransform:
        normalize = self.normalize

        def transform(
            image: torch.Tensor, mask: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            return normalize(image), mask

        return transform

    def __call__(
        self, image: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        for t in self.transforms_list:
            image, mask = t(image, mask)
        return image, mask


class ValTransform:
    def __init__(self, size: int = 512) -> None:
        self.size = size
        self.normalize = Normalize(
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
        image = self.normalize(image)
        return image, mask
