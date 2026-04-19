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
        crop_scale: tuple[float, float] | None = (0.5, 1.0),
        hflip_p: float | None = 0.5,
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
            mean=[0.47087333, 0.44731208, 0.40772682],
            std=[0.2517867, 0.2472999, 0.25216556],
        )

        if crop_scale is not None:
            self.transforms_list.append(self._random_resized_crop(crop_scale))

        if hflip_p is not None:
            self.transforms_list.append(self._horizontal_flip(hflip_p))

        if jitter_params is not None:
            self.transforms_list.append(self._color_jitter(jitter_params))

        self.transforms_list.append(self._final_resize())
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

    def _horizontal_flip(self, p: float) -> PairTransform:
        def transform(
            image: torch.Tensor, mask: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            if random.random() < p:
                image = F.hflip(image)
                mask = F.hflip(mask)
            return image, mask

        return transform

    def _color_jitter(self, params: tuple[float, float, float, float]) -> PairTransform:
        cj = ColorJitter(*params)

        def transform(
            image: torch.Tensor, mask: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            return cj(image), mask

        return transform

    def _final_resize(self) -> PairTransform:
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
    def __init__(self, size: int = 512, use_ccrop: bool = False) -> None:
        self.size = size
        self.use_ccrop = use_ccrop

        self.normalize = Normalize(
            mean=[0.47087333, 0.44731208, 0.40772682],
            std=[0.2517867, 0.2472999, 0.25216556],
        )

    def __call__(
        self, image: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.use_ccrop:
            image = Resize(self.size + 32)(image)
            mask = Resize(self.size + 32, interpolation=F.InterpolationMode.NEAREST)(
                mask
            )
            image = F.center_crop(image, [self.size, self.size])
            mask = F.center_crop(mask, [self.size, self.size])
        else:
            image = Resize((self.size, self.size))(image)
            mask = Resize(
                (self.size, self.size), interpolation=F.InterpolationMode.NEAREST
            )(mask)

        image = self.normalize(image)

        return image, mask
