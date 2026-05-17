# FixMatch Training for LandCover.ai

This repo contains a semi-supervised segmentation path inspired by FixMatch:
<https://arxiv.org/abs/2001.07685>

The current implementation keeps the existing supervised segmentation objective and adds an unlabeled consistency term based on confidence-thresholded pseudo-labels.

## Current dataset structure

The repo currently starts from a fully labeled LandCover.ai split:

- `data/landcover.ai.v1/train.txt`: 7470 samples
- `data/landcover.ai.v1/val.txt`: 1602 samples
- `data/landcover.ai.v1/test.txt`: 1602 samples
- `data/landcover.ai.v1/output/`: labeled training images and masks used by the loaders

The supervised loader expects IDs from the split files and resolves them as:

- image: `data/landcover.ai.v1/output/<sample_id>.jpg`
- mask: `data/landcover.ai.v1/output/<sample_id>_m.png`

This structure supports the current fully labeled setup without any semi-supervised changes. If SSL is disabled, training continues to use the labeled train/val/test splits exactly as before.

## What the semi-supervised path supports

There are two supported unlabeled sources:

1. Unlabeled IDs from the same LandCover.ai training pool.
   Those entries are resolved relative to `data_dir`, usually `data/landcover.ai.v1/output/`.
2. External unlabeled imagery listed as full or relative file paths in a separate split file.
   Those entries are read directly and do not need masks.

That means the structure already supports:

- starting with all 7470 training samples labeled
- later carving out a labeled/unlabeled split from `train.txt`
- later attaching a completely new unlabeled dataset without changing the labeled dataset layout

## What is implemented

- Split generation with `scripts/create_ssl_splits.py`
- `UnlabeledLandcoverDataset` for unlabeled images
- Weak and strong augmentations for unlabeled data
- Confidence masking with threshold `0.95` by default
- Supervised loss plus unsupervised consistency loss
- Optional EMA teacher mode
- Resume support for `global_step`, scaler state, and EMA teacher weights

## Augmentation details

The current unlabeled augmentation path is defined in `torch_datasets/transforms.py` and is intentionally kept as-is.

Shared spatial transform applied once and replayed for both views:

- `RandomResizedCrop(size=(img_size, img_size), scale=(crop_scale_min, crop_scale_max), ratio=(1.0, 1.0))`
- `HorizontalFlip(p=0.5)`
- `VerticalFlip(p=0.5)`
- `RandomRotate90(p=0.5)`

Weak view:

- shared spatial transform
- `Normalize(mean, std)`
- `ToTensorV2()`

Strong view:

- shared spatial transform
- `ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)`
- `GaussianBlur(blur_limit=(3, 7), p=0.5)`
- `GaussNoise(p=0.5)`
- `Normalize(mean, std)`
- `ToTensorV2()`

This is a lighter strong augmentation than the one used in the original FixMatch paper, but it is the augmentation stack currently implemented in the repo.

## Training recipe

For each unlabeled image:

1. Apply a shared spatial transform.
2. Produce a weak view and a strong view.
3. Predict pseudo-labels from the weak view.
4. Keep only predictions above the confidence threshold.
5. Train the model on the strong view against those pseudo-labels.
6. Add the unsupervised term to the standard supervised segmentation loss.

Relevant code:

- `training/train.py`
- `training/trainer.py`
- `training/checkpointing.py`
- `torch_datasets/landcover_dataset.py`
- `torch_datasets/transforms.py`

## Creating splits from the current labeled training set

If you want to simulate a semi-supervised setup from the current `train.txt`, generate a labeled/unlabeled partition:

```bash
uv run python scripts/create_ssl_splits.py \
  --train-split data/landcover.ai.v1/train.txt \
  --label-ratio 0.1 \
  --seed 42 \
  --out-dir data/landcover.ai.v1/ssl_10
```

This writes:

- `data/landcover.ai.v1/ssl_10/labeled.txt`
- `data/landcover.ai.v1/ssl_10/unlabeled.txt`
- `data/landcover.ai.v1/ssl_10/unlabeled_extra.txt` if `--extra-unlabeled-dir` is used

## Using a future external unlabeled dataset

If you later add a separate unlabeled source, place one image path per line in a file such as:

- `data/landcover.ai.v1/ssl_10/unlabeled_extra.txt`

Then set only:

- `semi_supervised.extra_unlabeled_split_file`

or combine it with:

- `semi_supervised.unlabeled_split_file`

The code now supports either source independently, or both together.

Reference candidate for future unlabeled imagery:
<https://zenodo.org/records/7223446>

## Example config

```python
from dataclasses import dataclass, field
from pathlib import Path

from training.configs.baseline import BaselineConfig, SemiSupervisedConfig


@dataclass
class FixMatchFPNConfig(BaselineConfig):
    name: str = "fpn_fixmatch"
    model: str = "fpn"
    batch_size: int = 8
    num_epochs: int = 30
    semi_supervised: SemiSupervisedConfig = field(
        default_factory=lambda: SemiSupervisedConfig(
            enabled=True,
            unlabeled_split_file=Path("data/landcover.ai.v1/ssl_10/unlabeled.txt"),
            extra_unlabeled_split_file=None,
            unlabeled_batch_ratio=1,
            threshold=0.95,
            lambda_u=1.0,
            unsup_warmup_epochs=5,
            use_ema_teacher=False,
        )
    )
```

Ready-made SSL config modules are also provided:

- `training/configs/fpn_fixmatch_config.py`
- `training/configs/unet_fixmatch_config.py`
- `training/configs/deeplabv3_fixmatch_config.py`

These expect split files such as `data/landcover.ai.v1/ssl_10/labeled.txt` and `data/landcover.ai.v1/ssl_10/unlabeled.txt` to exist first.

## Running

Standard supervised training:

```bash
uv run python -m training.train --config fpn_config
```

Semi-supervised training:

```bash
uv run python -m training.train --config your_fixmatch_config
```

## Notes

- Augmentations are intentionally left as currently implemented in `torch_datasets/transforms.py`.
- This is a segmentation adaptation of FixMatch, so pseudo-labeling is done per pixel rather than per image.
- EMA teacher mode is optional and disabled by default.
