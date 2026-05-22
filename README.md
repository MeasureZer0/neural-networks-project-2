# Neural Networks Project 2

Semantic segmentation experiments on the LandCover.ai dataset. The repository contains a PyTorch training pipeline for three model families, configuration-driven experiments, Optuna-based hyperparameter tuning, and supporting scripts for dataset preparation and analysis.

This README documents:

- the current `main` branch
- the pending semi-supervised branch `feat/init-semi-supervised`
- the pending visualisation branch `feat/flask-model-showcase`

Those two feature branches are not merged into `main` yet, but they are already accounted for here because they are part of the planned integration path for this repository.

## Project scope

The project focuses on land-cover semantic segmentation with five classes:

- `0`: Background
- `1`: Building
- `2`: Woodland
- `3`: Water
- `4`: Road

Core model families:

- `fpn`
- `unet`
- `deeplabv3`

The training code is config-driven and supports:

- baseline supervised training
- architecture and hyperparameter ablations
- checkpoint resume
- Weights & Biases logging
- Optuna tuning

## Repository layout

```text
.
├── models/                 # FPN, U-Net, DeepLabV3 and ablation variants
├── torch_datasets/         # LandCover dataset loaders and transforms
├── training/               # training loop, metrics, checkpointing, configs, tuning
├── scripts/                # dataset utilities and helper scripts
├── notebooks/              # exploratory work
├── data/                   # expected dataset location
├── checkpoints/            # saved checkpoints
└── visualisation/          # Flask demo app branch target
```

## Environment setup

The project is managed with `uv` and targets Python `3.13`.

1. Install `uv`.
2. Create the environment and install dependencies:

```bash
uv sync
```

3. Run commands through `uv`:

```bash
uv run python -m training.train --config fpn_config
```

Development tools included in the project:

- `ruff`
- `pyright`
- `ipykernel`

## Dataset layout

The code expects the LandCover.ai dataset in the following structure:

```text
data/
└── landcover.ai.v1/
    ├── train.txt
    ├── val.txt
    ├── test.txt
    └── output/
        ├── <sample_id>.jpg
        └── <sample_id>_m.png
```

Split files contain sample IDs without extensions. The loader resolves them as:

- image: `data/landcover.ai.v1/output/<sample_id>.jpg`
- mask: `data/landcover.ai.v1/output/<sample_id>_m.png`

`main` currently uses the labeled `train.txt`, `val.txt`, and `test.txt` splits directly.

## Training pipeline

Entry point:

- `training/train.py`

Training flow:

1. Load a config from `training/configs/`.
2. Build the requested model.
3. Build the loss function.
4. Create train and validation data loaders.
5. Optionally enable cosine scheduling with warmup.
6. Train with checkpointing and validation metrics.

Supported model CLI values:

- `fpn`
- `unet`
- `deeplabv3`

Supported loss CLI values on `main`:

- `dice`
- `ce`
- `focal`
- `ce_dice`

Example supervised runs:

```bash
uv run python -m training.train --config fpn_config
uv run python -m training.train --config unet_config
uv run python -m training.train --config deeplabv3_finetune_config
```

Resume from checkpoint:

```bash
uv run python -m training.train --config fpn_config --resume checkpoints/your_run/best.pth
```

## Available configs

Baseline and experiment configs live in `training/configs/`.

Useful starting points:

- `fpn_config`
- `unet_config`
- `deeplabv3_finetune_config`
- `deeplabv3_frozen_config`
- `baseline`

Ablation modules are also present for:

- architecture variants
- augmentation variants
- pretrained vs. non-pretrained comparisons
- loss comparisons
- hyperparameter comparisons

## Hyperparameter tuning

Entry point:

- `training/tune.py`

The tuning script uses Optuna and can optionally log results to Weights & Biases.

Example:

```bash
uv run python -m training.tune \
  --model fpn \
  --loss dice \
  --config baseline \
  --n_trials 40 \
  --n_epochs_tune 10 \
  --storage sqlite:///optuna.db
```

Enable W&B logging:

```bash
uv run python -m training.tune --model fpn --loss dice --wandb
```

## Tooling and quality checks

Lint:

```bash
uv run ruff check .
```

Type checking:

```bash
uv run pyright
```

## Current branch capabilities

`main` already provides:

- supervised semantic segmentation training
- FPN, U-Net, and DeepLabV3 implementations
- multiple ablation configs
- checkpoint save/resume support
- Optuna HPO
- W&B integration

## Pending branch: semi-supervised training

Branch:

- `feat/init-semi-supervised`

This branch extends the project with a FixMatch-style semi-supervised segmentation path built around teacher-student pseudo-labeling.

What it adds:

- split generation via `scripts/create_ssl_splits.py`
- unlabeled dataset support
- weak and strong augmentation pipelines for unlabeled data
- confidence-thresholded pseudo-labeling
- unsupervised consistency loss
- EMA teacher support
- checkpoint resume for teacher state, scaler state, and global step

Expected usage after merge:

```bash
uv run python scripts/create_ssl_splits.py \
  --train-split data/landcover.ai.v1/train.txt \
  --label-ratio 0.1 \
  --seed 42 \
  --out-dir data/landcover.ai.v1/ssl_10
```

Then train with one of the SSL configs:

- `training/configs/fpn_fixmatch_config.py`
- `training/configs/unet_fixmatch_config.py`
- `training/configs/deeplabv3_fixmatch_config.py`

The branch documentation currently lives in `semi-supervised.md` on that branch.

## Pending branch: Flask visualisation app

Branch:

- `feat/flask-model-showcase`

This branch adds a local Flask app for checkpoint inspection and side-by-side model comparison.

What it adds:

- checkpoint discovery from `checkpoints/`
- checkpoint-driven model reconstruction
- inference on LandCover test samples
- optional custom image upload
- optional custom mask upload
- predicted mask and overlay rendering
- class coverage and per-image metrics
- side-by-side checkpoint comparison

Expected app entry point after merge:

```bash
uv run python visualisation/app.py
```

If the dataset is outside the default repo location, the app uses:

```bash
LANDCOVER_DATA_ROOT=/path/to/data/landcover.ai.v1
```

The branch documentation currently lives in `visualisation/README.md` on that branch.

## Scripts and notebooks

Current utility scripts include:

- `scripts/filter.py`
- `scripts/profile.py`
- `scripts/split.py`

Exploration notebooks live in `notebooks/`.

## Notes

- The project metadata in `pyproject.toml` still contains an outdated description string and should not be treated as the functional project summary.
- The visualisation and semi-supervised sections in this README describe branch work that still needs to be merged.
