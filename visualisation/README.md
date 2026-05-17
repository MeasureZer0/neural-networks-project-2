# Visualisation

This folder contains the Flask-based model showcase for the LandCover.ai
segmentation checkpoints in this repository.

## Structure

- `app.py`: top-level entry point
- `demo_app/`: Flask routes, inference code, templates, and styles

## Run

From the repository root:

```bash
uv run python visualisation/app.py
```

## What the app does

- discovers available `.pth` checkpoints from `checkpoints/`
- rebuilds the saved model architecture from checkpoint config metadata
- lets you pick a LandCover training tile from `data/landcover.ai.v1/train.txt`
- loads the paired image and mask from `data/landcover.ai.v1/output/`
- renders the predicted mask, overlay, class coverage, and per-image metrics
- still supports custom image uploads, with an optional custom mask

## LandCover labels

- `0`: Background
- `1`: Building
- `2`: Woodland
- `3`: Water
- `4`: Road
