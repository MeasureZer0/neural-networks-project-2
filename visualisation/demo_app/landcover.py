from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

LANDCOVER_CLASS_NAMES = [
    "Background",
    "Building",
    "Woodland",
    "Water",
    "Road",
]


@dataclass(frozen=True)
class DatasetSample:
    sample_id: str
    image_path: Path
    mask_path: Path


@dataclass(frozen=True)
class CheckpointOption:
    label: str
    path: str


def discover_checkpoints(checkpoint_dir: Path) -> list[CheckpointOption]:
    options: list[CheckpointOption] = []
    for path in sorted(checkpoint_dir.glob("*.pth")):
        options.append(
            CheckpointOption(
                label=path.stem.replace("_", " "),
                path=str(path.resolve()),
            )
        )
    return options


def load_split_samples(image_dir: Path, split_file: Path) -> list[DatasetSample]:
    if not split_file.is_file():
        return []

    samples: list[DatasetSample] = []
    with split_file.open("r") as handle:
        for line in handle:
            sample_id = line.strip()
            if not sample_id:
                continue
            samples.append(
                DatasetSample(
                    sample_id=sample_id,
                    image_path=image_dir / f"{sample_id}.jpg",
                    mask_path=image_dir / f"{sample_id}_m.png",
                )
            )
    return samples


def index_samples(samples: list[DatasetSample]) -> dict[str, DatasetSample]:
    return {sample.sample_id: sample for sample in samples}
