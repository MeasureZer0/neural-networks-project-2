from __future__ import annotations

import argparse
import random
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def read_split(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def write_split(path: Path, items: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(f"{item}\n")


def collect_extra_unlabeled(extra_dir: Path) -> list[str]:
    return sorted(
        str(path)
        for path in extra_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def create_ssl_split(
    train_split: Path,
    label_ratio: float,
    seed: int,
    out_dir: Path,
    extra_unlabeled_dir: Path | None = None,
) -> None:
    if not 0.0 < label_ratio <= 1.0:
        raise ValueError("--label-ratio must be in the interval (0, 1].")

    ids = read_split(train_split)
    if not ids:
        raise ValueError(f"No samples found in {train_split}.")

    rng = random.Random(seed)
    shuffled = ids.copy()
    rng.shuffle(shuffled)

    labeled_count = max(1, round(len(ids) * label_ratio))
    labeled_ids = set(shuffled[:labeled_count])

    labeled = [sample_id for sample_id in ids if sample_id in labeled_ids]
    unlabeled = [sample_id for sample_id in ids if sample_id not in labeled_ids]

    write_split(out_dir / "labeled.txt", labeled)
    write_split(out_dir / "unlabeled.txt", unlabeled)

    if extra_unlabeled_dir is not None:
        extra = collect_extra_unlabeled(extra_unlabeled_dir)
        write_split(out_dir / "unlabeled_extra.txt", extra)

    print(f"Wrote {len(labeled)} labeled ids to {out_dir / 'labeled.txt'}")
    print(f"Wrote {len(unlabeled)} unlabeled ids to {out_dir / 'unlabeled.txt'}")
    if extra_unlabeled_dir is not None:
        print(f"Wrote extra unlabeled paths to {out_dir / 'unlabeled_extra.txt'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create labeled/unlabeled splits for semi-supervised training."
    )
    parser.add_argument("--train-split", type=Path, required=True)
    parser.add_argument("--label-ratio", type=float, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--extra-unlabeled-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    create_ssl_split(
        train_split=args.train_split,
        label_ratio=args.label_ratio,
        seed=args.seed,
        out_dir=args.out_dir,
        extra_unlabeled_dir=args.extra_unlabeled_dir,
    )


if __name__ == "__main__":
    main()
