from pathlib import Path

import torchvision.io as io

BASE_DIR = Path(__file__).parent.resolve()
DATA_DIR = BASE_DIR / ".." / "data" / "landcover.ai.v1"
OUTPUT_DIR = DATA_DIR / "output"
SPLIT_FILES = [
    DATA_DIR / "train.txt",
    DATA_DIR / "val.txt",
    DATA_DIR / "test.txt",
]

THRESHOLD = 0.99


def get_sample_id(mask_path: Path) -> str:
    return mask_path.stem.replace("_m", "")


def main() -> None:
    to_remove = set()

    for mask_path in OUTPUT_DIR.glob("*_m.png"):
        mask = io.read_image(str(mask_path))[0]

        ratio = (mask == 0).float().mean().item()

        if ratio >= THRESHOLD:
            to_remove.add(get_sample_id(mask_path))

    print(f"Removing {len(to_remove)} samples")

    for sid in to_remove:
        (OUTPUT_DIR / f"{sid}.jpg").unlink(missing_ok=True)
        (OUTPUT_DIR / f"{sid}_m.png").unlink(missing_ok=True)

    for split_file in SPLIT_FILES:
        with open(split_file) as f:
            lines = [line.strip() for line in f]

        lines = [line for line in lines if line not in to_remove]

        with open(split_file, "w") as f:
            for line in lines:
                f.write(line + "\n")


if __name__ == "__main__":
    main()
