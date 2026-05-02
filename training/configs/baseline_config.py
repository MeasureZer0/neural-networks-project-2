from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    name: str = "baseline"
    weight_decay: float = 1e-4
    lr: float = 1e-3
    num_epochs: int = 10
    batch_size: int = 16
    num_workers: int = 4
    device: str = "cuda"  # or "cpu"
    precision: str = "fp16"  # or "fp32"
    use_wandb: bool = False
    wandb_project: str = "semantic-segmentation"
    checkpoint_dir: str = "checkpoints"
    train_image_dir: Path = Path("data/train/images")
    val_image_dir: Path = Path("data/val/images")
    test_image_dir: Path = Path("data/test/images")
    train_split_file: Path = Path("data/train.txt")
    val_split_file: Path = Path("data/val.txt")
    test_split_file: Path = Path("data/test.txt")
