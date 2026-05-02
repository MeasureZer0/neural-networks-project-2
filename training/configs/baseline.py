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
    device: str = "cuda"
    precision: str = "bfp16"
    use_wandb: bool = False
    wandb_project: str = "semantic-segmentation"
    checkpoint_dir: str = "checkpoints"
    data_dir: Path = Path("data/landcover.ai.v1/output")
    train_split_file: Path = Path("data/landcover.ai.v1/train.txt")
    val_split_file: Path = Path("data/landcover.ai.v1/val.txt")
    test_split_file: Path = Path("data/landcover.ai.v1/test.txt")
    use_cosine_schedule: bool = False
    warmup_epochs: int = 1
    compile_model: bool = True
    channels_last: bool = True
