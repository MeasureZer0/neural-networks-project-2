from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from training.configs.baseline import BaselineConfig, SemiSupervisedConfig


@dataclass
class DeepLabV3FixMatchConfig(BaselineConfig):
    name: str = "deeplabv3_fixmatch"
    model: str = "deeplabv3"
    pretrained: bool = True
    freeze_backbone: bool = False
    num_classes: int = 5
    loss: str = "dice"
    lr: float = 1e-4
    weight_decay: float = 1e-4
    num_epochs: int = 30
    batch_size: int = 8
    use_cosine_schedule: bool = True
    warmup_epochs: int = 3
    compile_model: bool = True
    channels_last: bool = True
    use_wandb: bool = True
    wandb_project: str = "semantic-segmentation"
    wandb_group: str = "ssl"
    wandb_tags: List[str] = field(
        default_factory=lambda: ["ssl", "fixmatch", "deeplabv3"]
    )
    train_split_file: Path = Path("data/landcover.ai.v1/ssl_10/labeled.txt")
    semi_supervised: SemiSupervisedConfig = field(
        default_factory=lambda: SemiSupervisedConfig(
            enabled=True,
            unlabeled_split_file=Path("data/landcover.ai.v1/ssl_10/unlabeled.txt"),
            extra_unlabeled_split_file=None,
            unlabeled_batch_ratio=1,
            threshold=0.95,
            lambda_u=1.0,
            unsup_warmup_epochs=5,
            use_ema_teacher=True,
        )
    )
