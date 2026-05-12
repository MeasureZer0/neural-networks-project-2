from dataclasses import dataclass, field
from typing import List

from training.configs.baseline import BaselineConfig


@dataclass
class UNetConfig(BaselineConfig):
    name: str = "unet_baseline"
    model: str = "unet"
    pretrained: bool = False
    num_classes: int = 5
    loss: str = "dice"
    lr: float = 3e-4
    weight_decay: float = 1e-4
    num_epochs: int = 30
    batch_size: int = 8
    use_cosine_schedule: bool = True
    warmup_epochs: int = 2
    compile_model: bool = True
    channels_last: bool = True
    use_wandb: bool = True
    wandb_project: str = "semantic-segmentation"
    wandb_group: str = "baselines"
    wandb_tags: List[str] = field(default_factory=lambda: ["baseline", "unet"])
