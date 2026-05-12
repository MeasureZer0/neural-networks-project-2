from dataclasses import dataclass, field
from typing import List

from training.configs.baseline import BaselineConfig


@dataclass
class DeepLabV3FinetuneConfig(BaselineConfig):
    name: str = "deeplabv3_finetune"
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
    wandb_group: str = "baselines"
    wandb_tags: List[str] = field(
        default_factory=lambda: ["baseline", "deeplabv3", "finetune"]
    )
