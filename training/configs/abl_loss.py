from dataclasses import dataclass, field
from typing import List

from training.configs.baseline import BaselineConfig


@dataclass
class _LossAblationBase(BaselineConfig):
    model: str = "fpn"
    pretrained: bool = True
    freeze_backbone: bool = False
    num_classes: int = 5
    lr: float = 1e-3
    weight_decay: float = 1e-4
    num_epochs: int = 12  # ~40% of 30
    batch_size: int = 8
    use_cosine_schedule: bool = True
    warmup_epochs: int = 2
    compile_model: bool = True
    channels_last: bool = True
    use_wandb: bool = True
    wandb_project: str = "semantic-segmentation"
    wandb_group: str = "abl_loss"
    wandb_tags: List[str] = field(default_factory=lambda: ["ablation", "loss"])


@dataclass
class AblationLossCE(_LossAblationBase):
    name: str = "ablation_loss_ce"
    loss: str = "ce"


@dataclass
class AblationLossDice(_LossAblationBase):
    name: str = "ablation_loss_dice"
    loss: str = "dice"


@dataclass
class AblationLossFocal(_LossAblationBase):
    name: str = "ablation_loss_focal"
    loss: str = "focal"
    focal_alpha: float = 1.0
    focal_gamma: float = 2.0


@dataclass
class AblationLossCEDice(_LossAblationBase):
    name: str = "ablation_loss_ce_dice"
    loss: str = "ce_dice"

    ce_weight: float = 1.0
    dice_weight: float = 1.0


Config = AblationLossDice
