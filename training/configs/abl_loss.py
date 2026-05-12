<<<<<<< feat/optuna
from dataclasses import dataclass, field
from typing import List
=======
from dataclasses import dataclass
>>>>>>> main

from training.configs.baseline import BaselineConfig


@dataclass
class _LossAblationBase(BaselineConfig):
    model: str = "fpn"
    pretrained: bool = True
    freeze_backbone: bool = False
    num_classes: int = 5
    lr: float = 1e-3
    weight_decay: float = 1e-4
<<<<<<< feat/optuna
    num_epochs: int = 12  # ~40% of 30
=======
    num_epochs: int = 20  # shorter runs for ablations
>>>>>>> main
    batch_size: int = 8
    use_cosine_schedule: bool = True
    warmup_epochs: int = 2
    compile_model: bool = True
    channels_last: bool = True
    use_wandb: bool = True
<<<<<<< feat/optuna
    wandb_project: str = "semantic-segmentation"
    wandb_group: str = "abl_loss"
    wandb_tags: List[str] = field(default_factory=lambda: ["ablation", "loss"])
=======
    wandb_project: str = "semantic-segmentation-ablations"
>>>>>>> main


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


Config = AblationLossDice
