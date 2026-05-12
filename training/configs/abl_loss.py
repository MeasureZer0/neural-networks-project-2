from dataclasses import dataclass

from training.configs.baseline import BaselineConfig


@dataclass
class _LossAblationBase(BaselineConfig):
    model: str = "fpn"
    pretrained: bool = True
    freeze_backbone: bool = False
    num_classes: int = 5
    lr: float = 1e-3
    weight_decay: float = 1e-4
    num_epochs: int = 20  # shorter runs for ablations
    batch_size: int = 8
    use_cosine_schedule: bool = True
    warmup_epochs: int = 2
    compile_model: bool = True
    channels_last: bool = True
    use_wandb: bool = True
    wandb_project: str = "semantic-segmentation-ablations"


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
