<<<<<<< feat/optuna
from dataclasses import dataclass, field
from typing import List
=======
from dataclasses import dataclass
>>>>>>> main

from training.configs.baseline import BaselineConfig


@dataclass
class _UNetArchBase(BaselineConfig):
    model: str = "unet"
    pretrained: bool = False
    num_classes: int = 5
    loss: str = "dice"
    lr: float = 3e-4
    weight_decay: float = 1e-4
<<<<<<< feat/optuna
    num_epochs: int = 12  # ~40% of 30
=======
    num_epochs: int = 25
>>>>>>> main
    batch_size: int = 8
    use_cosine_schedule: bool = True
    warmup_epochs: int = 2
    compile_model: bool = True
    channels_last: bool = True
<<<<<<< feat/optuna
    use_wandb: bool = True
    wandb_project: str = "semantic-segmentation"
    wandb_group: str = "abl_unet_arch"
    wandb_tags: List[str] = field(
        default_factory=lambda: ["ablation", "architecture", "unet"]
    )
=======
    use_wandb: bool = False
    wandb_project: str = "seg-arch-ablations"
>>>>>>> main


@dataclass
class AblationUNetNoSkip(_UNetArchBase):
    name: str = "unet_no_skip"
    unet_variant: str = "no_skip"


@dataclass
class AblationUNetShallow(_UNetArchBase):
    name: str = "unet_shallow"
    unet_variant: str = "shallow"


@dataclass
class AblationUNetWide(_UNetArchBase):
    name: str = "unet_wide"
    unet_variant: str = "wide"
    batch_size: int = 4


@dataclass
class AblationUNetNarrow(_UNetArchBase):
    name: str = "unet_narrow"
    unet_variant: str = "narrow"


@dataclass
class AblationUNetNoBN(_UNetArchBase):
    name: str = "unet_no_bn"
    unet_variant: str = "no_bn"
    lr: float = 1e-4


@dataclass
class AblationUNetResidual(_UNetArchBase):
    name: str = "unet_residual"
    unet_variant: str = "residual"


@dataclass
class AblationUNetDeepBottleneck(_UNetArchBase):
    name: str = "unet_deep_bottleneck"
    unet_variant: str = "deep_bottleneck"
    batch_size: int = 4


Config = AblationUNetNoSkip
