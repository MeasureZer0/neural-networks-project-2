<<<<<<< feat/optuna
from dataclasses import dataclass, field
from typing import List
=======
from dataclasses import dataclass
>>>>>>> main

from training.configs.baseline import BaselineConfig


@dataclass
class _FPNArchBase(BaselineConfig):
    model: str = "fpn"
    pretrained: bool = True
    freeze_backbone: bool = False
    num_classes: int = 5
    loss: str = "dice"
    lr: float = 1e-3
    weight_decay: float = 1e-4
<<<<<<< feat/optuna
    num_epochs: int = 12  # ~40% of 30
=======
    num_epochs: int = 25
>>>>>>> main
    batch_size: int = 8
    use_cosine_schedule: bool = True
    warmup_epochs: int = 2
    compile_model: bool = False
    channels_last: bool = True
<<<<<<< feat/optuna
    use_wandb: bool = True
    wandb_project: str = "semantic-segmentation"
    wandb_group: str = "abl_fpn_arch"
    wandb_tags: List[str] = field(
        default_factory=lambda: ["ablation", "architecture", "fpn"]
    )
=======
    use_wandb: bool = False
    wandb_project: str = "seg-arch-ablations"
>>>>>>> main


@dataclass
class AblationFPNNoLateral(_FPNArchBase):
    name: str = "fpn_no_lateral"
    fpn_variant: str = "no_lateral"


@dataclass
class AblationFPNSingleScale(_FPNArchBase):
    name: str = "fpn_single_scale"
    fpn_variant: str = "single_scale"


@dataclass
class AblationFPNConcatMerge(_FPNArchBase):
    name: str = "fpn_concat_merge"
    fpn_variant: str = "concat_merge"


@dataclass
class AblationFPNShallowHead(_FPNArchBase):
    name: str = "fpn_shallow_head"
    fpn_variant: str = "shallow_head"


@dataclass
class AblationFPNNoP5(_FPNArchBase):
    name: str = "fpn_no_p5"
    fpn_variant: str = "no_p5"


Config = AblationFPNNoLateral
