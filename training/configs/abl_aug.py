<<<<<<< feat/optuna
from dataclasses import dataclass, field
from typing import List
=======
from dataclasses import dataclass
>>>>>>> main

from training.configs.baseline import BaselineConfig


@dataclass
class _AugAblationBase(BaselineConfig):
    model: str = "unet"
    pretrained: bool = False
    num_classes: int = 5
    loss: str = "dice"
    lr: float = 3e-4
    weight_decay: float = 1e-4
<<<<<<< feat/optuna
    num_epochs: int = 12  # ~40% of 30 — wystarczy do porównania
=======
    num_epochs: int = 20
>>>>>>> main
    batch_size: int = 8
    use_cosine_schedule: bool = True
    warmup_epochs: int = 2
    compile_model: bool = True
    channels_last: bool = True
<<<<<<< feat/optuna
    use_wandb: bool = True
    wandb_project: str = "semantic-segmentation"
    wandb_group: str = "abl_augmentation"
    wandb_tags: List[str] = field(default_factory=lambda: ["ablation", "augmentation"])
=======
    use_wandb: bool = False
    wandb_project: str = "semantic-segmentation-ablations"
>>>>>>> main


@dataclass
class AblationAugNone(_AugAblationBase):
    name: str = "ablation_aug_none"
    hflip_p: float = 0.0
    vflip_p: float = 0.0
    rotate90_p: float = 0.0
    color_jitter: bool = False
    crop_scale_min: float = 1.0
    crop_scale_max: float = 1.0


@dataclass
class AblationAugLight(_AugAblationBase):
    name: str = "ablation_aug_light"
    hflip_p: float = 0.5
    vflip_p: float = 0.5
    rotate90_p: float = 0.5
    color_jitter: bool = False
    crop_scale_min: float = 1.0
    crop_scale_max: float = 1.0


@dataclass
<<<<<<< feat/optuna
class AblationAugNoCrop(_AugAblationBase):
    name: str = "ablation_aug_no_crop"
=======
class AblationAugHeavy(_AugAblationBase):
    name: str = "ablation_aug_heavy"
>>>>>>> main
    hflip_p: float = 0.5
    vflip_p: float = 0.5
    rotate90_p: float = 0.5
    color_jitter: bool = True
<<<<<<< feat/optuna
    crop_scale_min: float = 1.0
=======
    jitter_brightness: float = 0.6
    jitter_contrast: float = 0.6
    jitter_saturation: float = 0.4
    jitter_hue: float = 0.15
    crop_scale_min: float = 0.5
>>>>>>> main
    crop_scale_max: float = 1.0


@dataclass
<<<<<<< feat/optuna
class AblationAugHeavy(_AugAblationBase):
    name: str = "ablation_aug_heavy"
=======
class AblationAugNoCrop(_AugAblationBase):
    name: str = "ablation_aug_no_crop"
>>>>>>> main
    hflip_p: float = 0.5
    vflip_p: float = 0.5
    rotate90_p: float = 0.5
    color_jitter: bool = True
<<<<<<< feat/optuna
    jitter_brightness: float = 0.6
    jitter_contrast: float = 0.6
    jitter_saturation: float = 0.4
    jitter_hue: float = 0.15
    crop_scale_min: float = 0.5
=======
    crop_scale_min: float = 1.0
>>>>>>> main
    crop_scale_max: float = 1.0


Config = AblationAugHeavy
