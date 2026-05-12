from dataclasses import dataclass, field
from typing import List

from training.configs.baseline import BaselineConfig


@dataclass
class _DLV3ArchBase(BaselineConfig):
    model: str = "deeplabv3"
    pretrained: bool = True
    freeze_backbone: bool = False
    num_classes: int = 5
    loss: str = "dice"
    lr: float = 1e-4
    weight_decay: float = 1e-4
    num_epochs: int = 12  # ~40% of 30
    batch_size: int = 8
    use_cosine_schedule: bool = True
    warmup_epochs: int = 3
    compile_model: bool = False
    channels_last: bool = True
    use_wandb: bool = True
    wandb_project: str = "semantic-segmentation"
    wandb_group: str = "abl_deeplabv3_arch"
    wandb_tags: List[str] = field(
        default_factory=lambda: ["ablation", "architecture", "deeplabv3"]
    )


@dataclass
class AblationDLV3NoASPP(_DLV3ArchBase):
    name: str = "dlv3_no_aspp"
    dlv3_variant: str = "no_aspp"


@dataclass
class AblationDLV3NoGlobalPool(_DLV3ArchBase):
    name: str = "dlv3_no_global_pool"
    dlv3_variant: str = "no_global_pool"


@dataclass
class AblationDLV3NarrowASPP(_DLV3ArchBase):
    name: str = "dlv3_narrow_aspp"
    dlv3_variant: str = "narrow_aspp"


@dataclass
class AblationDLV3SmallDilations(_DLV3ArchBase):
    name: str = "dlv3_small_dil"
    dlv3_variant: str = "small_dil"


@dataclass
class AblationDLV3LargeDilations(_DLV3ArchBase):
    name: str = "dlv3_large_dil"
    dlv3_variant: str = "large_dil"


@dataclass
class AblationDLV3ShallowHead(_DLV3ArchBase):
    name: str = "dlv3_shallow_head"
    dlv3_variant: str = "shallow_head"


@dataclass
class AblationDLV3Dropout(_DLV3ArchBase):
    name: str = "dlv3_dropout"
    dlv3_variant: str = "dropout"


@dataclass
class AblationDLV3Cascaded(_DLV3ArchBase):
    name: str = "dlv3_cascaded"
    dlv3_variant: str = "cascaded"


Config = AblationDLV3NoASPP
