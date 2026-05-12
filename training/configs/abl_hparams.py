from dataclasses import dataclass

from training.configs.baseline import BaselineConfig


@dataclass
class _HPBase(BaselineConfig):
    model: str = "fpn"
    pretrained: bool = True
    freeze_backbone: bool = False
    num_classes: int = 5
    loss: str = "dice"
    num_epochs: int = 25
    compile_model: bool = True
    channels_last: bool = True
    use_wandb: bool = False
    wandb_project: str = "seg-hp-ablations"


@dataclass
class AblationLR1e5(_HPBase):
    name: str = "fpn_lr_1e5"
    lr: float = 1e-5
    batch_size: int = 8
    use_cosine_schedule: bool = True
    warmup_epochs: int = 2


@dataclass
class AblationLR1e4(_HPBase):
    name: str = "fpn_lr_1e4"
    lr: float = 1e-4
    batch_size: int = 8
    use_cosine_schedule: bool = True
    warmup_epochs: int = 2


@dataclass
class AblationLR1e3(_HPBase):
    name: str = "fpn_lr_1e3"
    lr: float = 1e-3
    batch_size: int = 8
    use_cosine_schedule: bool = True
    warmup_epochs: int = 2


@dataclass
class AblationLR5e3(_HPBase):
    name: str = "fpn_lr_5e3"
    lr: float = 5e-3
    batch_size: int = 8
    use_cosine_schedule: bool = True
    warmup_epochs: int = 3


@dataclass
class AblationSchedConstant(_HPBase):
    name: str = "fpn_sched_constant"
    lr: float = 1e-3
    batch_size: int = 8
    use_cosine_schedule: bool = False


@dataclass
class AblationSchedCosineNoWarmup(_HPBase):
    name: str = "fpn_sched_cosine_nowarmup"
    lr: float = 1e-3
    batch_size: int = 8
    use_cosine_schedule: bool = True
    warmup_epochs: int = 0


@dataclass
class AblationSchedCosineShortWarmup(_HPBase):
    name: str = "fpn_sched_cosine_warmup1"
    lr: float = 1e-3
    batch_size: int = 8
    use_cosine_schedule: bool = True
    warmup_epochs: int = 1


@dataclass
class AblationSchedCosineLongWarmup(_HPBase):
    name: str = "fpn_sched_cosine_warmup5"
    lr: float = 1e-3
    batch_size: int = 8
    use_cosine_schedule: bool = True
    warmup_epochs: int = 5


@dataclass
class AblationBatch4(_HPBase):
    name: str = "fpn_batch4"
    lr: float = 5e-4
    batch_size: int = 4
    use_cosine_schedule: bool = True
    warmup_epochs: int = 2


@dataclass
class AblationBatch8(_HPBase):
    name: str = "fpn_batch8"
    lr: float = 1e-3
    batch_size: int = 8
    use_cosine_schedule: bool = True
    warmup_epochs: int = 2


@dataclass
class AblationBatch16(_HPBase):
    name: str = "fpn_batch16"
    lr: float = 2e-3
    batch_size: int = 16
    use_cosine_schedule: bool = True
    warmup_epochs: int = 3


@dataclass
class AblationWD0(_HPBase):
    name: str = "fpn_wd_0"
    lr: float = 1e-3
    batch_size: int = 8
    weight_decay: float = 0.0
    use_cosine_schedule: bool = True
    warmup_epochs: int = 2


@dataclass
class AblationWD1e5(_HPBase):
    name: str = "fpn_wd_1e5"
    lr: float = 1e-3
    batch_size: int = 8
    weight_decay: float = 1e-5
    use_cosine_schedule: bool = True
    warmup_epochs: int = 2


@dataclass
class AblationWD1e4(_HPBase):
    name: str = "fpn_wd_1e4"
    lr: float = 1e-3
    batch_size: int = 8
    weight_decay: float = 1e-4
    use_cosine_schedule: bool = True
    warmup_epochs: int = 2


@dataclass
class AblationWD1e2(_HPBase):
    name: str = "fpn_wd_1e2"
    lr: float = 1e-3
    batch_size: int = 8
    weight_decay: float = 1e-2
    use_cosine_schedule: bool = True
    warmup_epochs: int = 2


@dataclass
class AblationPrecisionFP32(_HPBase):
    name: str = "fpn_fp32"
    lr: float = 1e-3
    batch_size: int = 8
    precision: str = "fp32"
    use_cosine_schedule: bool = True
    warmup_epochs: int = 2


@dataclass
class AblationPrecisionBF16(_HPBase):
    name: str = "fpn_bf16"
    lr: float = 1e-3
    batch_size: int = 8
    precision: str = "bf16"
    use_cosine_schedule: bool = True
    warmup_epochs: int = 2


@dataclass
class AblationPrecisionFP16(_HPBase):
    name: str = "fpn_fp16"
    lr: float = 1e-3
    batch_size: int = 8
    precision: str = "fp16"
    use_cosine_schedule: bool = True
    warmup_epochs: int = 2


Config = AblationLR1e3
