from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class SemiSupervisedConfig:
    enabled: bool = False
    unlabeled_split_file: Path | None = None
    extra_unlabeled_split_file: Path | None = None
    unlabeled_batch_ratio: int = 1
    threshold: float = 0.95
    lambda_u: float = 1.0
    unsup_warmup_epochs: int = 5
    use_ema_teacher: bool = True
    ema_decay: float = 0.996
    use_dual_strong_views: bool = False


@dataclass
class BaselineConfig:
    name: str = "baseline"
    description: str = ""

    model: str = "fpn"  # fpn | unet | deeplabv3
    num_classes: int = 5
    pretrained: bool = True
    freeze_backbone: bool = False

    loss: str = "dice"  # dice | ce | focal
    ce_weight: float = 0.5
    focal_alpha: float = 1.0
    focal_gamma: float = 2.0

    lr: float = 1e-3
    weight_decay: float = 1e-4
    adam_beta1: float = 0.9
    adam_beta2: float = 0.98
    adam_eps: float = 1e-6

    use_cosine_schedule: bool = True
    warmup_epochs: int = 1

    num_epochs: int = 10
    batch_size: int = 8
    num_workers: int = 4
    device: str = "cuda"
    precision: str = "bf16"  # fp32 | fp16 | bf16
    compile_model: bool = True
    channels_last: bool = True

    img_size: int = 512
    crop_scale_min: float = 0.7
    crop_scale_max: float = 1.0
    hflip_p: float = 0.5
    vflip_p: float = 0.5
    rotate90_p: float = 0.5
    color_jitter: bool = True
    jitter_brightness: float = 0.4
    jitter_contrast: float = 0.4
    jitter_saturation: float = 0.2
    jitter_hue: float = 0.1

    data_dir: Path = Path("data/landcover.ai.v1/output")
    train_split_file: Path = Path("data/landcover.ai.v1/train.txt")
    val_split_file: Path = Path("data/landcover.ai.v1/val.txt")
    test_split_file: Path = Path("data/landcover.ai.v1/test.txt")

    checkpoint_dir: str = "checkpoints"

    use_wandb: bool = True
    wandb_project: str = "semantic-segmentation"
    wandb_entity: Optional[str] = None
    wandb_tags: List[str] = field(default_factory=list)
    semi_supervised: SemiSupervisedConfig = field(default_factory=SemiSupervisedConfig)
