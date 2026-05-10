from dataclasses import dataclass

from training.configs.baseline import BaselineConfig


@dataclass
class _PretrainBase(BaselineConfig):
    model: str = "fpn"
    num_classes: int = 5
    loss: str = "dice"
    lr: float = 1e-3
    weight_decay: float = 1e-4
    num_epochs: int = 30
    batch_size: int = 8
    use_cosine_schedule: bool = True
    warmup_epochs: int = 2
    compile_model: bool = True
    channels_last: bool = True
    use_wandb: bool = True
    wandb_project: str = "semantic-segmentation-ablations"


@dataclass
class AblationPretrained(_PretrainBase):
    name: str = "ablation_pretrained"
    pretrained: bool = True


@dataclass
class AblationScratch(_PretrainBase):
    name: str = "ablation_scratch"
    pretrained: bool = False
    lr: float = 3e-4


Config = AblationPretrained
