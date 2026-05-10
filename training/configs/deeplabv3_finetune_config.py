from dataclasses import dataclass

from training.configs.baseline import BaselineConfig


@dataclass
class Config(BaselineConfig):
    name: str = "deeplabv3_finetune"
    model: str = "deeplabv3"
    pretrained: bool = True
    freeze_backbone: bool = False
    num_classes: int = 5
    loss: str = "dice"
    lr: float = 1e-4
    weight_decay: float = 1e-4
    num_epochs: int = 30
    batch_size: int = 8
    use_cosine_schedule: bool = True
    warmup_epochs: int = 3
    compile_model: bool = True
    channels_last: bool = True
    use_wandb: bool = True
