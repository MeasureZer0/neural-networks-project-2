from dataclasses import dataclass

from training.configs.baseline import BaselineConfig


@dataclass
class Config(BaselineConfig):
    name: str = "unet"
    num_classes: int = 5
    lr: float = 3e-4
    num_epochs: int = 10
    batch_size: int = 8
    use_cosine_schedule: bool = True
    warmup_epochs: int = 2
    compile_model: bool = True
    channels_last: bool = True
