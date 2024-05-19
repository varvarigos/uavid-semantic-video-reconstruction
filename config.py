import math
from dataclasses import dataclass, field

# from datetime import datetime
from pathlib import Path

import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader


@dataclass
class MapperConfig:
    norm_layer: torch.nn.Module = field(init=False)
    activation_layer: torch.nn.Module = field(init=False)
    in_channels: int = 768
    norm_layer_type: str = "layernorm"
    hidden_channels: list = field(default_factory=lambda: [768])
    activation_layer_type: str = "relu"
    inplace: bool | None = None
    bias: bool = True
    dropout: float = 0.0
    train: bool = True

    def __post_init__(self):
        if self.norm_layer_type == "layernorm":
            self.norm_layer = torch.nn.LayerNorm
        else:
            self.norm_layer = None

        if self.activation_layer_type == "relu":
            self.activation_layer = torch.nn.ReLU
        else:
            self.activation_layer = None


@dataclass
class LearningRateConfig:
    unet: float = 1e-3
    controlnet: float = 1e-4
    mapper: float = 7.5e-3

    scale: bool = True


@dataclass
class LRSchedulerConfig:
    scheduler_type: str = "constant"
    warmup_steps: int = 500
    num_cycles: int = 1
    power: float = 1.0


@dataclass
class OptimizerConfig:
    optimizer_class: str = "adam"
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_weight_decay: float = 1e-2
    adam_epsilon: float = 1e-08


@dataclass
class DatasetConfig:
    dataset_name: str = "uavid"
    dataset_path: Path = Path("./uavid")
    resolution: int = 512
    center_crop: bool = True
    max_previous_frames: int | None = None

    def __post_init__(self):
        self.dataset_path = Path(self.dataset_path)


@dataclass
class DataloaderConfig:
    train_batch_size: int = 6
    val_batch_size: int = 4
    num_workers: int = 4


@dataclass
class ModelConfig:
    model_name: str = "lambdalabs/sd-image-variations-diffusers"
    model_revision: str = "v2.0"
    image_encoder_name: str = "lambdalabs/sd-image-variations-diffusers"
    image_encoder_revision: str = "v2.0"
    train_unet: bool = True
    train_control_net: bool = True
    lora_rank: int = 4
    use_ip_adapter: bool = False
    ip_adapter_scale: float | None = None
    use_mapper: bool = True


@dataclass
class TrainerConfig:
    num_update_steps_per_epoch: int = field(init=False)
    total_batch_size: int = field(init=False)

    gradient_accumulation_steps: int = 1
    num_train_epochs: int = 60
    max_train_steps: int | None = None
    resume_from_checkpoint: Path | None = None
    checkpointing_steps: int = 500
    checkpoints_total_limit: int | None = None
    prediction_steps: int = 50
    predictions_dir_name: str = "predictions"

    max_grad_norm: float = 1.0

    output_dir: Path = Path("output")
    local_rank: int = -1

    num_processes: int = 1

    logging_dir_name: str = "logs"
    allow_tf32: bool = False
    logger: str = "tensorboard"

    mapper: MapperConfig = field(default_factory=MapperConfig)
    lr: LearningRateConfig = field(default_factory=LearningRateConfig)
    lr_scheduler: LRSchedulerConfig = field(default_factory=LRSchedulerConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    dataloader: DataloaderConfig = field(default_factory=DataloaderConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)

    device: str = "cuda"
    dtype: str = "fp16"

    overrode_max_train_steps: bool = False

    def __post_init__(self):
        self.device = torch.device(self.device)
        self.dtype = torch.float16 if self.dtype == "fp16" else torch.float32

        self.resume_from_checkpoint = (
            Path(self.resume_from_checkpoint)
            if self.resume_from_checkpoint is not None
            else None
        )

        # date = datetime.today().strftime("%Y-%m-%d__%H-%M-%S")
        self.output_dir = Path(self.output_dir)
        # self.output_dir = self.output_dir / date
        # self.output_dir.mkdir(exist_ok=True, parents=True)
        self.logging_dir = self.output_dir / self.logging_dir_name
        self.logging_dir.mkdir(exist_ok=True, parents=True)
        self.predictions_dir = self.output_dir / self.predictions_dir_name
        self.predictions_dir.mkdir(exist_ok=True, parents=True)

        if self.lr.scale:
            self.lr.unet = (
                self.lr.unet
                * self.gradient_accumulation_steps
                * self.dataloader.train_batch_size
                * self.num_processes
            )

    def post_accelerator_init(self, accelerator: Accelerator):
        self.total_batch_size = (
            self.dataloader.train_batch_size
            * accelerator.num_processes
            * self.gradient_accumulation_steps
        )

    def post_dataloader_init(self, dataloader: DataLoader):
        # Scheduler and math around the number of training steps.
        self.overrode_max_train_steps = False
        self.num_update_steps_per_epoch = math.ceil(
            len(dataloader) / self.gradient_accumulation_steps
        )
        if self.max_train_steps is None:
            self.max_train_steps = (
                self.num_train_epochs * self.num_update_steps_per_epoch
            )
            self.overrode_max_train_steps = True

    def post_prepare_init(self, dataloader: DataLoader):
        # We need to recalculate our total training steps as the size of the training dataloader may have changed.
        self.num_update_steps_per_epoch = math.ceil(
            len(dataloader) / self.gradient_accumulation_steps
        )
        if self.overrode_max_train_steps:
            self.max_train_steps = (
                self.num_train_epochs * self.num_update_steps_per_epoch
            )
        # Afterwards we recalculate our number of training epochs
        self.num_train_epochs = math.ceil(
            self.max_train_steps / self.num_update_steps_per_epoch
        )
