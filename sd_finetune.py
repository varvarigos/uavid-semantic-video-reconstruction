import albumentations as A
import hydra
import torch
from diffusers.optimization import get_scheduler

# from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate

from config import (  # DataloaderConfig,; DatasetConfig,; LearningRateConfig,; LRSchedulerConfig,; ModelConfig,; OptimizerConfig,
    TrainerConfig,
)
from datasets import UavidDatasetWithTransform, uavid_collate_fn
from models import (
    ControlNet,
    LSTMModel,
    Mapper,
    StableDiffusion1xImageVariation,
)
from trainer import Trainer

# cs = ConfigStore.instance()
# cs.store(name="base_trainer", node=TrainerConfig)xw
# cs.store(name="base_model", node=ModelConfig)
# cs.store(name="base_learning_rate", node=LearningRateConfig)
# cs.store(name="base_lr_scheduler", node=LRSchedulerConfig)
# cs.store(name="base_optimizer", node=OptimizerConfig)
# cs.store(name="base_dataset", node=DatasetConfig)
# cs.store(name="base_dataloader", node=DataloaderConfig)


@hydra.main(
    version_base=None, config_path="conf", config_name="train_config_sd_ft"
)
def main(cfg: TrainerConfig) -> None:
    cfg = instantiate(cfg)

    # copy the confa env yml
    # create the conda env yml (conda export > conda_env.yml)
    # exit()

    # Dataset
    transform = A.ReplayCompose(
        [
            A.Rotate(limit=90, crop_border=True),
            A.HorizontalFlip(),
            A.RandomResizedCrop((2160, 3840), scale=(0.6, 1.0)),
        ],
        additional_targets={"prev_frame": "image", "seg_map": "mask"},
    )

    train_dataset = UavidDatasetWithTransform(
        path=cfg.dataset.dataset_path / "uavid_train",
        size=cfg.dataset.resolution,
        center_crop=cfg.dataset.center_crop,
        transform=transform,
        max_previous_frames=1,
        oracle=True,
    )

    # Model creation
    model = StableDiffusion1xImageVariation(
        model_name=cfg.model.model_name,
        model_revision=cfg.model.model_revision,
        image_encoder_name=cfg.model.image_encoder_name,
        image_encoder_revision=cfg.model.image_encoder_revision,
        device=cfg.device,
        dtype=cfg.dtype,
        train_lora_adapter=cfg.model.train_unet,
        controlnet=None,
        mapper=None,
        lstm=None,
    )

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if cfg.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # Optimizer creation
    params_to_optimize = []
    if cfg.model.train_unet:
        params_to_optimize.append(
            {
                "params": model.unet_trainable_parameters,
                "lr": cfg.lr.unet
                * cfg.gradient_accumulation_steps
                * cfg.dataloader.train_batch_size
                * cfg.num_processes,
            }
        )

    optimizer = torch.optim.AdamW(
        params_to_optimize,
        betas=(cfg.optimizer.adam_beta1, cfg.optimizer.adam_beta2),
        weight_decay=cfg.optimizer.adam_weight_decay,
        eps=cfg.optimizer.adam_epsilon,
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.dataloader.train_batch_size,
        shuffle=True,
        num_workers=cfg.dataloader.num_workers,
        collate_fn=uavid_collate_fn,
    )
    cfg.post_dataloader_init(train_dataloader)

    # val_dataloader ....
    val_dataset = UavidDatasetWithTransform(
        path=cfg.dataset.dataset_path / "uavid_val",
        size=cfg.dataset.resolution,
        center_crop=cfg.dataset.center_crop,
        indices=[0, 5, 10, 15],  # 20, 25, 30, 35, 40, 45, 50, 55, 60],
        max_previous_frames=cfg.dataset.max_previous_frames,
        oracle=True,
    )

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=cfg.dataloader.val_batch_size,
        shuffle=False,
        num_workers=cfg.dataloader.num_workers,
        collate_fn=uavid_collate_fn,
    )

    # max_train_steps is updated in cfg.post_dataloader_init
    lr_scheduler = get_scheduler(
        cfg.lr_scheduler.scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=cfg.lr_scheduler.warmup_steps * cfg.num_processes,
        num_training_steps=cfg.max_train_steps * cfg.num_processes,
        num_cycles=cfg.lr_scheduler.num_cycles,
        power=cfg.lr_scheduler.power,
    )

    trainer = Trainer(cfg)
    trainer.fit(
        model,
        train_dataloader,
        val_dataloader,
        optimizer,
        lr_scheduler,
    )


if __name__ == "__main__":
    main()