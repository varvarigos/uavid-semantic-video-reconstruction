import hydra
import torch

# from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate

from config import (  # DataloaderConfig,; DatasetConfig,; LearningRateConfig,; LRSchedulerConfig,; ModelConfig,; OptimizerConfig,
    TrainerConfig,
)
from datasets import UavidDatasetWithTransform, uavid_collate_fn
from models import ControlNet, StableDiffusion1xImageVariation
from trainer import Trainer

# cs = ConfigStore.instance()
# cs.store(name="base_trainer", node=TrainerConfig)xw
# cs.store(name="base_model", node=ModelConfig)
# cs.store(name="base_learning_rate", node=LearningRateConfig)
# cs.store(name="base_lr_scheduler", node=LRSchedulerConfig)
# cs.store(name="base_optimizer", node=OptimizerConfig)
# cs.store(name="base_dataset", node=DatasetConfig)
# cs.store(name="base_dataloader", node=DataloaderConfig)


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: TrainerConfig) -> None:
    cfg = instantiate(cfg)

    # Model creation
    model = StableDiffusion1xImageVariation(
        model_name=cfg.model.model_name,
        model_revision=cfg.model.model_revision,
        image_encoder_name=cfg.model.image_encoder_name,
        image_encoder_revision=cfg.model.image_encoder_revision,
        device=cfg.device,
        dtype=cfg.dtype,
        train_lora_adapter=cfg.model.train_unet,
        controlnet=ControlNet(
            model_name="lllyasviel/sd-controlnet-seg",
            device=cfg.device,
            dtype=cfg.dtype,
            train_lora_adapter=cfg.model.train_control_net,
            lora_rank=cfg.model.lora_rank,
        ),
    )

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if cfg.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # val_dataloader ....
    val_dataset = UavidDatasetWithTransform(
        path=cfg.dataset.dataset_path / "uavid_val",
        size=cfg.dataset.resolution,
        center_crop=cfg.dataset.center_crop,
        indices=[0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60],
        max_previous_frames=cfg.dataset.max_previous_frames,
        oracle=cfg.dataset.oracle,
    )

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=cfg.dataloader.val_batch_size,
        shuffle=False,
        num_workers=cfg.dataloader.num_workers,
        collate_fn=uavid_collate_fn,
    )

    cfg.post_dataloader_init(val_dataloader)

    trainer = Trainer(cfg)

    # Prepare everything with our `accelerator`.
    if cfg.model.train_control_net and not cfg.model.train_unet:
        (
            model.controlnet.controlnet,
            val_dataloader,
        ) = trainer.accelerator.prepare(
            model.controlnet.controlnet,
            val_dataloader,
        )
    elif cfg.model.train_unet and not cfg.model.train_control_net:
        model.unet, val_dataloader = trainer.accelerator.prepare(
            model.unet, val_dataloader
        )
    else:
        (
            model.unet,
            model.controlnet.controlnet,
            val_dataloader,
        ) = trainer.accelerator.prepare(
            model.unet,
            model.controlnet.controlnet,
            val_dataloader,
        )

    trainer.cfg.post_prepare_init(val_dataloader)

    model.unet.requires_grad_(False)
    model.controlnet.requires_grad_(False)
    model.eval()
    trainer.accelerator.load_state(
        "/teamspace/studios/this_studio/outputs/averaging/2024-05-15__02-00-15/checkpoints/checkpoint_5"
        # "/teamspace/studios/this_studio/outputs/2024-05-14__18-04-13/single_frame_exp_2_350epochs/checkpoints/checkpoint_9"
    )
    trainer.validation(model, val_dataloader)


if __name__ == "__main__":
    main()
