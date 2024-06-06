import hydra
import torch
from hydra.utils import instantiate
from tqdm import trange

from config import TrainerConfig
from datasets import (
    PredictionsUavidDataset,
    UavidDatasetWithTransform,
    uavid_collate_fn,
)
from models import ControlNet, LSTMModel, Mapper, StableDiffusion1xImageVariation
from trainer import Trainer

FUTURE_STEPS = 1  # 1 - 9


@hydra.main(version_base=None, config_path="conf", config_name="test_config")
def main(cfg: TrainerConfig) -> None:
    cfg = instantiate(cfg)

    # Model creation
    if cfg.model.use_mapper:
        mapper = Mapper(
            in_channels=cfg.mapper.in_channels,  # Number of input features
            hidden_channels=cfg.mapper.hidden_channels,  # List specifying the size of each hidden layer
            norm_layer=cfg.mapper.norm_layer,  # No normalization layer
            activation_layer=cfg.mapper.activation_layer,  # Using ReLU as the activation function
            inplace=cfg.mapper.inplace,  # In-place operation for activation
            bias=cfg.mapper.bias,  # Using bias in linear layers
            dropout=cfg.mapper.dropout,  # No dropout
            device=cfg.device,
            dtype=cfg.dtype,
            train_mapper=cfg.mapper.train,
        )
    if cfg.model.use_lstm:
        lstm = LSTMModel(
            input_size=cfg.lstm.input_size,
            num_layers=cfg.lstm.num_layers,
            hidden_size=cfg.lstm.hidden_size,
            bias=cfg.lstm.bias,
            batch_first=cfg.lstm.batch_first,
            dropout=cfg.lstm.dropout,
            bidirectional=cfg.lstm.bidirectional,
            proj_size=cfg.lstm.proj_size,
            device=cfg.device,
            dtype=cfg.dtype,
            train_lstm=cfg.lstm.train,
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
        controlnet=(
            ControlNet(
                model_name="lllyasviel/sd-controlnet-seg",
                device=cfg.device,
                dtype=cfg.dtype,
                train_lora_adapter=cfg.model.train_control_net,
                lora_rank=cfg.model.lora_rank,
            )
            if cfg.model.use_control_net
            else None
        ),
        mapper=mapper if cfg.model.use_mapper else None,
        lstm=lstm if cfg.model.use_lstm else None,
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
        prediction_steps=FUTURE_STEPS,
        shift_indices=0,
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
    to_prepare = (
        ([model.mapper] if cfg.mapper.train else [])
        + ([model.lstm] if cfg.lstm.train else [])
        + ([model.controlnet] if cfg.model.train_control_net else [])
        + ([model.unet] if cfg.model.train_unet else [])
        + [
            val_dataloader,
        ]
    )
    prepared = list(trainer.accelerator.prepare(*to_prepare))
    if cfg.mapper.train:
        model.mapper = prepared.pop(0)
    if cfg.lstm.train:
        model.lstm = prepared.pop(0)
    if cfg.model.train_control_net:
        model.controlnet = prepared.pop(0)
    if cfg.model.train_unet:
        model.unet = prepared.pop(0)
    val_dataloader = prepared.pop(0)

    # # Prepare everything with our `accelerator`.
    # if cfg.model.train_control_net and not cfg.model.train_unet:
    #     (
    #         model.controlnet.controlnet,
    #         val_dataloader,
    #     ) = trainer.accelerator.prepare(
    #         model.controlnet.controlnet,
    #         val_dataloader,
    #     )
    # elif cfg.model.train_unet and not cfg.model.train_control_net:
    #     model.unet, val_dataloader = trainer.accelerator.prepare(
    #         model.unet, val_dataloader
    #     )
    # else:
    #     (
    #         model.unet,
    #         model.controlnet.controlnet,
    #         val_dataloader,
    #     ) = trainer.accelerator.prepare(
    #         model.unet,
    #         model.controlnet.controlnet,
    #         val_dataloader,
    #     )

    trainer.cfg.post_prepare_init(val_dataloader)

    model.unet.requires_grad_(False)
    if cfg.model.use_mapper:
        model.mapper.requires_grad_(False)
    if cfg.model.use_lstm:
        model.lstm.requires_grad_(False)
    if cfg.model.use_control_net:
        model.controlnet.requires_grad_(False)

    model.eval()
    trainer.accelerator.load_state(cfg.use_checkpoint)
    predictions = []
    predictions.append(
        trainer.validation(
            model,
            val_dataloader,
            output_name=(
                "step_0"
                if val_dataset.shift_indices == 0
                else f"step_{val_dataset.shift_indices}_g"
            ),
            use_custom_inference=cfg.use_custom_inference,
            no_progress_bar=True,
            guidance_scale=cfg.guidance_scale,
        )
    )
    cur_dataset = val_dataset

    if val_dataset.shift_indices != 0:
        return

    for step in trange(1, FUTURE_STEPS):
        # update dataloader to get the predictions for the next step
        cur_dataset = PredictionsUavidDataset(
            orig_dataset=cur_dataset,
            predictions=predictions,
        )
        val_dataloader = torch.utils.data.DataLoader(
            cur_dataset,
            batch_size=cfg.dataloader.val_batch_size,
            shuffle=False,
            num_workers=cfg.dataloader.num_workers,
            collate_fn=uavid_collate_fn,
        )
        predictions.append(
            trainer.validation(
                model,
                val_dataloader,
                output_name=f"step_{step}",
                use_custom_inference=cfg.use_custom_inference,
                no_progress_bar=True,
            )
        )


if __name__ == "__main__":
    main()
