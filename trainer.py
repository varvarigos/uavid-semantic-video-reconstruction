import copy
from dataclasses import asdict
from pathlib import Path

import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from diffusers import (  # DDIMScheduler,
    PNDMScheduler,
    StableDiffusionControlNetImg2ImgPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionPipeline,
)
from diffusers.pipelines import StableDiffusionControlNetPipeline
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import TrainerConfig
from models import LSTMModel
from utils import get_parameters_stats, image_grid, tensor_to_pil


class Trainer:
    def __init__(self, config: TrainerConfig):
        self.cfg = config
        self.accelerator = Accelerator(
            project_config=ProjectConfiguration(
                project_dir=self.cfg.output_dir,
                logging_dir=self.cfg.output_dir / self.cfg.logging_dir,
                automatic_checkpoint_naming=True,
                total_limit=self.cfg.checkpoints_total_limit,
                iteration=0,
                save_on_each_node=False,
            ),
            gradient_accumulation_steps=self.cfg.gradient_accumulation_steps,
            mixed_precision="fp16",
            log_with=self.cfg.logger,
        )

        # Uncomment for loading a checkpoint
        # self.accelerator.load_state(
        #     "/teamspace/studios/this_studio/outputs/fixed/averaging/2024-05-25__03-48-51/checkpoints/checkpoint_12"
        # )

        self.cfg.post_accelerator_init(self.accelerator)
        self.logger = get_logger(__name__)

        self.global_step = 0
        self.epoch = 0

        self.accelerator.init_trackers(
            "uavid_frame_prediction",
            config={k: str(v) for k, v in asdict(self.cfg).items()},
        )

        if self.cfg.lstm.train:
            self.accelerator.register_save_state_pre_hook(self.save_lstm_hook)
            self.accelerator.register_load_state_pre_hook(self.load_lstm_hook)

    def save_lstm_hook(self, models, weights, output_dir):
        if self.accelerator.is_main_process:
            for i, model in enumerate(models):
                if isinstance(model, LSTMModel):
                    # make sure to pop weight so that corresponding model is not saved again
                    lstm_weights = weights.pop(i)
                    torch.save(lstm_weights, Path(output_dir) / "lstm.pth")

    def load_lstm_hook(self, models, input_dir):
        to_remove = []
        for i, model in enumerate(models):
            if isinstance(model, LSTMModel):
                model.load_state_dict(torch.load(Path(input_dir) / "lstm.pth"))
                to_remove.append(i)

        for i in to_remove:
            models.pop(i)

    def fit(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloder: DataLoader,
        optimizer,
        lr_scheduler,
    ):

        # Prepare everything with our `accelerator`.
        to_prepare = (
            ([model.mapper] if self.cfg.mapper.train else [])
            + ([model.lstm] if self.cfg.lstm.train else [])
            + ([model.controlnet] if self.cfg.model.train_control_net else [])
            + ([model.unet] if self.cfg.model.train_unet else [])
            + [
                optimizer,
                train_dataloader,
                lr_scheduler,
            ]
        )
        prepared = list(self.accelerator.prepare(*to_prepare))
        if self.cfg.mapper.train:
            model.mapper = prepared.pop(0)
        if self.cfg.lstm.train:
            model.lstm = prepared.pop(0)
        if self.cfg.model.train_control_net:
            model.controlnet = prepared.pop(0)
        if self.cfg.model.train_unet:
            model.unet = prepared.pop(0)
        optimizer = prepared.pop(0)
        train_dataloader = prepared.pop(0)
        lr_scheduler = prepared.pop(0)

        self.cfg.post_prepare_init(train_dataloader)

        # Train!
        self.logger.info("***** Running training *****")
        self.logger.info(f"  Num examples = {len(train_dataloader.dataset)}")
        self.logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
        self.logger.info(f"  Num Epochs = {self.cfg.num_train_epochs}")
        self.logger.info(
            f"  Instantaneous batch size per device = {self.cfg.dataloader.train_batch_size}"
        )
        self.logger.info(
            f"  Total train batch size (w. parallel, distributed & accumulation) = {self.cfg.total_batch_size}"
        )
        self.logger.info(
            f"  Gradient Accumulation steps = {self.cfg.gradient_accumulation_steps}"
        )
        self.logger.info(
            f"  Total optimization steps = {self.cfg.max_train_steps}"
        )

        # Print Parameter Stats
        unet_param_stats = get_parameters_stats(model.unet.parameters())
        self.logger.info(
            f"  Trainable UNet LoRA Parameters: {unet_param_stats['trainable']}/{unet_param_stats['all']}"
        )
        if model.controlnet:
            controlnet_param_stats = get_parameters_stats(
                model.controlnet.parameters()
            )
            self.logger.info(
                f"  Trainable ControlNet LoRA Parameters: {controlnet_param_stats['trainable']}/{controlnet_param_stats['all']}"
            )
        if self.cfg.model.use_mapper:
            mapper_param_stats = get_parameters_stats(model.mapper.parameters())
            self.logger.info(
                f"  Trainable Mapper Parameters: {mapper_param_stats['trainable']}/{mapper_param_stats['all']}"
            )
        if self.cfg.model.use_lstm:
            lstm_param_stats = get_parameters_stats(model.lstm.parameters())
            self.logger.info(
                f"  Trainable LSTM Parameters: {lstm_param_stats['trainable']}/{lstm_param_stats['all']}"
            )

        self.global_step = 0
        first_epoch = 0

        initial_global_step = 0

        progress_bar = tqdm(
            range(0, self.cfg.max_train_steps),
            initial=initial_global_step,
            desc="Steps",
            # Only show the progress bar once on each machine.
            disable=not self.accelerator.is_local_main_process,
        )

        # os.makedirs(
        #     f"runs/learning_rate_{learning_rate}_epochs_{num_train_epochs}",
        #     exist_ok=True,
        # )

        for self.epoch in range(first_epoch, self.cfg.num_train_epochs):
            model.train()

            total_loss = 0
            for step, batch in enumerate(train_dataloader):
                with self.accelerator.accumulate(model):
                    # TODO: we should not need this loop
                    for k, v in batch.items():
                        if k == "pixel_values_clip":
                            # v is a list of tensors
                            batch[k] = [
                                vi.to(device=self.cfg.device) for vi in v
                            ]
                            if torch.is_floating_point(
                                v[0]
                            ) and self.cfg.device != torch.device("cpu"):
                                batch[k] = [
                                    vi.to(dtype=self.cfg.dtype)
                                    for vi in batch[k]
                                ]
                        else:
                            batch[k] = v.to(device=self.cfg.device)
                            if torch.is_floating_point(
                                v
                            ) and self.cfg.device != torch.device("cpu"):
                                batch[k] = batch[k].to(dtype=self.cfg.dtype)

                    loss = model.training_step(batch, step)
                    total_loss += loss

                    self.accelerator.backward(loss)
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(
                            model.trainable_parameters, self.cfg.max_grad_norm
                        )
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                # Checks if the accelerator has performed an optimization step behind the scenes
                if self.accelerator.sync_gradients:
                    progress_bar.update(1)
                    self.global_step += 1

                    if self.accelerator.is_main_process:
                        if self.global_step % self.cfg.checkpointing_steps == 0:
                            self.accelerator.save_state()
                            self.logger.info("Saved state")

                        if self.global_step % self.cfg.prediction_steps == 0:
                            model.eval()

                            self.validation(
                                model=model,
                                val_dataloader=val_dataloder,
                                use_custom_inference=self.cfg.use_custom_inference,
                                guidance_scale=self.cfg.guidance_scale,
                            )
                            model.train()
                lrs = lr_scheduler.get_lr()
                logs = (
                    {"loss": loss.detach().item()}
                    | (
                        {"lr_unet": lrs.pop(0)}
                        if self.cfg.model.train_unet
                        else {}
                    )
                    | (
                        {"lr_cntrl": lrs.pop(0)}
                        if self.cfg.model.train_control_net
                        else {}
                    )
                    | (
                        {"lr_mapper": lrs.pop(0)}
                        if self.cfg.mapper.train and self.cfg.model.use_mapper
                        else {}
                    )
                    | (
                        {"lr_lstm": lrs.pop(0)}
                        if self.cfg.lstm.train and self.cfg.model.use_lstm
                        else {}
                    )
                )
                progress_bar.set_postfix(**logs)
                self.accelerator.log(logs, step=self.global_step)

                if self.global_step >= self.cfg.max_train_steps:
                    break

            print(total_loss / len(train_dataloader.dataset))
            epoch_logs = {"epoch": self.epoch, "loss_epoch": total_loss}
            self.accelerator.log(epoch_logs, step=self.global_step)

        ## Finish Up Training

        # Save the lora layers
        model.eval()

        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            self.accelerator.save_state()

        self.accelerator.end_training()

    def validation(
        self,
        model: nn.Module,
        val_dataloader: DataLoader,
        output_name: str | None = None,
        use_custom_inference: bool = True,
        no_progress_bar: bool = True,
        guidance_scale: float = 2,
    ):
        model.eval()
        generator = torch.manual_seed(42)
        if not use_custom_inference:
            if model.controlnet:
                if self.cfg.model.use_img2img_inference:
                    pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
                        "CompVis/stable-diffusion-v1-4",
                        safety_checker=None,
                        torch_dtype=self.cfg.dtype,
                        controlnet=model.controlnet.controlnet,
                        unet=model.unet,
                        vae=model.vae.to(torch.float32),
                        scheduler=model.train_noise_scheduler,
                    ).to(
                        device=self.cfg.device
                    )
                elif self.cfg.model.use_img2img_refinement:
                    pipe = StableDiffusionControlNetPipeline.from_pretrained(
                        "CompVis/stable-diffusion-v1-4",
                        safety_checker=None,
                        torch_dtype=self.cfg.dtype,
                        controlnet=model.controlnet.controlnet,
                        unet=model.unet,
                        vae=model.vae.to(torch.float32),
                    ).to(device=self.cfg.device)

                    pipe_refinement = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
                        "CompVis/stable-diffusion-v1-4",
                        safety_checker=None,
                        torch_dtype=self.cfg.dtype,
                        controlnet=copy.deepcopy(model.controlnet.controlnet),
                        unet=copy.deepcopy(model.unet),
                        vae=copy.deepcopy(model.vae.to(torch.float32)),
                        scheduler=copy.deepcopy(model.train_noise_scheduler),
                    ).to(
                        device=self.cfg.device
                    )
                else:
                    pipe = StableDiffusionControlNetPipeline.from_pretrained(
                        "CompVis/stable-diffusion-v1-4",
                        safety_checker=None,
                        torch_dtype=self.cfg.dtype,
                        controlnet=model.controlnet.controlnet,
                        unet=model.unet,
                        vae=model.vae.to(torch.float32),
                    ).to(device=self.cfg.device)
            else:
                if self.cfg.model.use_img2img_inference:
                    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                        "CompVis/stable-diffusion-v1-4",
                        safety_checker=None,
                        torch_dtype=self.cfg.dtype,
                        unet=model.unet,
                        vae=model.vae.to(torch.float32),
                        scheduler=model.train_noise_scheduler,
                    ).to(device=self.cfg.device)
                elif self.cfg.model.use_img2img_refinement:
                    pipe = StableDiffusionPipeline.from_pretrained(
                        "CompVis/stable-diffusion-v1-4",
                        safety_checker=None,
                        torch_dtype=self.cfg.dtype,
                        unet=model.unet,
                        vae=model.vae.to(torch.float32),
                    ).to(device=self.cfg.device)

                    pipe_refinement = (
                        StableDiffusionImg2ImgPipeline.from_pretrained(
                            "CompVis/stable-diffusion-v1-4",
                            safety_checker=None,
                            torch_dtype=self.cfg.dtype,
                            unet=copy.deepcopy(model.unet),
                            vae=copy.deepcopy(model.vae.to(torch.float32)),
                            scheduler=copy.deepcopy(
                                model.train_noise_scheduler
                            ),
                        ).to(device=self.cfg.device)
                    )
                else:
                    pipe = StableDiffusionPipeline.from_pretrained(
                        "CompVis/stable-diffusion-v1-4",
                        safety_checker=None,
                        torch_dtype=self.cfg.dtype,
                        unet=model.unet,
                        vae=model.vae.to(torch.float32),
                        scheduler=model.train_noise_scheduler,
                    ).to(device=self.cfg.device)

            if self.cfg.model.use_ip_adapter:
                # pipe.load_ip_adapter(
                #     "h94/IP-Adapter",
                #     subfolder="models",
                #     weight_name="ip-adapter_sd15.bin",
                # )
                pipe.load_ip_adapter(
                    "h94/IP-Adapter",
                    subfolder="models",
                    weight_name="ip-adapter-plus_sd15.bin",
                )

                if self.cfg.model.ip_adapter_scale:
                    pipe.set_ip_adapter_scale(self.cfg.model.ip_adapter_scale)

        if use_custom_inference:
            # PNDMScheduler, DDIMScheduler
            scheduler = PNDMScheduler.from_pretrained(
                "lambdalabs/sd-image-variations-diffusers",
                subfolder="scheduler",
            )
        initial_preds = []
        images = []
        gt_images = []
        seg_maps = []
        for batch in tqdm(val_dataloader):
            with torch.no_grad():
                # with accelerator.autocast():
                for k, v in batch.items():
                    if k == "pixel_values_clip":
                        # v is a list of tensors
                        batch[k] = [vi.to(device=self.cfg.device) for vi in v]
                        if torch.is_floating_point(
                            v[0]
                        ) and self.cfg.device != torch.device("cpu"):
                            batch[k] = [
                                vi.to(dtype=self.cfg.dtype) for vi in batch[k]
                            ]
                    else:
                        batch[k] = v.to(device=self.cfg.device)
                        if torch.is_floating_point(
                            v
                        ) and self.cfg.device != torch.device("cpu"):
                            batch[k] = batch[k].to(dtype=self.cfg.dtype)

                if use_custom_inference:
                    with self.accelerator.autocast():
                        seg_maps.extend(
                            tensor_to_pil(batch["segmentation_mask"])
                        )
                        gt_images.extend(tensor_to_pil(batch["pixel_values"]))
                        images.extend(
                            model.validation_step(
                                batch,
                                noise_scheduler=scheduler,
                                num_inference_steps=200,
                                guidance_scale=guidance_scale,
                                device=self.cfg.device,
                                dtype=self.cfg.dtype,
                                generator=generator,
                                no_progress_bar=no_progress_bar,
                            )
                        )
                else:
                    encoder_hidden_states = [
                        model.image_encoder(
                            previous_frames_i
                        ).image_embeds.unsqueeze(
                            1
                        )  # --> tensor[Ni x 1 x F]
                        for previous_frames_i in batch["pixel_values_clip"]
                    ]

                    if self.cfg.model.use_lstm:
                        lstm_outputs = []
                        lstm_hn_init = torch.randn(
                            (model.lstm.get_bidirectional + 1)
                            * model.lstm.get_num_layers,
                            encoder_hidden_states[0].shape[1],
                            model.lstm.get_output_size,
                            device=model.lstm.device,
                            dtype=model.lstm.dtype,
                        )
                        lstm_cn_init = torch.randn(
                            (model.lstm.get_bidirectional + 1)
                            * model.lstm.get_num_layers,
                            encoder_hidden_states[0].shape[1],
                            model.lstm.get_hidden_size,
                            device=model.lstm.device,
                            dtype=model.lstm.dtype,
                        )
                        for (
                            sequence_of_previous_frames
                        ) in encoder_hidden_states:
                            output, _ = model.lstm(
                                sequence_of_previous_frames.to(
                                    dtype=model.lstm.dtype
                                ),
                                (
                                    lstm_hn_init.to(dtype=model.lstm.dtype),
                                    lstm_cn_init.to(dtype=model.lstm.dtype),
                                ),
                            )
                            lstm_outputs.append(
                                (
                                    output[-1][: output[-1].shape[0] / 2]
                                    + output[0][output[-1].shape[0] / 2 :]
                                )
                                / 2
                                if model.lstm.get_bidirectional
                                else output[-1]
                            )
                        encoder_hidden_states = torch.stack(lstm_outputs)
                    else:
                        encoder_hidden_states = torch.stack(
                            [
                                i.mean(dim=0) for i in encoder_hidden_states
                            ]  # tensor[1 x F]
                        )  # tensor[batch_size x 1 x F]

                    if self.cfg.model.use_mapper:
                        encoder_hidden_states = model.mapper(
                            encoder_hidden_states.to(dtype=model.mapper.dtype)
                        ).to(dtype=model.image_encoder.dtype)

                    seg_maps.extend(tensor_to_pil(batch["segmentation_mask"]))
                    gt_images.extend(tensor_to_pil(batch["pixel_values"]))
                    pipe.set_progress_bar_config(disable=True)
                    if self.cfg.model.use_img2img_refinement:
                        pipe_refinement.set_progress_bar_config(disable=True)
                    with self.accelerator.autocast():
                        for (
                            encoder_hidden_state,
                            segmentation_mask,
                            gt_image,
                            prev_img,
                        ) in zip(
                            encoder_hidden_states,
                            batch["segmentation_mask"],
                            batch["pixel_values"],
                            batch["pixel_values_prev"],
                        ):
                            if self.cfg.model.use_img2img_inference:
                                images.extend(
                                    pipe(
                                        prompt_embeds=encoder_hidden_state.unsqueeze(
                                            0
                                        ),
                                        negative_prompt_embeds=torch.zeros_like(
                                            encoder_hidden_state
                                        ).unsqueeze(0),
                                        image=tensor_to_pil(prev_img),
                                        control_image=tensor_to_pil(
                                            segmentation_mask
                                            if model.controlnet.controlnet
                                            else None
                                        ),
                                        ip_adapter_image_embeds=(
                                            encoder_hidden_state.unsqueeze(0)
                                            if self.cfg.model.use_ip_adapter
                                            else None
                                        ),
                                        num_inference_steps=300,
                                        guidance_scale=guidance_scale,
                                        generator=generator,
                                        strength=0.8,
                                    ).images
                                )
                            elif self.cfg.model.use_img2img_refinement:
                                initial_preds = pipe(
                                    prompt_embeds=encoder_hidden_state.unsqueeze(
                                        0
                                    ),
                                    negative_prompt_embeds=torch.zeros_like(
                                        encoder_hidden_state
                                    ).unsqueeze(0),
                                    image=tensor_to_pil(
                                        segmentation_mask
                                        if model.controlnet.controlnet
                                        else None
                                    ),
                                    ip_adapter_image=(
                                        # TODO: this is wrong, we must use the
                                        # previous image here or some average
                                        # embeding of the previous images directly
                                        # into `ip_adapter_image_embeds``
                                        tensor_to_pil(gt_image)
                                        if self.cfg.model.use_ip_adapter
                                        else None
                                    ),
                                    num_inference_steps=250,
                                    guidance_scale=guidance_scale,
                                    generator=generator,
                                ).images

                                images.extend(
                                    pipe_refinement(
                                        prompt_embeds=encoder_hidden_state.unsqueeze(
                                            0
                                        ),
                                        negative_prompt_embeds=torch.zeros_like(
                                            encoder_hidden_state
                                        ).unsqueeze(0),
                                        image=initial_preds,
                                        control_image=tensor_to_pil(
                                            segmentation_mask
                                            if model.controlnet.controlnet
                                            else None
                                        ),
                                        num_inference_steps=250,
                                        guidance_scale=guidance_scale,
                                        generator=generator,
                                        strength=0.8,
                                    ).images
                                )
                            else:
                                images.extend(
                                    pipe(
                                        prompt_embeds=encoder_hidden_state.unsqueeze(
                                            0
                                        ),
                                        negative_prompt_embeds=torch.zeros_like(
                                            encoder_hidden_state
                                        ).unsqueeze(0),
                                        image=tensor_to_pil(
                                            segmentation_mask
                                            if model.controlnet.controlnet
                                            else None
                                        ),
                                        ip_adapter_image=(
                                            # TODO: this is wrong, we must use the
                                            # previous image here or some average
                                            # embeding of the previous images directly
                                            # into `ip_adapter_image_embeds``
                                            tensor_to_pil(gt_image)
                                            if self.cfg.model.use_ip_adapter
                                            else None
                                        ),
                                        num_inference_steps=250,
                                        guidance_scale=guidance_scale,
                                        generator=generator,
                                    ).images
                                )
        grid = image_grid(
            [val for tup in zip(images, gt_images, seg_maps) for val in tup],
            len(images),
            3,
        )  # .resize((3 * 256, len(images) * 256))
        grid.save(
            self.cfg.predictions_dir
            / (
                (
                    f"epoch_{self.epoch}_step_{self.global_step}"
                    if output_name is None
                    else output_name
                )
                + ".png"
            )
        )

        if self.cfg.model.use_ip_adapter and not use_custom_inference:
            pipe.unload_ip_adapter()

        model.vae.to(self.cfg.dtype)

        return grid
