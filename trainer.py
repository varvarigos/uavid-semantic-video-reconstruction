from dataclasses import asdict

import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from diffusers.pipelines import StableDiffusionControlNetPipeline
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import TrainerConfig
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
        #     "/teamspace/studios/this_studio/outputs/single_frame/controlnet_only/2024-05-15__13-30-02/checkpoints/checkpoint_2"
        # )

        self.cfg.post_accelerator_init(self.accelerator)
        self.logger = get_logger(__name__)

        self.global_step = 0
        self.epoch = 0

        self.accelerator.init_trackers(
            "uavid_frame_prediction",
            config={k: str(v) for k, v in asdict(self.cfg).items()},
        )

    def fit(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloder: DataLoader,
        optimizer,
        lr_scheduler,
    ):
        # Prepare everything with our `accelerator`.
        if self.cfg.model.train_control_net and not self.cfg.model.train_unet:
            (
                model.controlnet.controlnet,
                optimizer,
                train_dataloader,
                lr_scheduler,
            ) = self.accelerator.prepare(
                model.controlnet.controlnet,
                optimizer,
                train_dataloader,
                lr_scheduler,
            )
        elif self.cfg.model.train_unet and not self.cfg.model.train_control_net:
            model.unet, optimizer, train_dataloader, lr_scheduler = (
                self.accelerator.prepare(
                    model.unet, optimizer, train_dataloader, lr_scheduler
                )
            )
        else:
            (
                model.unet,
                model.controlnet.controlnet,
                optimizer,
                train_dataloader,
                lr_scheduler,
            ) = self.accelerator.prepare(
                model.unet,
                model.controlnet.controlnet,
                optimizer,
                train_dataloader,
                lr_scheduler,
            )

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
        controlnet_param_stats = get_parameters_stats(
            model.controlnet.parameters()
        )
        self.logger.info(
            f"  Trainable ControlNet LoRA Parameters: {controlnet_param_stats['trainable']}/{controlnet_param_stats['all']}"
        )

        self.global_step = 0
        first_epoch = 0

        # Potentially load in the weights and states from a previous save
        # TODO: for sure this needs debugging to work
        # if self.cfg.resume_from_checkpoint:
        #     if self.cfg.resume_from_checkpoint != "latest":
        #         # path = os.path.basename(resume_from_checkpoint)
        #         # do the same for Path type paths
        #         path = self.cfg.resume_from_checkpoint.name
        #     else:
        #         # Get the mos recent checkpoint
        #         # dirs = os.listdir(output_dir)
        #         # do the same for Path type paths
        #         dirs = output_dir.iterdir()

        #         dirs = [d for d in dirs if d.startswith("checkpoint")]
        #         dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
        #         path = dirs[-1] if len(dirs) > 0 else None

        #     if path is None:
        #         accelerator.print(
        #             f"Checkpoint '{self.cfg.resume_from_checkpoint}' does not exist. Starting a new training run."
        #         )
        #         self.cfg.resume_from_checkpoint = None
        #         initial_global_step = 0
        #     else:
        #         accelerator.print(f"Resuming from checkpoint {path}")
        #         accelerator.load_state("lora-dreambooth-model/checkpoint-2000")
        #         # accelerator.load_state(os.path.join(output_dir, path))
        #         # accelerator.load_state('/content/lora-dreambooth-model/checkpoint-2000')
        #         global_step = int(path.split("-")[1])

        #         initial_global_step = global_step
        #         first_epoch = global_step // num_update_steps_per_epoch
        # else:
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
                            self.validation(
                                model=model, val_dataloader=val_dataloder
                            )
                            model.train()
                logs = (
                    {"loss": loss.detach().item()}
                    | (
                        {"lr_unet": lr_scheduler.get_last_lr()[0]}
                        if self.cfg.model.train_unet
                        else {}
                    )
                    | (
                        {"lr_cntrl": lr_scheduler.get_last_lr()[-1]}
                        if self.cfg.model.train_control_net
                        else {}
                    )
                )
                progress_bar.set_postfix(**logs)
                self.accelerator.log(logs, step=self.global_step)

                if self.global_step >= self.cfg.max_train_steps:
                    break

            print(total_loss / len(train_dataloader.dataset))

        ## Finish Up Training

        # Save the lora layers
        model.eval()

        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            self.accelerator.save_state()
            # TODO: fix this
            # if train_unet:
            #     unet_temp = self.accelerator.unwrap_model(unet)
            #     unet_lora_layers = unet_lora_state_dict(unet_temp)

            #     LoraLoaderMixin.save_lora_weights(
            #         save_directory=output_unet_dir,
            #         unet_lora_layers=unet_lora_layers,
            #     )
            # if train_control_net:
            #     controlnet_temp = accelerator.unwrap_model(controlnet)
            #     controlnet_temp.save_pretrained(output_controlnet_dir)

        self.accelerator.end_training()

    def validation(self, model: nn.Module, val_dataloader: DataLoader):
        model.eval()
        generator = torch.manual_seed(42)

        if model.controlnet:
            # maybe those need to be unwrapped first, e.g.: self.accelerator.unwrap_model(model.controlnet)
            pipe = StableDiffusionControlNetPipeline.from_pretrained(
                "CompVis/stable-diffusion-v1-4",
                safety_checker=None,
                torch_dtype=self.cfg.dtype,
                controlnet=model.controlnet.controlnet,
                unet=model.unet,
                vae=model.vae.to(torch.float32),
            ).to(device=self.cfg.device)

        if self.cfg.model.use_ip_adapter:
            # pipe.load_ip_adapter(
            #     "h94/IP-Adapter-plus",
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

                encoder_hidden_states = [
                    model.image_encoder(
                        previous_frames_i
                    ).image_embeds.unsqueeze(
                        1
                    )  # --> tensor[Ni x 1 x F]
                    for previous_frames_i in batch["pixel_values_clip"]
                ]

                ## IF AVERAGE (not LSTM)
                encoder_hidden_states = torch.stack(
                    [i.mean(dim=0) for i in encoder_hidden_states]  # tensor[F]
                )  # tensor[batch_size x 1 x F]

                # encoder_hidden_states = model.image_encoder(
                #     batch["pixel_values_clip"]
                # ).image_embeds.unsqueeze(1)

                seg_maps.extend(tensor_to_pil(batch["segmentation_mask"]))
                gt_images.extend(tensor_to_pil(batch["pixel_values"]))
                pipe.set_progress_bar_config(disable=True)
                with self.accelerator.autocast():
                    for (
                        encoder_hidden_state,
                        segmentation_mask,
                        gt_image,
                    ) in zip(
                        encoder_hidden_states,
                        batch["segmentation_mask"],
                        batch["pixel_values"],
                    ):
                        images.extend(
                            pipe(
                                prompt_embeds=encoder_hidden_state.unsqueeze(0),
                                negative_prompt_embeds=torch.zeros_like(
                                    encoder_hidden_state
                                ).unsqueeze(0),
                                image=tensor_to_pil(segmentation_mask),
                                ip_adapter_image=(
                                    # TODO: this is wrong, we must use the
                                    # previous image here or some average
                                    # embeding of the previous images directly
                                    # into `ip_adapter_image_embeds``
                                    tensor_to_pil(gt_image)
                                    if self.cfg.model.use_ip_adapter
                                    else None
                                ),
                                num_inference_steps=100,
                                guidance_scale=5,
                                generator=generator,
                            ).images
                        )

        grid = image_grid(
            [val for tup in zip(images, gt_images, seg_maps) for val in tup],
            len(images),
            3,
        ).resize((3 * 256, len(images) * 256))
        grid.save(
            self.cfg.predictions_dir
            / f"epoch_{self.epoch}_step_{self.global_step}.png"
        )

        if self.cfg.model.use_ip_adapter:
            pipe.unload_ip_adapter()

        model.vae.to(self.cfg.dtype)
        return grid
