import torch
from diffusers.utils.torch_utils import randn_tensor
from torch.functional import F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import CLIPVisionModelWithProjection

from utils import tensor_to_pil

from .controlnet import ControlNet
from .lstm import LSTMModel
from .mapper import Mapper
from .stable_diffusion_1x import StableDiffusion1x


class StableDiffusion1xImageVariation(StableDiffusion1x):
    def __init__(
        self,
        model_name: str,
        model_revision: str | None = None,
        image_encoder_name: str | None = None,
        image_encoder_revision: str | None = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        controlnet: ControlNet | None = None,
        train_lora_adapter: bool = False,
        lora_rank: int = 4,
        mapper: Mapper | None = None,
        lstm: LSTMModel | None = None,
    ):
        super().__init__(
            model_name,
            model_revision,
            device,
            dtype,
            controlnet,
            train_lora_adapter,
            lora_rank,
            mapper,
            lstm,
        )

        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            model_name if image_encoder_name is None else image_encoder_name,
            revision=(
                model_revision
                if image_encoder_revision is None
                else image_encoder_revision
            ),
            subfolder="" if "clip" in image_encoder_name else "image_encoder",
        )

        self.image_encoder.requires_grad_(False)
        self.image_encoder.to(device=device, dtype=dtype)
        self.image_encoder.eval()

    def training_step(self, batch, batch_idx):
        # Convert images to latent space
        model_input = self.vae.encode(
            batch["pixel_values"]
        ).latent_dist.sample()
        model_input = model_input * self.vae.config.scaling_factor

        # Sample noise that we'll add to the latents
        noise = torch.randn_like(model_input)
        bsz, _, _, _ = model_input.shape
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0,
            self.train_noise_scheduler.config.num_train_timesteps,
            (bsz,),
            device=model_input.device,
        )
        timesteps = timesteps.long()

        # Add noise to the model input according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_model_input = self.train_noise_scheduler.add_noise(
            model_input, noise, timesteps
        )

        encoder_hidden_states = [
            self.image_encoder(previous_frames_i).image_embeds.unsqueeze(
                1
            )  # --> tensor[Ni x 1 x F]
            for previous_frames_i in batch["pixel_values_clip"]
        ]

        if self.lstm:
            lstm_outputs = []
            lstm_hn_init = torch.randn(
                (self.lstm.get_bidirectional + 1) * self.lstm.get_num_layers,
                encoder_hidden_states[0].shape[1],
                self.lstm.get_output_size,
                device=self.lstm.device,
                dtype=self.lstm.dtype,
            )
            lstm_cn_init = torch.randn(
                (self.lstm.get_bidirectional + 1) * self.lstm.get_num_layers,
                encoder_hidden_states[0].shape[1],
                self.lstm.get_hidden_size,
                device=self.lstm.device,
                dtype=self.lstm.dtype,
            )
            for sequence_of_previous_frames in encoder_hidden_states:
                output, _ = self.lstm(
                    sequence_of_previous_frames.to(dtype=self.lstm.dtype),
                    (
                        lstm_hn_init.to(dtype=self.lstm.dtype),
                        lstm_cn_init.to(dtype=self.lstm.dtype),
                    ),
                )
                lstm_outputs.append(
                    (
                        output[-1][: output[-1].shape[0] / 2]
                        + output[0][output[-1].shape[0] / 2 :]
                    )
                    / 2
                    if self.lstm.get_bidirectional
                    else output[-1]
                )
                # lstm_outputs.append(output.mean(dim=0))
            encoder_hidden_states = torch.stack(lstm_outputs)
        else:
            encoder_hidden_states = torch.stack(
                [i.mean(dim=0) for i in encoder_hidden_states]  # tensor[1 x F]
            )  # tensor[batch_size x 1 x F]

        if self.mapper:
            encoder_hidden_states = self.mapper(
                encoder_hidden_states.to(dtype=self.mapper.dtype)
            )

        current_segmentation_map = batch["segmentation_mask"]

        if self.controlnet:
            down_block_res_samples, mid_block_res_sample = self.controlnet(
                noisy_model_input,
                timesteps,
                encoder_hidden_states=encoder_hidden_states.to(
                    self.controlnet.dtype
                ),
                controlnet_cond=current_segmentation_map,
                return_dict=False,
            )
            noise_pred = self.unet(
                noisy_model_input.to(dtype=torch.float16),
                timesteps,
                encoder_hidden_states=encoder_hidden_states.to(
                    dtype=torch.float16
                ),
                down_block_additional_residuals=[
                    sample.to(dtype=torch.float16)
                    for sample in down_block_res_samples
                ],
                mid_block_additional_residual=mid_block_res_sample.to(
                    dtype=torch.float16
                ),
            ).sample
        else:
            noise_pred = self.unet(
                noisy_model_input,
                timesteps,
                encoder_hidden_states=encoder_hidden_states,
            ).sample

        return F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")

    @torch.no_grad()
    def validation_step(
        self,
        batch,
        noise_scheduler,
        num_inference_steps=50,
        guidance_scale=5,
        device=torch.cuda,
        dtype=torch.float16,
        generator=None,
        no_progress_bar=True,
    ):
        batch_size = batch["pixel_values"].shape[0]

        # 5. Prepare timesteps
        noise_scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = noise_scheduler.timesteps

        # 6. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels

        shape = (
            batch_size,
            num_channels_latents,
            64,
            64,
        )
        latents = torch.randn(
            shape, generator=generator, device="cpu", dtype=dtype
        ).to(device)
        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * noise_scheduler.init_noise_sigma

        do_classifier_free_guidance = True
        for i, t in enumerate(tqdm(timesteps, disable=no_progress_bar)):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = (
                torch.cat([latents] * 2)
                if do_classifier_free_guidance
                else latents
            )
            latent_model_input = noise_scheduler.scale_model_input(
                latent_model_input, t
            )

            encoder_hidden_states = [
                self.image_encoder(
                    previous_frames_i.to(dtype=torch.float16, device="cuda")
                ).image_embeds.unsqueeze(
                    1
                )  # --> tensor[Ni x 1 x F]
                for previous_frames_i in batch["pixel_values_clip"]
            ]

            if self.lstm:
                lstm_outputs = []
                lstm_hn = []
                lstm_cn = []
                lstm_hn.append(
                    torch.zeros(
                        (self.lstm.get_bidirectional + 1)
                        * self.lstm.get_num_layers,
                        encoder_hidden_states[0].shape[1],
                        self.lstm.get_output_size,
                        device=self.lstm.device,
                        dtype=self.lstm.dtype,
                    )
                )
                lstm_cn.append(
                    torch.zeros(
                        (self.lstm.get_bidirectional + 1)
                        * self.lstm.get_num_layers,
                        encoder_hidden_states[0].shape[1],
                        self.lstm.get_hidden_size,
                        device=self.lstm.device,
                        dtype=self.lstm.dtype,
                    )
                )
                for sequence_of_previous_frames in encoder_hidden_states:
                    output, (hn, cn) = self.lstm(
                        sequence_of_previous_frames.to(dtype=self.lstm.dtype),
                        (
                            lstm_cn[-1].to(dtype=self.lstm.dtype),
                            lstm_cn[-1].to(dtype=self.lstm.dtype),
                        ),
                    )
                    lstm_hn.append(hn)
                    lstm_cn.append(cn)
                    lstm_outputs.append(output[-1])
                encoder_hidden_states = torch.stack(lstm_outputs)
            else:
                encoder_hidden_states = torch.stack(
                    [
                        i.mean(dim=0) for i in encoder_hidden_states
                    ]  # tensor[1 x F]
                )  # tensor[batch_size x 1 x F]

            if self.mapper:
                encoder_hidden_states = self.mapper(
                    encoder_hidden_states.to(dtype=self.mapper.dtype)
                )

            current_segmentation_map = batch["segmentation_mask"]

            negative_prompt_embeds = torch.zeros_like(encoder_hidden_states).to(
                dtype=torch.float16, device="cuda"
            )
            encoder_hidden_states = torch.cat(
                [negative_prompt_embeds, encoder_hidden_states]
            )

            if self.controlnet:
                down_block_res_samples, mid_block_res_sample = self.controlnet(
                    latent_model_input.to(
                        dtype=self.controlnet.dtype, device="cuda"
                    ),
                    t,
                    encoder_hidden_states=encoder_hidden_states.to(
                        dtype=self.controlnet.dtype, device="cuda"
                    ),
                    controlnet_cond=torch.cat(
                        [
                            current_segmentation_map.to(
                                dtype=self.controlnet.dtype, device="cuda"
                            )
                        ]
                        * 2
                    ),
                    return_dict=False,
                )

                noise_pred = self.unet(
                    latent_model_input.to(dtype=torch.float16, device="cuda"),
                    t,
                    encoder_hidden_states=encoder_hidden_states.to(
                        dtype=torch.float16, device="cuda"
                    ),
                    down_block_additional_residuals=[
                        sample.to(dtype=torch.float16, device="cuda")
                        for sample in down_block_res_samples
                    ],
                    mid_block_additional_residual=mid_block_res_sample.to(
                        dtype=torch.float16, device="cuda"
                    ),
                ).sample
            else:
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=encoder_hidden_states,
                ).sample

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )

            # compute the previous noisy sample x_t -> x_t-1
            latents = noise_scheduler.step(
                noise_pred,
                t,
                latents,
                return_dict=False,
            )[0]

        images = self.vae.decode(
            latents / self.vae.config.scaling_factor,
            generator=generator,
        ).sample

        return [tensor_to_pil(image) for image in images]
