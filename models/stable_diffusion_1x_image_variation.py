import torch
from torch.functional import F
from transformers import CLIPVisionModelWithProjection

from .controlnet import ControlNet
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
    ):
        super().__init__(
            model_name,
            model_revision,
            device,
            dtype,
            controlnet,
            train_lora_adapter,
            lora_rank,
        )

        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            model_name if image_encoder_name is None else image_encoder_name,
            revision=(
                model_revision
                if image_encoder_revision is None
                else image_encoder_revision
            ),
            subfolder="image_encoder",
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

        # # Get the image embedding for conditioning
        # encoder_hidden_states = self.image_encoder(
        #     batch["pixel_values_clip"]
        # ).image_embeds.unsqueeze(1)

        # Get the image embedding for conditioning
        ### START --- FROM MANY PREVIOUS FRAMES TO A SINGLE EMBEDDING -- AGGREGATION
        # batch["pixel_values_clip"] --> list[tensors[Ni x WxHxC]]

        encoder_hidden_states = [
            self.image_encoder(previous_frames_i).image_embeds.unsqueeze(
                1
            )  # --> tensor[Ni x 1 x F]
            for previous_frames_i in batch["pixel_values_clip"]
        ]

        ## IF AVERAGE (not LSTM)
        encoder_hidden_states = torch.stack(
            [i.mean(dim=0) for i in encoder_hidden_states]  # tensor[1 x F]
        )  # tensor[batch_size x 1 x F]

        ### FINISH --- FROM MANY PREVIOUS FRAMES TO A SINGLE EMBEDDING -- AGGREGATION

        current_segmentation_map = batch["segmentation_mask"]

        if self.controlnet:
            down_block_res_samples, mid_block_res_sample = self.controlnet(
                noisy_model_input,
                timesteps,
                encoder_hidden_states=encoder_hidden_states,
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
