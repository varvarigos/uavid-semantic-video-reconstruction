import torch
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.training_utils import cast_training_params
from peft import LoraConfig
from torch import nn

from .controlnet import ControlNet
from .lstm import LSTMModel
from .mapper import Mapper


class StableDiffusion1x(nn.Module):
    def __init__(
        self,
        model_name: str,
        model_revision: str | None = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        controlnet: ControlNet | None = None,
        train_lora_adapter: bool = False,
        lora_rank: int = 4,
        mapper: Mapper | None = None,
        lstm: LSTMModel | None = None,
    ):
        super().__init__()

        self.train_noise_scheduler = DDPMScheduler.from_pretrained(
            model_name, subfolder="scheduler"
        )
        # TODO: add text encoder and tokenizer

        self.vae = AutoencoderKL.from_pretrained(
            model_name,
            revision=model_revision,
            subfolder="vae",
        )
        self.unet = UNet2DConditionModel.from_pretrained(
            model_name,
            revision=model_revision,
            subfolder="unet",
        )
        self.controlnet = controlnet
        self.mapper = mapper
        self.lstm = lstm

        # We only train the additional adapter LoRA layers
        self.vae.requires_grad_(False)
        self.unet.requires_grad_(False)

        # Move unet, vae, image_encoder, and contronet to device and cast to weight_dtype
        self.unet.to(device=device, dtype=dtype)
        self.vae.to(device=device, dtype=dtype)

        self.vae.eval()
        self.unet.eval()

        self.unet_trainable_parameters = []
        if train_lora_adapter:
            unet_lora_config = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_rank,
                init_lora_weights="gaussian",
                target_modules=["to_k", "to_q", "to_v", "to_out.0"],
            )

            # Add adapter and make sure the trainable params are in float32.
            self.unet.add_adapter(unet_lora_config)
            if dtype == torch.float16:
                # only upcast trainable parameters (LoRA) into fp32
                cast_training_params(self.unet, dtype=torch.float32)

            self.unet_trainable_parameters = [
                p for p in self.unet.parameters() if p.requires_grad
            ]

    @property
    def trainable_parameters(self):
        return (
            self.unet_trainable_parameters
            + (self.controlnet.trainable_parameters if self.controlnet else [])
            + (self.mapper.trainable_parameters if self.mapper else [])
            + (self.lstm.trainable_parameters if self.lstm else [])
        )

    def train(self, mode=True):
        self.unet.train(mode)
        if self.controlnet and self.controlnet.trainable_parameters:
            self.controlnet.train(mode)
        if self.mapper and self.mapper.trainable_parameters:
            self.mapper.train(mode)
        if self.lstm and self.lstm.trainable_parameters:
            self.lstm.train(mode)
        self.vae.train(False)

    def forward(self, x):
        return x

    def training_step(self, batch, batch_idx):
        pass
