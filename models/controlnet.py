"""ControlNet model with optional LoRA adapter."""

import torch
from diffusers import ControlNetModel
from diffusers.loaders.peft import PeftAdapterMixin
from diffusers.training_utils import cast_training_params
from peft import LoraConfig


class ControlNet(torch.nn.Module):
    """ControlNet model with optional LoRA adapter."""

    def __init__(
        self,
        model_name: str,
        model_revision: str | None = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        train_lora_adapter: bool = False,
        lora_rank: int = 4,
    ):
        super().__init__()

        ControlNetModel.__bases__ += (PeftAdapterMixin,)
        self.controlnet = ControlNetModel.from_pretrained(
            model_name, model_revision=model_revision
        )

        self.controlnet.requires_grad_(False)
        self.controlnet.to(device=device, dtype=dtype)
        self.controlnet.eval()

        self.trainable_parameters = []
        if train_lora_adapter:
            # Add adapter and make sure the trainable params are in float32.
            self.controlnet.add_adapter(
                LoraConfig(
                    r=lora_rank,
                    lora_alpha=lora_rank,
                    init_lora_weights="gaussian",
                    target_modules=["to_k", "to_q", "to_v", "to_out.0"],
                )
            )
            if dtype == torch.float16:
                # only upcast trainable parameters (LoRA) into fp32
                cast_training_params(self.controlnet, dtype=torch.float32)

            self.trainable_parameters = [
                p for p in self.controlnet.parameters() if p.requires_grad
            ]

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def __call__(self, *args, **kwargs):
        return self.controlnet(*args, **kwargs)

    def forward(self, *args, **kwargs):
        return self.controlnet(*args, **kwargs)

    def train(self, mode=True):
        self.controlnet.train(mode)

    def requires_grad_(self, requires_grad=True):
        self.controlnet.requires_grad_(requires_grad)

    # def __setattr__(self, name, value):
    #     if name == "controlnet":
    #         raise AttributeError("You should probably not be here.")
    #         # super().__setattr__(name, value)
    #     elif name == "trainable_parameters":
    #         raise AttributeError("You should probably not be here2.")
    #         # super().__setattr__(name, value)
    #     else:
    #         setattr(self.controlnet, name, value)


# def wrapped__getattr__(self, name):
#     if name == "trainable_parameters":
#         return self.trainable_parameters
#     if name == "controlnet":
#         return self.controlnet
#     return getattr(self.controlnet, name)


# ControlNet.original__get_attr__ = ControlNet.__getattr__
# ControlNet.__getattr__ = wrapped__getattr__
