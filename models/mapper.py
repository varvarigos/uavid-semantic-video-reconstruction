"""Simple Embeddings Mapper MLP."""

import torch
from diffusers.training_utils import cast_training_params
from torchvision.ops import MLP


class Mapper(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: list,
        norm_layer: torch.nn.Module | None = torch.nn.LayerNorm,
        activation_layer: str = "relu",
        inplace: bool | None = None,
        bias: bool = True,
        dropout: float = 0.0,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        train_mapper: bool = False,
    ):
        super().__init__()

        self.mapper = MLP(
            in_channels=in_channels,  # Number of input features
            hidden_channels=hidden_channels,  # List specifying the size of each hidden layer
            norm_layer=norm_layer,  # No normalization layer
            activation_layer=activation_layer,  # Using ReLU as the activation function
            inplace=inplace,  # In-place operation for activation
            bias=bias,  # Using bias in linear layers
            dropout=dropout,  # No dropout
        )

        self.mapper.requires_grad_(train_mapper)
        self.mapper.to(device=device, dtype=dtype)
        self.mapper.train(train_mapper)

        self.trainable_parameters = []
        if train_mapper:
            if dtype == torch.float16:
                cast_training_params(self.mapper, dtype=torch.float32)

            self.trainable_parameters = [
                p for p in self.mapper.parameters() if p.requires_grad
            ]

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def __call__(self, *args, **kwargs):
        return self.mapper(*args, **kwargs)

    def forward(self, *args, **kwargs):
        return self.mapper(*args, **kwargs)

    def train(self, mode=True):
        self.mapper.train(mode)

    def requires_grad_(self, requires_grad=True):
        self.mapper.requires_grad_(requires_grad)
