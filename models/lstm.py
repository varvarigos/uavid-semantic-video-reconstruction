"""LSTM for Frame Aggregation."""

import torch
from diffusers.training_utils import cast_training_params
from torch.nn import LSTM


class LSTMModel(torch.nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        bias: bool = True,
        batch_first: bool = False,
        dropout: float = 0.0,
        bidirectional: bool = False,
        proj_size: int = 0,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        train_lstm: bool = False,
    ):
        super().__init__()

        self.lstm = LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=bias,
            batch_first=batch_first,
            dropout=dropout,
            bidirectional=bidirectional,
            proj_size=proj_size,
        )

        self.lstm.input_size = input_size
        self.lstm.hidden_size = hidden_size
        self.lstm.bidirectional = bidirectional
        self.lstm.num_layers = num_layers
        self.lstm.output_size = proj_size if proj_size > 0 else hidden_size

        self.lstm.requires_grad_(train_lstm)
        self.lstm.to(device=device, dtype=dtype)
        self.lstm.train(train_lstm)

        self.trainable_parameters = []
        if train_lstm:
            if dtype == torch.float16:
                cast_training_params(self.lstm, dtype=torch.float32)

            self.trainable_parameters = [
                p for p in self.lstm.parameters() if p.requires_grad
            ]

    @property
    def get_input_size(self):
        return self.lstm.input_size

    @property
    def get_hidden_size(self):
        return self.lstm.hidden_size

    @property
    def get_bidirectional(self):
        return self.lstm.bidirectional

    @property
    def get_num_layers(self):
        return self.lstm.num_layers

    @property
    def get_output_size(self):
        return self.lstm.output_size

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def __call__(self, *args, **kwargs):
        return self.lstm(*args, **kwargs)

    def forward(self, *args, **kwargs):
        return self.lstm(*args, **kwargs)

    def train(self, mode=True):
        self.lstm.train(mode)

    def requires_grad_(self, requires_grad=True):
        self.lstm.requires_grad_(requires_grad)
