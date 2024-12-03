import math

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.requires_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return self.pe[:, : x.size(1)]


class BaseTransformer(nn.Module):
    def __init__(
        self,
        num_control: int,
        num_pred: int,
        num_all_features: int,
        out_len: int,
        dim: int,
        depth: int,
        heads: int,
        fc_dim: int,
        dropout: float,
        emb_dropout: float,
    ):

        super().__init__()

        self.out_len = out_len

        self.x_past_embedding = nn.Linear(num_all_features, dim)
        self.x_control_embedding = nn.Linear(num_control, dim)

        self.positional_encoding = PositionalEncoding(dim)

        self.emb_dropout = nn.Dropout(emb_dropout)
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=fc_dim,
            dropout=dropout,
            batch_first=True,
        )
        encoder_norm = nn.LayerNorm(dim)
        self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layers, num_layers=depth, norm=encoder_norm)

        self.generator = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_pred))

    def forward(self, x_past, x_control):
        x_past = self.x_past_embedding(x_past)
        x_control = self.x_control_embedding(x_control)
        x_combined = torch.cat((x_past, x_control), dim=1)
        x_combined += self.positional_encoding(x_combined)

        x_combined = self.emb_dropout(x_combined)
        x_combined = self.encoder(x_combined)

        y = self.generator(x_combined)

        return y[:, -(self.out_len * 2) : -self.out_len, :]
