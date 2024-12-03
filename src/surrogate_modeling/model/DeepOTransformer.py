import math

import torch
from einops import rearrange
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


class DeepOTransformer(nn.Module):
    def __init__(
        self,
        num_control: int,
        num_pred: int,
        num_all_features: int,
        out_len: int,
        heads: int,
        fc_dim: int,
        trunk_depth: int,
        trunk_dim: int,
        branch_depth: int,
        branch_dim: int,
        width: int,
        dropout: float,
        emb_dropout: float,
    ):
        super().__init__()

        self.num_pred = num_pred
        self.out_len = out_len

        # Trunk Transformer and Linear layers
        self.x_past_embedding = nn.Linear(num_all_features, trunk_dim)
        self.positional_encoding = PositionalEncoding(trunk_dim)

        self.emb_dropout = nn.Dropout(emb_dropout)

        encoder_layers = nn.TransformerEncoderLayer(
            d_model=trunk_dim,
            nhead=heads,
            dim_feedforward=fc_dim,
            dropout=dropout,
            batch_first=True,
        )
        encoder_norm = nn.LayerNorm(trunk_dim)
        self.trunk_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layers,
            num_layers=trunk_depth,
            norm=encoder_norm,
        )

        self.trunk_linear = nn.Sequential(nn.Linear(trunk_dim, num_pred * width), nn.ReLU(inplace=True))

        # Branch network
        self.branch_linear = self._build_branch(num_control, num_pred, branch_depth, branch_dim, width)

        # Bias parameter
        self.bias = nn.Parameter(torch.zeros([num_pred]))

    def _build_branch(self, num_control, num_pred, branch_depth, branch_dim, width):
        """Helper function to build the branch network."""
        if branch_depth == 1:
            return nn.Linear(num_control, num_pred * width)

        layers = [nn.Sequential(nn.Linear(num_control, branch_dim), nn.ReLU(inplace=True))]
        for _ in range(branch_depth - 2):
            layers.append(nn.Sequential(nn.Linear(branch_dim, branch_dim), nn.ReLU(inplace=True)))
        layers.append(nn.Linear(branch_dim, num_pred * width))

        return nn.ModuleList(layers)

    def forward(self, x_past, x_control):
        # Trunk processing
        x_past = self.x_past_embedding(x_past)
        x_past += self.positional_encoding(x_past)

        x_past = self.emb_dropout(x_past)

        x_past = self.trunk_encoder(x_past)
        x_past = self.trunk_linear(x_past)

        # Branch processing
        for layer in self.branch_linear:
            x_control = layer(x_control)

        x_past = rearrange(x_past, "B L (N D) -> B L N D", N=self.num_pred)
        x_control = rearrange(x_control, "B L (N D) -> B L N D", N=self.num_pred)

        # Combine trunk and branch outputs
        y = torch.sum(x_past[:, -self.out_len :, :] * x_control, dim=-1, keepdim=False) + self.bias

        return y
