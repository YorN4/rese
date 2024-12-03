import torch
from einops import rearrange
from torch import nn


class DeepOLSTM(nn.Module):
    def __init__(
        self,
        num_control: int,
        num_pred: int,
        num_all_features: int,
        out_len: int,
        dropout: float,
        trunk_depth: int,
        trunk_dim: int,
        branch_depth: int,
        branch_dim: int,
        width: int,
    ):
        super(DeepOLSTM, self).__init__()

        self.num_pred = num_pred
        self.out_len = out_len

        # Trunk LSTM and Linear layers
        self.trunk_lstm = nn.LSTM(
            input_size=num_all_features,
            hidden_size=trunk_dim,
            num_layers=trunk_depth,
            dropout=dropout,
            batch_first=True,
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

    def forward(self, x_past, x_control, h=None):
        # Trunk processing
        x_past, _ = self.trunk_lstm(x_past, h)
        x_past = self.trunk_linear(x_past)

        # Branch processing
        for layer in self.branch_linear:
            x_control = layer(x_control)

        x_past = rearrange(x_past, "B L (N D) -> B L N D", N=self.num_pred)
        x_control = rearrange(x_control, "B L (N D) -> B L N D", N=self.num_pred)

        # Combine trunk and branch outputs
        y = torch.sum(x_past[:, -self.out_len :, :] * x_control, dim=-1, keepdim=False) + self.bias

        return y
