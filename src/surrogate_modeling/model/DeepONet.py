import torch
from einops import rearrange
from torch import nn


class DeepONet(nn.Module):
    def __init__(
        self,
        num_control: int,
        num_pred: int,
        num_all_features: int,
        out_len: int,
        trunk_depth: int,
        trunk_dim: int,
        branch_depth: int,
        branch_dim: int,
        width: int,
    ):
        super(DeepONet, self).__init__()

        self.num_pred = num_pred
        self.out_len = out_len

        # Trunk network: Replacing LSTM with Linear layers
        self.trunk_linear = self._build_trunk(num_all_features, trunk_depth, trunk_dim, num_pred, width)

        # Branch network
        self.branch_linear = self._build_branch(num_control, num_pred, branch_depth, branch_dim, width)

        # Bias parameter
        self.bias = nn.Parameter(torch.zeros([num_pred]))

    def _build_trunk(self, input_size, trunk_depth, trunk_dim, num_pred, width):
        """Helper function to build the trunk network."""
        if trunk_depth == 1:
            return nn.Sequential(nn.Linear(input_size, num_pred * width), nn.ReLU(inplace=True))

        layers = [nn.Sequential(nn.Linear(input_size, trunk_dim), nn.ReLU(inplace=True))]
        for _ in range(trunk_depth - 2):
            layers.append(nn.Sequential(nn.Linear(trunk_dim, trunk_dim), nn.ReLU(inplace=True)))
        layers.append(nn.Sequential(nn.Linear(trunk_dim, num_pred * width), nn.ReLU(inplace=True)))

        return nn.ModuleList(layers)

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
        for layer in self.trunk_linear:
            x_past = layer(x_past)

        # Branch processing
        for layer in self.branch_linear:
            x_control = layer(x_control)

        x_past = rearrange(x_past, "B L (N D) -> B L N D", N=self.num_pred)
        x_control = rearrange(x_control, "B L (N D) -> B L N D", N=self.num_pred)

        # Combine trunk and branch outputs
        y = torch.sum(x_past[:, -self.out_len :, :] * x_control, dim=-1, keepdim=False) + self.bias

        return y
