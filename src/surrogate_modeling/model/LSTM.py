import torch
from torch import nn


class LSTM(nn.Module):
    def __init__(
        self,
        num_control: int,
        num_pred: int,
        num_all_features: int,
        out_len: int,
        dim: int,
        depth: int,
        dropout: float,
    ):
        super(LSTM, self).__init__()

        self.out_len = out_len

        # LSTM Layer
        self.lstm_layer = nn.LSTM(
            input_size=num_all_features,
            hidden_size=dim,
            num_layers=depth,
            dropout=dropout,
            batch_first=True,
        )

        # Linear transformation for the control input
        self.control_linear = nn.Linear(num_control, dim)

        # Fully connected layer with activation
        self.fc_layer = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(inplace=True),
        )

        # Output generator
        self.output_generator = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_pred))

    def forward(self, x_past, x_control, h=None):
        # LSTM forward pass
        lstm_out, _ = self.lstm_layer(x_past, h)

        # Linear transformation on control input
        control_out = self.control_linear(x_control)

        # Concatenating LSTM output and control input features
        combined_out = self.fc_layer(torch.cat([lstm_out, control_out], dim=1))

        # Generating the final output
        y = self.output_generator(combined_out)

        # Returning only the required time steps
        return y[:, -(self.out_len * 2) : -self.out_len, :]
