import torch
import torch.nn as nn


class NetworkStateLSTM(nn.Module):
    def __init__(
        self,
        input_size=32,
        hidden_size=64,
        num_layers=2,
        output_size=32
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.fc = nn.Linear(
            hidden_size,
            output_size
        )

    def forward(self, x):
        output, _ = self.lstm(x)

        last_output = output[:, -1, :]

        prediction = self.fc(last_output)

        return prediction


class MultitaskNetworkStateLSTM(nn.Module):
    """Shared temporal encoder with delta-prediction and attack heads."""

    def __init__(
        self,
        input_size=32,
        hidden_size=64,
        num_layers=2,
        output_size=32,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.delta_head = nn.Linear(hidden_size, output_size)
        self.attack_head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        temporal_output, _ = self.lstm(x)
        shared_representation = temporal_output[:, -1, :]
        predicted_delta = self.delta_head(shared_representation)
        attack_logit = self.attack_head(shared_representation).squeeze(-1)
        return predicted_delta, attack_logit
