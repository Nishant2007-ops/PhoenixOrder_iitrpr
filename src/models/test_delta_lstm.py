
import torch
import torch.nn as nn
import numpy as np

from src.preprocessing.normalize_cse import find_cse_files
from src.temporal.build_experiment_dataset import build_experiment_states
from src.temporal.prepare_delta_lstm_data import prepare_delta_lstm_data
from src.models.lstm_model import NetworkStateLSTM


DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)


def main():

    print("Device:", DEVICE)

    files = find_cse_files()[:3]

    state_groups = build_experiment_states(
        files
    )

    (
        X_train,
        y_train,
        X_validation,
        y_validation,
        X_test,
        y_test,
        feature_columns
    ) = prepare_delta_lstm_data(
        state_groups
    )

    X_test = torch.tensor(
        X_test,
        dtype=torch.float32
    ).to(DEVICE)

    y_test = torch.tensor(
        y_test,
        dtype=torch.float32
    ).to(DEVICE)

    model = NetworkStateLSTM(
        input_size=32,
        hidden_size=64,
        num_layers=2,
        output_size=32
    ).to(DEVICE)

    model.load_state_dict(
        torch.load(
            "delta_lstm_model.pth",
            map_location=DEVICE
        )
    )

    model.eval()

    with torch.no_grad():

        predicted_delta = model(
            X_test
        )

    last_state = X_test[:, -1, :]

    predicted_state = (
        last_state + predicted_delta
    )

    actual_next_state = (
        last_state + y_test
    )

    persistence_state = last_state

    mse = nn.MSELoss()
    mae = nn.L1Loss()

    delta_lstm_mse = mse(
        predicted_state,
        actual_next_state
    ).item()

    delta_lstm_mae = mae(
        predicted_state,
        actual_next_state
    ).item()

    persistence_mse = mse(
        persistence_state,
        actual_next_state
    ).item()

    persistence_mae = mae(
        persistence_state,
        actual_next_state
    ).item()

    improvement = (
        (persistence_mse - delta_lstm_mse)
        / persistence_mse
    ) * 100

    print("\n==============================")
    print("DELTA-LSTM TEST RESULTS")
    print("==============================")

    print(
        f"Delta-LSTM MSE:   {delta_lstm_mse:.6f}"
    )

    print(
        f"Delta-LSTM MAE:   {delta_lstm_mae:.6f}"
    )

    print(
        f"Persistence MSE:  {persistence_mse:.6f}"
    )

    print(
        f"Persistence MAE:  {persistence_mae:.6f}"
    )

    print(
        f"Improvement:      {improvement:.2f}%"
    )

    predicted_state_np = (
        predicted_state.cpu().numpy()
    )

    actual_state_np = (
        actual_next_state.cpu().numpy()
    )

    print("\n==============================")
    print("SAMPLE PREDICTIONS")
    print("==============================")

    for i in range(5):

        print(
            f"\nTest sample {i + 1}"
        )

        print(
            "Actual:    ",
            np.round(
                actual_state_np[i][:10],
                3
            )
        )

        print(
            "Predicted: ",
            np.round(
                predicted_state_np[i][:10],
                3
            )
        )

    persistence_errors = torch.abs(
        persistence_state -
        actual_next_state
    ).cpu().numpy()

    delta_errors = torch.abs(
        predicted_state -
        actual_next_state
    ).cpu().numpy()

    print("\n==============================")
    print("PER-FEATURE ERROR ANALYSIS")
    print("==============================")

    print(
        f"{'Feature':35s}"
        f"{'Persistence MAE':>20s}"
        f"{'Delta-LSTM MAE':>20s}"
    )

    print("-" * 75)

    for i, feature in enumerate(
        feature_columns
    ):

        persistence_feature_mae = (
            persistence_errors[:, i].mean()
        )

        delta_feature_mae = (
            delta_errors[:, i].mean()
        )

        print(
            f"{feature:35s}"
            f"{persistence_feature_mae:20.6f}"
            f"{delta_feature_mae:20.6f}"
        )

    persistence_squared_errors = (
        (
            persistence_state -
            actual_next_state
        ) ** 2
    ).cpu().numpy()

    delta_squared_errors = (
        (
            predicted_state -
            actual_next_state
        ) ** 2
    ).cpu().numpy()

    print("\n==============================")
    print("PER-FEATURE MSE ANALYSIS")
    print("==============================")

    print(
        f"{'Feature':35s}"
        f"{'Persistence MSE':>20s}"
        f"{'Delta-LSTM MSE':>20s}"
    )

    print("-" * 75)

    for i, feature in enumerate(
        feature_columns
    ):

        persistence_feature_mse = (
            persistence_squared_errors[:, i].mean()
        )

        delta_feature_mse = (
            delta_squared_errors[:, i].mean()
        )

        print(
            f"{feature:35s}"
            f"{persistence_feature_mse:20.6f}"
            f"{delta_feature_mse:20.6f}"
        )


if __name__ == "__main__":
    main()

