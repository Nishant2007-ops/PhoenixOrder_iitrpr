import torch
import torch.nn as nn
import numpy as np

from src.preprocessing.normalize_cse import find_cse_files
from src.temporal.build_experiment_dataset import build_experiment_states
from src.temporal.prepare_lstm_data import prepare_lstm_data
from src.models.lstm_model import NetworkStateLSTM


DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)


def main():

    print("Device:", DEVICE)

    files = find_cse_files()[:3]

    state_groups = build_experiment_states(files)

    (
        X_train,
        y_train,
        X_validation,
        y_validation,
        X_test,
        y_test,
    ) = prepare_lstm_data(state_groups)

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
            "lstm_model.pth",
            map_location=DEVICE
        )
    )

    model.eval()

    with torch.no_grad():
        predictions = model(X_test)

    mse = nn.MSELoss()
    mae = nn.L1Loss()

    lstm_mse = mse(
        predictions,
        y_test
    ).item()

    lstm_mae = mae(
        predictions,
        y_test
    ).item()

    persistence_predictions = X_test[:, -1, :]

    persistence_mse = mse(
        persistence_predictions,
        y_test
    ).item()

    persistence_mae = mae(
        persistence_predictions,
        y_test
    ).item()

    improvement = (
        (persistence_mse - lstm_mse)
        / persistence_mse
    ) * 100

    print("\n==============================")
    print("TEST SET INFERENCE")
    print("==============================")

    print(f"LSTM MSE:          {lstm_mse:.6f}")
    print(f"LSTM MAE:          {lstm_mae:.6f}")
    print(f"Persistence MSE:   {persistence_mse:.6f}")
    print(f"Persistence MAE:   {persistence_mae:.6f}")
    print(f"LSTM improvement:  {improvement:.2f}%")

    predictions = predictions.cpu().numpy()
    actual = y_test.cpu().numpy()

    print("\n==============================")
    print("PREDICTION VARIABILITY")
    print("==============================")

    print(
        "Actual mean std:",
        np.mean(np.std(actual, axis=0))
    )

    print(
        "Prediction mean std:",
        np.mean(np.std(predictions, axis=0))
    )

    print(
        "Actual overall std:",
        np.std(actual)
    )

    print(
        "Prediction overall std:",
        np.std(predictions)
    )

    print("\n==============================")
    print("SAMPLE PREDICTIONS")
    print("==============================")

    for i in range(5):

        print(f"\nTest sample {i + 1}")

        print(
            "Actual:    ",
            np.round(actual[i][:10], 3)
        )

        print(
            "Predicted: ",
            np.round(predictions[i][:10], 3)
        )


if __name__ == "__main__":
    main()

