import torch
import torch.nn as nn
import numpy as np

from src.models.lstm_model import NetworkStateLSTM
from src.temporal.build_experiment_dataset import build_experiment_states
from src.temporal.prepare_lstm_data import prepare_lstm_data
from src.preprocessing.normalize_cse import find_cse_files


DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

EPOCHS = 20
LEARNING_RATE = 0.001
BATCH_SIZE = 64


def train_model(
    model,
    X_train,
    y_train,
    X_validation,
    y_validation
):
    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    for epoch in range(EPOCHS):
        model.train()

        permutation = torch.randperm(
            X_train.size(0)
        )

        for i in range(
            0,
            X_train.size(0),
            BATCH_SIZE
        ):
            indices = permutation[
                i:i + BATCH_SIZE
            ]

            X = X_train[indices].to(DEVICE)
            y = y_train[indices].to(DEVICE)

            optimizer.zero_grad()

            predictions = model(X)

            loss = criterion(
                predictions,
                y
            )

            loss.backward()
            optimizer.step()

        model.eval()

        with torch.no_grad():
            predictions = model(
                X_validation.to(DEVICE)
            )

            validation_loss = criterion(
                predictions,
                y_validation.to(DEVICE)
            )

        print(
            f"Epoch {epoch + 1:02d}/{EPOCHS} "
            f"| Validation Loss: "
            f"{validation_loss.item():.6f}"
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
    ) = prepare_lstm_data(
        state_groups
    )

    X_train = torch.tensor(
        X_train,
        dtype=torch.float32
    )

    y_train = torch.tensor(
        y_train,
        dtype=torch.float32
    )

    X_validation = torch.tensor(
        X_validation,
        dtype=torch.float32
    )

    y_validation = torch.tensor(
        y_validation,
        dtype=torch.float32
    )

    X_test = torch.tensor(
        X_test,
        dtype=torch.float32
    )

    y_test = torch.tensor(
        y_test,
        dtype=torch.float32
    )

    model = NetworkStateLSTM(
        input_size=32,
        hidden_size=64,
        num_layers=2,
        output_size=32
    ).to(DEVICE)

    print("\nTraining model...\n")

    train_model(
        model,
        X_train,
        y_train,
        X_validation,
        y_validation
    )

    model.eval()

    with torch.no_grad():
        predictions = model(
            X_test.to(DEVICE)
        )

    predictions = predictions.cpu().numpy()
    actual = y_test.numpy()

    lstm_mse = np.mean(
        (predictions - actual) ** 2
    )

    lstm_mae = np.mean(
        np.abs(predictions - actual)
    )

    persistence_predictions = (
        X_test[:, -1, :].numpy()
    )

    baseline_mse = np.mean(
        (persistence_predictions - actual) ** 2
    )

    baseline_mae = np.mean(
        np.abs(
            persistence_predictions - actual
        )
    )

    improvement = (
        (baseline_mse - lstm_mse)
        / baseline_mse
        * 100
    )

    print("\n==============================")
    print("FINAL TEST RESULTS")
    print("==============================")

    print(
        f"LSTM MSE:          {lstm_mse:.6f}"
    )

    print(
        f"LSTM MAE:          {lstm_mae:.6f}"
    )

    print(
        f"Persistence MSE:   {baseline_mse:.6f}"
    )

    print(
        f"Persistence MAE:   {baseline_mae:.6f}"
    )

    print(
        f"LSTM improvement:  {improvement:.2f}%"
    )


if __name__ == "__main__":
    main()

