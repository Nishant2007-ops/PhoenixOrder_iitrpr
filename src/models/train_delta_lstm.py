import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from src.preprocessing.normalize_cse import find_cse_files
from src.temporal.build_experiment_dataset import build_experiment_states
from src.temporal.prepare_delta_lstm_data import prepare_delta_lstm_data
from src.models.lstm_model import NetworkStateLSTM


DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 0.001


def train_model(
    model,
    train_loader,
    validation_loader
):

    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    for epoch in range(EPOCHS):

        model.train()

        train_loss = 0.0

        for X, y in train_loader:

            X = X.to(DEVICE)
            y = y.to(DEVICE)

            optimizer.zero_grad()

            predictions = model(X)

            loss = criterion(
                predictions,
                y
            )

            loss.backward()

            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()

        validation_loss = 0.0

        with torch.no_grad():

            for X, y in validation_loader:

                X = X.to(DEVICE)
                y = y.to(DEVICE)

                predictions = model(X)

                loss = criterion(
                    predictions,
                    y
                )

                validation_loss += loss.item()

        validation_loss /= len(validation_loader)

        print(
            f"Epoch {epoch + 1:02d}/{EPOCHS} "
            f"| Train Loss: {train_loss:.6f} "
            f"| Validation Loss: {validation_loss:.6f}"
        )


def main():

    print("Device:", DEVICE)

    files = find_cse_files()[:3]

    state_groups = build_experiment_states(
        files
    )

    print("\nSession sizes:")

    for i, states in enumerate(state_groups):

        print(
            f"Session {i + 1}: "
            f"{len(states)} states"
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

    print("\nDelta data shapes:")

    print("X_train:", X_train.shape)
    print("y_train:", y_train.shape)

    print("X_validation:", X_validation.shape)
    print("y_validation:", y_validation.shape)

    print("X_test:", X_test.shape)
    print("y_test:", y_test.shape)

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

    train_dataset = TensorDataset(
        X_train,
        y_train
    )

    validation_dataset = TensorDataset(
        X_validation,
        y_validation
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    model = NetworkStateLSTM(
        input_size=32,
        hidden_size=64,
        num_layers=2,
        output_size=32
    ).to(DEVICE)

    print("\nModel:")
    print(model)

    print("\nStarting delta training...\n")

    train_model(
        model,
        train_loader,
        validation_loader
    )

    torch.save(
        model.state_dict(),
        "delta_lstm_model.pth"
    )

    print(
        "\nModel saved as delta_lstm_model.pth"
    )


if __name__ == "__main__":
    main()

