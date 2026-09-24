import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from src.preprocessing.normalize_cse import find_cse_files
from src.temporal.build_multisession_dataset import build_multisession_states
from src.temporal.prepare_delta_lstm_data import (
    prepare_multisession_delta_lstm_data
)
from src.models.lstm_model import NetworkStateLSTM


DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

SEQUENCE_LENGTH = 12
INPUT_SIZE = 32
HIDDEN_SIZE = 64
NUM_LAYERS = 2
OUTPUT_SIZE = 32

BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001


TRAIN_NAMES = [
    "Friday-02-03-2018_TrafficForML_CICFlowMeter.csv",
    "Thursday-01-03-2018_TrafficForML_CICFlowMeter.csv",
    "Wednesday-28-02-2018_TrafficForML_CICFlowMeter.csv",
    "Friday-23-02-2018_TrafficForML_CICFlowMeter.csv",
    "Thursday-22-02-2018_TrafficForML_CICFlowMeter.csv",
    "Wednesday-21-02-2018_TrafficForML_CICFlowMeter.csv",
]

VALIDATION_NAMES = [
    "Friday-16-02-2018_TrafficForML_CICFlowMeter.csv",
    "Thursday-15-02-2018_TrafficForML_CICFlowMeter.csv",
]

TEST_NAMES = [
    "Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv",
    "Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv",
]


def get_files():

    files = find_cse_files()

    file_map = {
        file.name: file
        for file in files
    }

    train_files = [
        file_map[name]
        for name in TRAIN_NAMES
    ]

    validation_files = [
        file_map[name]
        for name in VALIDATION_NAMES
    ]

    test_files = [
        file_map[name]
        for name in TEST_NAMES
    ]

    return (
        train_files,
        validation_files,
        test_files
    )


def main():

    print("==============================")
    print("MULTI-SESSION DELTA-LSTM")
    print("==============================")

    print("Device:", DEVICE)

    (
        train_files,
        validation_files,
        test_files
    ) = get_files()

    (
        train_sessions,
        validation_sessions,
        test_sessions
    ) = build_multisession_states(
        train_files,
        validation_files,
        test_files
    )

    (
        X_train,
        y_train,
        X_validation,
        y_validation,
        X_test,
        y_test,
        feature_columns,
        scaler
    ) = prepare_multisession_delta_lstm_data(
        train_sessions,
        validation_sessions,
        test_sessions,
        sequence_length=SEQUENCE_LENGTH
    )

    print("\n==============================")
    print("DATA")
    print("==============================")

    print("Training:", X_train.shape)
    print("Validation:", X_validation.shape)
    print("Test:", X_test.shape)
    print("Features:", len(feature_columns))

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

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    X_validation = X_validation.to(DEVICE)
    y_validation = y_validation.to(DEVICE)

    model = NetworkStateLSTM(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        output_size=OUTPUT_SIZE
    ).to(DEVICE)

    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    best_validation_loss = float("inf")

    print("\n==============================")
    print("TRAINING")
    print("==============================")

    for epoch in range(EPOCHS):

        model.train()

        total_train_loss = 0.0

        for batch_X, batch_y in train_loader:

            batch_X = batch_X.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            optimizer.zero_grad()

            prediction = model(batch_X)

            loss = criterion(
                prediction,
                batch_y
            )

            loss.backward()

            optimizer.step()

            total_train_loss += (
                loss.item() * batch_X.size(0)
            )

        train_loss = (
            total_train_loss
            / len(train_dataset)
        )

        model.eval()

        with torch.no_grad():

            validation_prediction = model(
                X_validation
            )

            validation_loss = criterion(
                validation_prediction,
                y_validation
            ).item()

        print(
            f"Epoch {epoch + 1:02d}/{EPOCHS} "
            f"| Train Loss: {train_loss:.6f} "
            f"| Validation Loss: {validation_loss:.6f}"
        )

        if validation_loss < best_validation_loss:

            best_validation_loss = validation_loss

            torch.save(
                model.state_dict(),
                "multisession_delta_lstm_model.pth"
            )

    print("\n==============================")
    print("TRAINING COMPLETE")
    print("==============================")

    print(
        "Best validation loss:",
        f"{best_validation_loss:.6f}"
    )

    print(
        "Model saved as:",
        "multisession_delta_lstm_model.pth"
    )


if __name__ == "__main__":
    main()
