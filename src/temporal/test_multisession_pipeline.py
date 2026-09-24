from pathlib import Path

from src.preprocessing.normalize_cse import find_cse_files
from src.temporal.build_multisession_dataset import build_multisession_states
from src.temporal.prepare_delta_lstm_data import (
    prepare_multisession_delta_lstm_data
)


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


def main():

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

    print("\n==============================")
    print("MULTI-SESSION PIPELINE")
    print("==============================")

    print("\nTraining sessions:")
    for file in train_files:
        print(" -", file.name)

    print("\nValidation sessions:")
    for file in validation_files:
        print(" -", file.name)

    print("\nTest sessions:")
    for file in test_files:
        print(" -", file.name)

    (
        train_sessions,
        validation_sessions,
        test_sessions
    ) = build_multisession_states(
        train_files,
        validation_files,
        test_files
    )

    print("\n==============================")
    print("STATE COUNTS")
    print("==============================")

    for file, states in zip(
        train_files,
        train_sessions
    ):
        print(
            "TRAIN",
            file.name,
            "->",
            len(states),
            "states"
        )

    for file, states in zip(
        validation_files,
        validation_sessions
    ):
        print(
            "VALIDATION",
            file.name,
            "->",
            len(states),
            "states"
        )

    for file, states in zip(
        test_files,
        test_sessions
    ):
        print(
            "TEST",
            file.name,
            "->",
            len(states),
            "states"
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
        sequence_length=12
    )

    print("\n==============================")
    print("SEQUENCE COUNTS")
    print("==============================")

    print("X_train:", X_train.shape)
    print("y_train:", y_train.shape)

    print("X_validation:", X_validation.shape)
    print("y_validation:", y_validation.shape)

    print("X_test:", X_test.shape)
    print("y_test:", y_test.shape)

    print("\nNumber of features:", len(feature_columns))
    print("Sequence length:", X_train.shape[1])

    print("\n==============================")
    print("PIPELINE READY")
    print("==============================")


if __name__ == "__main__":
    main()
