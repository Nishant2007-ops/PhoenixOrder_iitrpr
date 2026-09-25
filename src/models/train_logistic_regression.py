"""Binary attack detection baseline for the multisession state benchmark."""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.preprocessing.normalize_cse import find_cse_files
from src.temporal.build_multisession_dataset import build_multisession_states
from src.temporal.prepare_delta_lstm_data import (
    prepare_multisession_delta_lstm_data,
)


SEQUENCE_LENGTH = 12

# Keep this file split identical to the multisession delta LSTM benchmark.
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


def _files_for_split():
    file_map = {path.name: path for path in find_cse_files()}
    missing = sorted(
        set(TRAIN_NAMES + VALIDATION_NAMES + TEST_NAMES) - set(file_map)
    )
    if missing:
        raise FileNotFoundError(
            "Missing benchmark CSV files: " + ", ".join(missing)
        )
    return tuple(
        [file_map[name] for name in names]
        for names in (TRAIN_NAMES, VALIDATION_NAMES, TEST_NAMES)
    )


def _target_labels(sessions, sequence_length):
    """Labels for the next state after each LSTM-style context window."""
    targets = []
    for states in sessions:
        segments = (
            states.groupby("segment_id", sort=True)
            if "segment_id" in states.columns
            else [(None, states)]
        )
        for _, segment in segments:
            labels = segment["is_attack"].to_numpy(dtype=np.int8)
            targets.append(labels[sequence_length:])
    return np.concatenate(targets) if targets else np.empty(0, dtype=np.int8)


def _metrics(y_true, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(np.int8)
    result = {
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, predictions),
        "precision": precision_score(y_true, predictions, zero_division=0),
        "recall": recall_score(y_true, predictions, zero_division=0),
        "f1": f1_score(y_true, predictions, zero_division=0),
    }
    if np.unique(y_true).size == 2:
        result["roc_auc"] = roc_auc_score(y_true, probabilities)
    return result


def main():
    train_files, validation_files, test_files = _files_for_split()
    train_sessions, validation_sessions, test_sessions = build_multisession_states(
        train_files, validation_files, test_files
    )
    (
        X_train,
        _,
        X_validation,
        _,
        X_test,
        _,
        feature_columns,
        _,
    ) = prepare_multisession_delta_lstm_data(
        train_sessions,
        validation_sessions,
        test_sessions,
        sequence_length=SEQUENCE_LENGTH,
    )

    y_train = _target_labels(train_sessions, SEQUENCE_LENGTH)
    y_validation = _target_labels(validation_sessions, SEQUENCE_LENGTH)
    y_test = _target_labels(test_sessions, SEQUENCE_LENGTH)
    X_train = X_train.reshape(len(X_train), -1)
    X_validation = X_validation.reshape(len(X_validation), -1)
    X_test = X_test.reshape(len(X_test), -1)

    if not len(y_train) or np.unique(y_train).size < 2:
        raise ValueError("Training split must contain both benign and attack windows.")

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
    )
    model.fit(X_train, y_train)

    validation_probabilities = model.predict_proba(X_validation)[:, 1]
    thresholds = np.linspace(0.05, 0.95, 91)
    threshold = max(
        thresholds,
        key=lambda value: f1_score(
            y_validation,
            validation_probabilities >= value,
            zero_division=0,
        ),
    )
    test_probabilities = model.predict_proba(X_test)[:, 1]

    print("LOGISTIC REGRESSION ATTACK DETECTION BASELINE")
    print(f"Feature count: {len(feature_columns)} per state")
    print(f"Context: {SEQUENCE_LENGTH} states ({SEQUENCE_LENGTH * len(feature_columns)} inputs)")
    for name, y, probabilities in (
        ("Validation", y_validation, validation_probabilities),
        ("Test", y_test, test_probabilities),
    ):
        print(f"\n{name} ({len(y)} windows)")
        for metric, value in _metrics(y, probabilities, threshold).items():
            print(f"{metric}: {value:.4f}")


if __name__ == "__main__":
    main()
