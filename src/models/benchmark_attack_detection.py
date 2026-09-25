"""Offline, validation-tuned comparison of Delta-LSTM and Logistic Regression."""

from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score

from src.features.behavioral_detector import analyze_behavioral_transition
from src.models.lstm_model import NetworkStateLSTM
from src.preprocessing.normalize_cse import find_cse_files
from src.temporal.build_multisession_dataset import build_multisession_states
from src.temporal.prepare_delta_lstm_data import (
    LOG_FEATURES,
    prepare_multisession_delta_lstm_data,
    prepare_session_states,
)
from src.temporal.create_sequences import create_delta_sequences


SEQUENCE_LENGTH = 12
HIDDEN_SIZE = 64
NUM_LAYERS = 2
EPOCHS = 20
LEARNING_RATE = 0.001
BATCH_SIZE = 32
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
OUTPUT_CSV = Path("reports/attack_detection_benchmark.csv")

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


def _get_files():
    file_map = {path.name: path for path in find_cse_files()}
    requested = TRAIN_NAMES + VALIDATION_NAMES + TEST_NAMES
    missing = sorted(set(requested) - set(file_map))
    if missing:
        raise FileNotFoundError("Missing CSE benchmark files: " + ", ".join(missing))
    return tuple([file_map[name] for name in names] for names in
                 (TRAIN_NAMES, VALIDATION_NAMES, TEST_NAMES))


def _segments(states):
    if "segment_id" in states.columns:
        return [segment.sort_values("flow_start_time").reset_index(drop=True)
                for _, segment in states.groupby("segment_id", sort=True)]
    return [states.sort_values("flow_start_time").reset_index(drop=True)]


def _targets(sessions):
    """Next-window labels and ratios, aligned to the existing delta sequences."""
    binary, ratios, observed = [], [], []
    for session in sessions:
        for segment in _segments(session):
            binary.append(segment["is_attack"].to_numpy(dtype=np.int8)[SEQUENCE_LENGTH:])
            ratios.append(segment["attack_ratio"].to_numpy(dtype=float)[SEQUENCE_LENGTH:])
            observed.append(segment["total_flows"].to_numpy(dtype=int)[SEQUENCE_LENGTH:] > 0)
    return np.concatenate(binary), np.concatenate(ratios), np.concatenate(observed)


def _fit_lstm(X_train, y_train, X_validation, y_validation):
    torch.manual_seed(42)
    model = NetworkStateLSTM(
        input_size=X_train.shape[2],
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        output_size=y_train.shape[1],
    ).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    train_x = torch.as_tensor(X_train, dtype=torch.float32)
    train_y = torch.as_tensor(y_train, dtype=torch.float32)
    val_x = torch.as_tensor(X_validation, dtype=torch.float32, device=DEVICE)
    val_y = torch.as_tensor(y_validation, dtype=torch.float32, device=DEVICE)
    best_loss, best_state = float("inf"), None

    for epoch in range(EPOCHS):
        model.train()
        order = torch.randperm(len(train_x))
        for start in range(0, len(order), BATCH_SIZE):
            indices = order[start:start + BATCH_SIZE]
            batch_x = train_x[indices].to(DEVICE)
            batch_y = train_y[indices].to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(val_x), val_y).item()
        print(f"Delta-LSTM epoch {epoch + 1:02d}/{EPOCHS}, validation MSE={val_loss:.6f}")
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = deepcopy(model.state_dict())

    if best_state is None:
        raise ValueError("Unable to select a Delta-LSTM checkpoint from validation loss.")
    model.load_state_dict(best_state)
    model.eval()
    return model


def _inverse_state(scaled_state, scaler, feature_columns):
    raw = scaler.inverse_transform(np.asarray(scaled_state).reshape(1, -1))[0]
    for index, name in enumerate(feature_columns):
        if name in LOG_FEATURES:
            raw[index] = np.expm1(raw[index])
    return raw


def _behavior_scores(model, sessions, scaler, feature_columns):
    scores, target_binary, target_ratios = [], [], []
    for session in sessions:
        for segment in _segments(session):
            prepared = prepare_session_states(segment, feature_columns)
            scaled = scaler.transform(prepared).astype(np.float32)
            X, _ = create_delta_sequences(scaled, SEQUENCE_LENGTH)
            if len(X) == 0:
                continue
            with torch.no_grad():
                current = torch.as_tensor(X[:, -1, :], dtype=torch.float32, device=DEVICE)
                predicted_delta = model(
                    torch.as_tensor(X, dtype=torch.float32, device=DEVICE)
                )
                predicted = (current + predicted_delta).cpu().numpy()
                current = current.cpu().numpy()
            for current_scaled, predicted_scaled in zip(current, predicted):
                analysis = analyze_behavioral_transition(
                    _inverse_state(current_scaled, scaler, feature_columns),
                    _inverse_state(predicted_scaled, scaler, feature_columns),
                    feature_columns,
                )
                scores.append(analysis["behavior_score"])
            target_binary.append(segment["is_attack"].to_numpy(dtype=np.int8)[SEQUENCE_LENGTH:])
            target_ratios.append(segment["attack_ratio"].to_numpy(dtype=float)[SEQUENCE_LENGTH:])
    return np.asarray(scores), np.concatenate(target_binary), np.concatenate(target_ratios)


def _select_threshold(y_true, scores):
    if not len(y_true) or not np.isfinite(scores).all():
        raise ValueError("Validation labels/scores are empty or non-finite.")
    unique = np.unique(scores)
    candidates = np.unique(np.concatenate((
        [np.nextafter(unique[0], -np.inf)],
        (unique[:-1] + unique[1:]) / 2 if len(unique) > 1 else [],
        [unique[-1]],
    )))
    # Deterministic tie break favors fewer false positives (higher threshold).
    return max(
        candidates,
        key=lambda threshold: (
            f1_score(y_true, scores >= threshold, zero_division=0), threshold
        ),
    )


def _report_row(name, split, threshold, y_true, scores, ratios):
    predictions = (scores >= threshold).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    return {
        "model": name,
        "split": split,
        "threshold": float(threshold),
        "windows": int(len(y_true)),
        "benign_windows": int((y_true == 0).sum()),
        "attack_windows": int((y_true == 1).sum()),
        "mean_attack_ratio": float(np.mean(ratios)) if len(ratios) else float("nan"),
        "min_attack_ratio": float(np.min(ratios)) if len(ratios) else float("nan"),
        "max_attack_ratio": float(np.max(ratios)) if len(ratios) else float("nan"),
        "precision": float(tp / (tp + fp)) if tp + fp else 0.0,
        "recall": float(tp / (tp + fn)) if tp + fn else 0.0,
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "false_positive_rate": float(fp / (fp + tn)) if fp + tn else 0.0,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def _print_split(name, y, ratios):
    print(f"{name}: windows={len(y)}, benign={(y == 0).sum()}, attack={(y == 1).sum()}, "
          f"mean_attack_ratio={ratios.mean():.6f}, positive_rule=attack_ratio > 0")


def main():
    train_files, validation_files, test_files = _get_files()
    train, validation, test = build_multisession_states(
        train_files, validation_files, test_files
    )
    (X_train, y_delta_train, X_val, y_delta_val, X_test, _, feature_columns,
     scaler) = prepare_multisession_delta_lstm_data(
        train, validation, test, sequence_length=SEQUENCE_LENGTH
    )
    y_train_all, ratio_train_all, observed_train = _targets(train)
    y_val_all, ratio_val_all, observed_val = _targets(validation)
    y_test_all, ratio_test_all, observed_test = _targets(test)

    if not (len(X_train) == len(y_train_all) and len(X_val) == len(y_val_all)
            and len(X_test) == len(y_test_all)):
        raise RuntimeError("Feature/label alignment failed; refusing to evaluate.")
    # Resampling may create empty time bins. They have no ground-truth label,
    # so omit them as prediction targets for both models (they may remain in
    # the preceding context, consistently with the existing temporal design).
    X_train, y_train = X_train[observed_train], y_train_all[observed_train]
    X_val, y_val = X_val[observed_val], y_val_all[observed_val]
    X_test, y_test = X_test[observed_test], y_test_all[observed_test]
    ratio_train, ratio_val, ratio_test = (
        ratio_train_all[observed_train],
        ratio_val_all[observed_val],
        ratio_test_all[observed_test],
    )
    if not len(X_train) or np.unique(y_train).size != 2:
        raise ValueError("Training windows must include benign and attack examples.")
    if not len(X_val) or not len(X_test):
        raise ValueError("Validation and test splits must each contain windows.")

    _print_split("Train", y_train, ratio_train)
    _print_split("Validation", y_val, ratio_val)
    _print_split("Test", y_test, ratio_test)

    # The Logistic Regression sees exactly the same past-state context windows.
    logistic = LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=42
    )
    logistic.fit(X_train.reshape(len(X_train), -1), y_train)
    logistic_val = logistic.predict_proba(X_val.reshape(len(X_val), -1))[:, 1]
    logistic_test = logistic.predict_proba(X_test.reshape(len(X_test), -1))[:, 1]
    logistic_threshold = _select_threshold(y_val, logistic_val)

    lstm = _fit_lstm(X_train, y_delta_train, X_val, y_delta_val)
    lstm_val_all, val_y_all_from_score, val_ratio_all_from_score = _behavior_scores(
        lstm, validation, scaler, feature_columns
    )
    lstm_test_all, test_y_all_from_score, test_ratio_all_from_score = _behavior_scores(
        lstm, test, scaler, feature_columns
    )
    if (not np.array_equal(val_y_all_from_score, y_val_all)
            or not np.array_equal(test_y_all_from_score, y_test_all)):
        raise RuntimeError("Delta-LSTM score windows do not align with Logistic Regression windows.")
    lstm_val = lstm_val_all[observed_val]
    lstm_test = lstm_test_all[observed_test]
    if (not np.array_equal(val_ratio_all_from_score, ratio_val_all)
            or not np.array_equal(test_ratio_all_from_score, ratio_test_all)):
        raise RuntimeError("Attack ratios do not align between model scoring and labels.")
    if len(lstm_test) != len(logistic_test):
        raise RuntimeError("Models do not cover identical test windows.")
    lstm_threshold = _select_threshold(y_val, lstm_val)

    rows = []
    for model_name, threshold, val_scores, test_scores in (
        ("Logistic Regression", logistic_threshold, logistic_val, logistic_test),
        ("Delta-LSTM behavioral score", lstm_threshold, lstm_val, lstm_test),
    ):
        rows.append(_report_row(model_name, "validation", threshold, y_val,
                                val_scores, ratio_val))
        row = _report_row(model_name, "test", threshold, y_test,
                          test_scores, ratio_test)
        rows.append(row)
        print(f"\n{model_name} test confusion matrix [[tn, fp], [fn, tp]]: "
              f"[[{row['tn']}, {row['fp']}], [{row['fn']}, {row['tp']}]]")
        print(f"precision={row['precision']:.4f} recall={row['recall']:.4f} "
              f"F1={row['f1']:.4f} FPR={row['false_positive_rate']:.4f} "
              f"(threshold fixed from validation={threshold:.8g})")

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    import pandas as pd
    pd.DataFrame(rows).to_csv(OUTPUT_CSV, index=False)
    print(f"\nComparison saved: {OUTPUT_CSV}")
    print("Limitations: binary ground truth marks any window with attack_ratio > 0 as attack; "
          "attack_ratio is retained in the report. The LSTM behavioral score is a heuristic "
          "from predicted feature changes, not a calibrated attack probability. Thresholds "
          "are selected on validation F1 and frozen for test. Windows within a day are "
          "temporally correlated, and results are specific to the fixed date split.")


if __name__ == "__main__":
    main()
