"""Test-only inference for the saved multitask Delta-LSTM checkpoint."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

from src.models.lstm_model import MultitaskNetworkStateLSTM
from src.preprocessing.normalize_cse import find_cse_files
from src.temporal.build_multisession_dataset import build_states_for_file
from src.temporal.create_sequences import create_delta_sequences
from src.temporal.prepare_delta_lstm_data import prepare_session_states


SEQUENCE_LENGTH = 12
THRESHOLD = 0.39090657
CHECKPOINT_PATH = Path("multitask_delta_lstm_checkpoint.pth")
SCALER_PATH = Path("multitask_delta_lstm_scaler.joblib")
FEATURES_PATH = Path("multitask_delta_lstm_features.json")
PREVIOUS_BENCHMARK = Path("reports/attack_detection_benchmark.csv")
PREDICTIONS_OUTPUT = Path("reports/multitask_delta_lstm_test_predictions.csv")
COMPARISON_OUTPUT = Path("reports/attack_detection_comparison_with_multitask.csv")
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

TEST_NAMES = [
    "Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv",
    "Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv",
]


def _test_files():
    file_map = {path.name: path for path in find_cse_files()}
    missing = [name for name in TEST_NAMES if name not in file_map]
    if missing:
        raise FileNotFoundError("Missing test files: " + ", ".join(missing))
    return [file_map[name] for name in TEST_NAMES]


def _load_saved_model(feature_order):
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=True,
    )
    saved_threshold = float(checkpoint["validation_attack_score_threshold"])
    if not np.isclose(saved_threshold, THRESHOLD, rtol=0.0, atol=5e-9):
        raise ValueError(
            f"Checkpoint validation threshold {saved_threshold:.12g} does not match "
            f"the requested fixed threshold {THRESHOLD:.8f}."
        )
    if checkpoint["feature_order"] != feature_order:
        raise ValueError("Feature JSON does not match checkpoint feature ordering.")
    if checkpoint["sequence_length"] != SEQUENCE_LENGTH:
        raise ValueError("Checkpoint sequence length is not 12.")

    model = MultitaskNetworkStateLSTM(
        input_size=checkpoint["input_size"],
        hidden_size=checkpoint["hidden_size"],
        num_layers=checkpoint["num_layers"],
        output_size=checkpoint["output_size"],
    ).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def _infer_test(model, scaler, feature_order, test_files):
    rows = []
    for file_path in test_files:
        states = build_states_for_file(file_path)
        segments = (
            states.groupby("segment_id", sort=True)
            if "segment_id" in states.columns
            else [(0, states)]
        )
        for segment_id, segment in segments:
            segment = segment.sort_values("flow_start_time").reset_index(drop=True)
            prepared = prepare_session_states(segment, feature_order)
            scaled = scaler.transform(prepared.to_numpy()).astype(np.float32)
            X, _ = create_delta_sequences(scaled, SEQUENCE_LENGTH)
            if not len(X):
                continue

            with torch.no_grad():
                _, attack_logits = model(
                    torch.as_tensor(X, dtype=torch.float32, device=DEVICE)
                )
                probabilities = torch.sigmoid(attack_logits).cpu().numpy()

            next_states = segment.iloc[SEQUENCE_LENGTH:].reset_index(drop=True)
            if len(next_states) != len(probabilities):
                raise RuntimeError("Sequence scores are not aligned to next-state rows.")
            for sequence_index, (probability, target_state) in enumerate(
                zip(probabilities, next_states.to_dict("records"))
            ):
                if int(target_state["total_flows"]) == 0:
                    continue
                rows.append({
                    "session": file_path.name,
                    "segment_id": int(segment_id),
                    "sequence_index": sequence_index,
                    "target_timestamp": target_state["flow_start_time"],
                    "attack_ratio": float(target_state["attack_ratio"]),
                    "ground_truth_attack": int(target_state["is_attack"]),
                    "attack_probability": float(probability),
                    "threshold": THRESHOLD,
                    "predicted_attack": int(probability >= THRESHOLD),
                })
    return pd.DataFrame(rows)


def _metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "windows": int(len(y_true)),
        "benign_windows": int((y_true == 0).sum()),
        "attack_windows": int((y_true == 1).sum()),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "false_positive_rate": float(fp / (fp + tn)) if fp + tn else 0.0,
        "predicted_attack_windows": int(y_pred.sum()),
        "predicted_benign_windows": int(len(y_pred) - y_pred.sum()),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def _previous_rows(y_true):
    previous = pd.read_csv(PREVIOUS_BENCHMARK)
    previous = previous[previous["split"] == "test"].copy()
    expected_names = {"Logistic Regression", "Delta-LSTM behavioral score"}
    if set(previous["model"]) != expected_names:
        raise RuntimeError("Previous benchmark CSV is missing one or more test models.")
    expected_benign = int((y_true == 0).sum())
    expected_attack = int((y_true == 1).sum())
    for row in previous.to_dict("records"):
        if (int(row["windows"]) != len(y_true)
                or int(row["benign_windows"]) != expected_benign
                or int(row["attack_windows"]) != expected_attack
                or int(row["tn"] + row["fp"]) != expected_benign
                or int(row["fn"] + row["tp"]) != expected_attack):
            raise RuntimeError(
                f"Prior benchmark ground-truth distribution differs for {row['model']}."
            )
    return previous


def main():
    for path in (CHECKPOINT_PATH, SCALER_PATH, FEATURES_PATH, PREVIOUS_BENCHMARK):
        if not path.exists():
            raise FileNotFoundError(path)

    feature_order = json.loads(FEATURES_PATH.read_text(encoding="utf-8"))
    scaler = joblib.load(SCALER_PATH)
    if scaler.n_features_in_ != len(feature_order):
        raise RuntimeError("Saved scaler dimensionality does not match saved feature JSON.")
    if len(feature_order) != 32:
        raise RuntimeError(f"Expected 32 saved features, found {len(feature_order)}.")
    model, checkpoint = _load_saved_model(feature_order)

    predictions = _infer_test(model, scaler, feature_order, _test_files())
    if len(predictions) != 218:
        raise RuntimeError(f"Expected the prior benchmark's 218 windows, got {len(predictions)}.")
    y_true = predictions["ground_truth_attack"].to_numpy(dtype=np.int8)
    y_pred = predictions["predicted_attack"].to_numpy(dtype=np.int8)
    new_metrics = _metrics(y_true, y_pred)
    if (new_metrics["benign_windows"], new_metrics["attack_windows"]) != (164, 54):
        raise RuntimeError("New inference labels do not match prior benchmark class counts.")

    previous = _previous_rows(y_true)
    new_row = {
        "model": "New Multitask Delta-LSTM",
        "split": "test",
        "threshold": THRESHOLD,
        **new_metrics,
    }
    compare_cols = [
        "model", "split", "threshold", "windows", "benign_windows", "attack_windows",
        "precision", "recall", "f1", "false_positive_rate",
        "predicted_attack_windows", "predicted_benign_windows", "tn", "fp", "fn", "tp",
    ]
    # Retain old benchmark values verbatim; only add predicted-count fields for table consistency.
    old_rows = []
    for row in previous.to_dict("records"):
        old_rows.append({
            "model": row["model"],
            "split": "test",
            "threshold": row["threshold"],
            "windows": row["windows"],
            "benign_windows": row["benign_windows"],
            "attack_windows": row["attack_windows"],
            "precision": row["precision"],
            "recall": row["recall"],
            "f1": row["f1"],
            "false_positive_rate": row["false_positive_rate"],
            "predicted_attack_windows": int(row["fp"] + row["tp"]),
            "predicted_benign_windows": int(row["tn"] + row["fn"]),
            "tn": row["tn"], "fp": row["fp"], "fn": row["fn"], "tp": row["tp"],
        })
    comparison = pd.DataFrame(old_rows + [new_row], columns=compare_cols)

    PREDICTIONS_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(PREDICTIONS_OUTPUT, index=False)
    comparison.to_csv(COMPARISON_OUTPUT, index=False)

    print("Test class distribution:",
          f"benign={new_metrics['benign_windows']}, attack={new_metrics['attack_windows']}")
    print("Confusion matrix [[TN, FP], [FN, TP]]:",
          [[new_metrics["tn"], new_metrics["fp"]],
           [new_metrics["fn"], new_metrics["tp"]]])
    for key in ("precision", "recall", "f1", "false_positive_rate",
                "predicted_attack_windows", "predicted_benign_windows"):
        print(f"{key}: {new_metrics[key]}")
    print("\nComparison:")
    print(comparison.to_string(index=False))
    print(f"\nPer-window predictions: {PREDICTIONS_OUTPUT}")
    print(f"Separate comparison: {COMPARISON_OUTPUT}")
    print(f"Loaded checkpoint from selected epoch {checkpoint['best_epoch']} with fixed threshold {THRESHOLD:.8f}.")
    print("Verification: the reconstructed test IDs use the prior benchmark's two test files, "
          "12-state contexts and next-state targets; test label counts match both prior model rows. "
          "The prior benchmark CSV contains aggregate metrics only, so it has no stored per-window "
          "IDs for a row-by-row identity comparison.")


if __name__ == "__main__":
    main()
