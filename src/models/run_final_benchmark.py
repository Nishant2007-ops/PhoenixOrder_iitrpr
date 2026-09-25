"""Reproduce the saved multitask model's final test benchmark without training."""

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

from src.models.analyze_multitask_threshold_sensitivity import _score_split
from src.models.benchmark_attack_detection import (
    TEST_NAMES,
    TRAIN_NAMES,
    VALIDATION_NAMES,
)
from src.models.lstm_model import MultitaskNetworkStateLSTM
from src.models.train_multitask_delta_lstm import select_validation_score_threshold
from src.preprocessing.normalize_cse import find_cse_files


CHECKPOINT_PATH = Path("multitask_delta_lstm_checkpoint.pth")
SCALER_PATH = Path("multitask_delta_lstm_scaler.joblib")
FEATURES_PATH = Path("multitask_delta_lstm_features.json")
BASELINE_PATH = Path("reports/attack_detection_benchmark.csv")
OUTPUT_PATH = Path("reports/final_benchmark.csv")
SEQUENCE_LENGTH = 12
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metric_row(name, threshold, labels, predictions, source):
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "model": name,
        "split": "test",
        "threshold": float(threshold),
        "windows": int(len(labels)),
        "benign_windows": int((labels == 0).sum()),
        "attack_windows": int((labels == 1).sum()),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "false_positive_rate": float(fp / (fp + tn)) if fp + tn else 0.0,
        "predicted_attack_windows": int(predictions.sum()),
        "predicted_benign_windows": int(len(predictions) - predictions.sum()),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "result_source": source,
    }


def main():
    for artifact in (CHECKPOINT_PATH, SCALER_PATH, FEATURES_PATH, BASELINE_PATH):
        if not artifact.is_file():
            raise FileNotFoundError(artifact)

    # Load each requested saved artifact from disk and verify its relationship
    # to the model checkpoint; none is reconstructed or refit here.
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=True)
    scaler = joblib.load(SCALER_PATH)
    feature_order = json.loads(FEATURES_PATH.read_text(encoding="utf-8"))
    if checkpoint["feature_order"] != feature_order:
        raise RuntimeError("Saved feature ordering does not match checkpoint metadata.")
    if scaler.n_features_in_ != len(feature_order) or len(feature_order) != 32:
        raise RuntimeError("Saved scaler does not match the 32 saved features.")
    if checkpoint["sequence_length"] != SEQUENCE_LENGTH:
        raise RuntimeError("Saved checkpoint does not use 12-state sequences.")
    if (checkpoint["training_files"] != TRAIN_NAMES
            or checkpoint["validation_files"] != VALIDATION_NAMES):
        raise RuntimeError("Saved checkpoint split metadata differs from fixed configuration.")

    model = MultitaskNetworkStateLSTM(
        input_size=checkpoint["input_size"],
        hidden_size=checkpoint["hidden_size"],
        num_layers=checkpoint["num_layers"],
        output_size=checkpoint["output_size"],
    ).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    file_map = {path.name: path for path in find_cse_files()}
    missing = [name for name in VALIDATION_NAMES + TEST_NAMES if name not in file_map]
    if missing:
        raise FileNotFoundError("Missing validation/test sessions: " + ", ".join(missing))
    validation_files = [file_map[name] for name in VALIDATION_NAMES]
    test_files = [file_map[name] for name in TEST_NAMES]

    validation = _score_split(
        model, scaler, feature_order, validation_files, "validation"
    )
    validation_labels = validation["ground_truth_attack"].to_numpy(dtype=np.int8)
    validation_scores = validation["attack_probability"].to_numpy(dtype=float)
    reproduced_threshold = select_validation_score_threshold(
        validation_labels, validation_scores, minimum_recall=0.80
    )
    saved_threshold = float(checkpoint["validation_attack_score_threshold"])
    if not np.isclose(reproduced_threshold, saved_threshold, rtol=0.0, atol=1e-7):
        raise RuntimeError(
            "Recomputed validation threshold differs from saved checkpoint metadata: "
            f"{reproduced_threshold:.12g} vs {saved_threshold:.12g}"
        )

    # Only after freezing the validation-derived threshold do we score test.
    test = _score_split(model, scaler, feature_order, test_files, "test evaluation")
    test_labels = test["ground_truth_attack"].to_numpy(dtype=np.int8)
    test_probabilities = test["attack_probability"].to_numpy(dtype=float)
    if len(test) != 218:
        raise RuntimeError(f"Expected exactly 218 test windows; found {len(test)}.")
    if (int((test_labels == 0).sum()), int((test_labels == 1).sum())) != (164, 54):
        raise RuntimeError("Test label distribution differs from previous benchmark.")
    multitask_predictions = (test_probabilities >= saved_threshold).astype(np.int8)

    previous = pd.read_csv(BASELINE_PATH)
    previous_test = previous[previous["split"] == "test"]
    logistic = previous_test[previous_test["model"] == "Logistic Regression"]
    if len(logistic) != 1:
        raise RuntimeError("Could not identify exactly one existing Logistic Regression test row.")
    logistic = logistic.iloc[0]
    if (int(logistic["windows"]) != 218
            or int(logistic["benign_windows"]) != 164
            or int(logistic["attack_windows"]) != 54):
        raise RuntimeError("Existing Logistic Regression baseline has a different test set.")

    # Copy the existing baseline's aggregate test metrics without fitting it again.
    logistic_row = {
        "model": "Logistic Regression",
        "split": "test",
        "threshold": logistic["threshold"],
        "windows": int(logistic["windows"]),
        "benign_windows": int(logistic["benign_windows"]),
        "attack_windows": int(logistic["attack_windows"]),
        "precision": logistic["precision"],
        "recall": logistic["recall"],
        "f1": logistic["f1"],
        "false_positive_rate": logistic["false_positive_rate"],
        "predicted_attack_windows": int(logistic["fp"] + logistic["tp"]),
        "predicted_benign_windows": int(logistic["tn"] + logistic["fn"]),
        "tn": int(logistic["tn"]), "fp": int(logistic["fp"]),
        "fn": int(logistic["fn"]), "tp": int(logistic["tp"]),
        "result_source": "existing reports/attack_detection_benchmark.csv; no retraining",
    }
    multitask_row = _metric_row(
        "Multitask Delta-LSTM",
        saved_threshold,
        test_labels,
        multitask_predictions,
        "saved checkpoint inference; threshold reproduced on validation",
    )
    result = pd.DataFrame([logistic_row, multitask_row])
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_PATH, index=False)

    print("PhoenixOrder final benchmark (test split)")
    print(f"Fixed split: {len(TRAIN_NAMES)} train sessions, "
          f"{len(VALIDATION_NAMES)} validation sessions, {len(TEST_NAMES)} test sessions")
    print(f"Validation windows: {len(validation)}; recomputed threshold={reproduced_threshold:.12g}; "
          f"saved threshold={saved_threshold:.12g}")
    print(f"Test windows: {len(test)} (benign={(test_labels == 0).sum()}, "
          f"attack={(test_labels == 1).sum()})")
    print(f"Loaded checkpoint: {CHECKPOINT_PATH} sha256={_sha256(CHECKPOINT_PATH)}")
    print(f"Loaded scaler: {SCALER_PATH} sha256={_sha256(SCALER_PATH)}")
    print(f"Loaded feature ordering: {FEATURES_PATH} sha256={_sha256(FEATURES_PATH)}")
    for row in result.to_dict("records"):
        print(f"\n{row['model']}: confusion=[[{row['tn']}, {row['fp']}], "
              f"[{row['fn']}, {row['tp']}]]; precision={row['precision']:.4f}; "
              f"recall={row['recall']:.4f}; F1={row['f1']:.4f}; "
              f"FPR={row['false_positive_rate']:.4f}")
    print(f"\nSaved final benchmark: {OUTPUT_PATH}")
    print("The Logistic Regression row was copied from the prior benchmark; it was not refit. "
          "Test labels were used only to compute final metrics, never to set the threshold.")


if __name__ == "__main__":
    main()
