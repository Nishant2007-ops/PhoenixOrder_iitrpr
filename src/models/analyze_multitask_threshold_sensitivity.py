"""Threshold sensitivity for the saved multitask Delta-LSTM, without training."""

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
CURRENT_THRESHOLD = 0.39090657
TARGET_RECALLS = (0.80, 0.85, 0.90, 0.95, 1.00)
CHECKPOINT_PATH = Path("multitask_delta_lstm_checkpoint.pth")
SCALER_PATH = Path("multitask_delta_lstm_scaler.joblib")
FEATURES_PATH = Path("multitask_delta_lstm_features.json")
SWEEP_OUTPUT = Path("reports/multitask_threshold_sensitivity.csv")
REPORT_OUTPUT = Path("reports/multitask_threshold_sensitivity.md")
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

VALIDATION_NAMES = [
    "Friday-16-02-2018_TrafficForML_CICFlowMeter.csv",
    "Thursday-15-02-2018_TrafficForML_CICFlowMeter.csv",
]
TEST_NAMES = [
    "Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv",
    "Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv",
]


def _load_files(names):
    by_name = {path.name: path for path in find_cse_files()}
    missing = [name for name in names if name not in by_name]
    if missing:
        raise FileNotFoundError("Missing session files: " + ", ".join(missing))
    return [by_name[name] for name in names]


def _load_model():
    feature_order = json.loads(FEATURES_PATH.read_text(encoding="utf-8"))
    scaler = joblib.load(SCALER_PATH)
    if len(feature_order) != 32 or scaler.n_features_in_ != len(feature_order):
        raise ValueError("Saved scaler/features do not match the 32-feature model input.")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=True)
    if checkpoint["feature_order"] != feature_order:
        raise ValueError("Feature order differs between checkpoint and JSON.")
    if checkpoint["sequence_length"] != SEQUENCE_LENGTH:
        raise ValueError("Expected the saved 12-state context length.")
    saved_threshold = float(checkpoint["validation_attack_score_threshold"])
    if not np.isclose(saved_threshold, CURRENT_THRESHOLD, rtol=0.0, atol=5e-9):
        raise ValueError("Saved validation threshold does not match 0.39090657.")

    model = MultitaskNetworkStateLSTM(
        input_size=checkpoint["input_size"],
        hidden_size=checkpoint["hidden_size"],
        num_layers=checkpoint["num_layers"],
        output_size=checkpoint["output_size"],
    ).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, scaler, feature_order


def _score_split(model, scaler, feature_order, files, split_name):
    rows = []
    for file_path in files:
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
                _, logits = model(torch.as_tensor(X, dtype=torch.float32, device=DEVICE))
                probabilities = torch.sigmoid(logits).cpu().numpy()
            targets = segment.iloc[SEQUENCE_LENGTH:].reset_index(drop=True)
            if len(targets) != len(probabilities):
                raise RuntimeError(f"{split_name}: scores and targets are misaligned.")
            for index, (probability, target) in enumerate(
                zip(probabilities, targets.to_dict("records"))
            ):
                if int(target["total_flows"]) == 0:
                    continue
                rows.append({
                    "split": split_name,
                    "session": file_path.name,
                    "segment_id": int(segment_id),
                    "sequence_index": index,
                    "target_timestamp": target["flow_start_time"],
                    "attack_ratio": float(target["attack_ratio"]),
                    "ground_truth_attack": int(target["is_attack"]),
                    "attack_probability": float(probability),
                })
    return pd.DataFrame(rows)


def _metrics(labels, probabilities, threshold):
    predictions = probabilities >= threshold
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "fpr": float(fp / (fp + tn)) if fp + tn else 0.0,
        "predicted_attack_count": int(predictions.sum()),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def _threshold_for_target_recall(labels, probabilities, target_recall):
    # Include a below-minimum cutoff (all positive) and above-maximum cutoff
    # (all negative), then choose the closest attained validation recall.
    candidates = np.unique(np.concatenate((
        [np.nextafter(probabilities.min(), -np.inf)],
        probabilities,
        [np.nextafter(probabilities.max(), np.inf)],
    )))
    scored = []
    for threshold in candidates:
        metrics = _metrics(labels, probabilities, threshold)
        scored.append((
            abs(metrics["recall"] - target_recall),
            -metrics["precision"],
            -threshold,
            float(threshold),
        ))
    return min(scored)[-1]


def main():
    for artifact in (CHECKPOINT_PATH, SCALER_PATH, FEATURES_PATH):
        if not artifact.exists():
            raise FileNotFoundError(artifact)
    model, scaler, feature_order = _load_model()

    validation = _score_split(
        model, scaler, feature_order, _load_files(VALIDATION_NAMES), "validation"
    )
    test = _score_split(
        model, scaler, feature_order, _load_files(TEST_NAMES), "test evaluation"
    )
    if len(validation) != 134 or len(test) != 218:
        raise RuntimeError(
            f"Unexpected aligned window counts: validation={len(validation)}, test={len(test)}"
        )

    validation_labels = validation["ground_truth_attack"].to_numpy(dtype=np.int8)
    validation_scores = validation["attack_probability"].to_numpy(dtype=float)
    test_labels = test["ground_truth_attack"].to_numpy(dtype=np.int8)
    test_scores = test["attack_probability"].to_numpy(dtype=float)

    target_thresholds = {
        target: _threshold_for_target_recall(
            validation_labels, validation_scores, target
        )
        for target in TARGET_RECALLS
    }
    thresholds = sorted(set(
        [CURRENT_THRESHOLD, *validation_scores.tolist(), *target_thresholds.values()]
    ))

    sweep_rows = []
    for threshold in thresholds:
        for split, labels, scores in (
            ("validation", validation_labels, validation_scores),
            ("test evaluation", test_labels, test_scores),
        ):
            sweep_rows.append({"split": split, **_metrics(labels, scores, threshold)})
    sweep = pd.DataFrame(sweep_rows)

    lines = [
        "# Saved multitask Delta-LSTM threshold sensitivity",
        "",
        "No training was performed. The saved checkpoint, scaler, and feature order were "
        "loaded without modification. The current threshold is 0.39090657.",
        "",
        "Threshold choices and approximate recall operating points below are derived from "
        "validation scores only. Test labels are used only after cutoffs are frozen, to "
        "calculate test evaluation metrics.",
        "",
        f"Validation: {len(validation)} windows ({int((validation_labels == 0).sum())} benign, "
        f"{int((validation_labels == 1).sum())} attack). Test: {len(test)} windows "
        f"({int((test_labels == 0).sum())} benign, {int((test_labels == 1).sum())} attack).",
        "",
        "## Validation threshold sweep",
        "",
        "The complete set of attainable score cutoffs from unique validation probabilities "
        "plus the current threshold is saved in the CSV. This summary includes the current "
        "threshold and the threshold nearest each requested recall target.",
        "",
        "| Operating point | Threshold | Precision | Recall | F1 | FPR | Predicted attacks |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    named_thresholds = [("Current", CURRENT_THRESHOLD)] + [
        (f"Nearest {target:.0%} recall", threshold)
        for target, threshold in target_thresholds.items()
    ]
    for name, threshold in named_thresholds:
        result = _metrics(validation_labels, validation_scores, threshold)
        lines.append(
            f"| {name} | {threshold:.10g} | {result['precision']:.4f} "
            f"| {result['recall']:.4f} | {result['f1']:.4f} | {result['fpr']:.4f} "
            f"| {result['predicted_attack_count']} |"
        )

    lines.extend([
        "",
        "## Test evaluation using fixed validation-derived thresholds",
        "",
        "These cutoffs were fixed before evaluating test labels. Test metrics did not affect "
        "threshold selection.",
        "",
        "| Operating point | Fixed threshold | Precision | Recall | F1 | FPR | Predicted attacks |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for name, threshold in named_thresholds:
        result = _metrics(test_labels, test_scores, threshold)
        lines.append(
            f"| {name} | {threshold:.10g} | {result['precision']:.4f} "
            f"| {result['recall']:.4f} | {result['f1']:.4f} | {result['fpr']:.4f} "
            f"| {result['predicted_attack_count']} |"
        )

    lines.extend([
        "",
        "The CSV contains each validation-derived cutoff evaluated on both splits, along "
        "with confusion-matrix counts. The test split is labeled `test evaluation`; no "
        "threshold was selected from its results.",
    ])
    SWEEP_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    sweep.to_csv(SWEEP_OUTPUT, index=False)
    REPORT_OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nFull sweep CSV: {SWEEP_OUTPUT}")
    print(f"Report: {REPORT_OUTPUT}")


if __name__ == "__main__":
    main()
