"""Train a shared-encoder Delta-LSTM with an auxiliary attack head.

This runner reads only the fixed training and validation CSV sessions. It does
not open or inspect any test session.
"""

import argparse
import json
from copy import deepcopy
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import recall_score

from src.models.lstm_model import MultitaskNetworkStateLSTM
from src.preprocessing.normalize_cse import find_cse_files
from src.temporal.build_multisession_dataset import build_multisession_states
from src.temporal.prepare_delta_lstm_data import (
    prepare_multisession_delta_lstm_data,
)


SEQUENCE_LENGTH = 12
HIDDEN_SIZE = 64
NUM_LAYERS = 2
EPOCHS = 30
BATCH_SIZE = 32
LEARNING_RATE = 0.001
CLASSIFICATION_LOSS_WEIGHT = 1.0
MIN_VALIDATION_RECALL_FOR_THRESHOLD = 0.80
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

CHECKPOINT_PATH = Path("multitask_delta_lstm_checkpoint.pth")
SCALER_PATH = Path("multitask_delta_lstm_scaler.joblib")
FEATURES_PATH = Path("multitask_delta_lstm_features.json")

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


def _files_for_train_and_validation():
    file_map = {path.name: path for path in find_cse_files()}
    requested = TRAIN_NAMES + VALIDATION_NAMES
    missing = sorted(set(requested) - set(file_map))
    if missing:
        raise FileNotFoundError("Missing training/validation files: " + ", ".join(missing))
    return (
        [file_map[name] for name in TRAIN_NAMES],
        [file_map[name] for name in VALIDATION_NAMES],
    )


def _segments(states):
    if "segment_id" in states.columns:
        return [segment.sort_values("flow_start_time").reset_index(drop=True)
                for _, segment in states.groupby("segment_id", sort=True)]
    return [states.sort_values("flow_start_time").reset_index(drop=True)]


def aligned_attack_targets(sessions, sequence_length=SEQUENCE_LENGTH):
    """Return next-state labels/masks in the same segment order as delta sequences."""
    labels, observed = [], []
    for session in sessions:
        for segment in _segments(session):
            labels.append(
                segment["is_attack"].to_numpy(dtype=np.float32)[sequence_length:]
            )
            observed.append(
                segment["total_flows"].to_numpy(dtype=np.int64)[sequence_length:] > 0
            )
    if not labels:
        return np.empty(0, dtype=np.float32), np.empty(0, dtype=bool)
    return np.concatenate(labels), np.concatenate(observed)


def select_validation_score_threshold(
    labels,
    probabilities,
    minimum_recall=MIN_VALIDATION_RECALL_FOR_THRESHOLD,
):
    """Choose the strictest validation cutoff that satisfies a recall floor."""
    candidates = np.unique(np.concatenate((
        [np.nextafter(np.min(probabilities), -np.inf)], probabilities
    )))
    eligible = [
        threshold for threshold in candidates
        if recall_score(labels, probabilities >= threshold, zero_division=0)
        >= minimum_recall
    ]
    if not eligible:
        raise ValueError("No validation threshold meets the requested recall floor.")
    return float(max(eligible))


def _classification_loss(logits, labels, observed_mask, criterion):
    if not torch.any(observed_mask):
        return logits.sum() * 0.0
    return criterion(logits[observed_mask], labels[observed_mask]).mean()


def _run_sanity_check():
    """Small synthetic alignment, architecture, loss, and gradient check."""
    from datetime import datetime, timedelta
    import pandas as pd

    synthetic_states = pd.DataFrame({
        "flow_start_time": [datetime(2024, 1, 1) + timedelta(minutes=5 * i)
                            for i in range(15)],
        "segment_id": [0] * 15,
        "is_attack": [0] * 12 + [1, 0, 1],
        "total_flows": [1] * 15,
    })
    labels, observed = aligned_attack_targets([synthetic_states])
    assert labels.tolist() == [1.0, 0.0, 1.0]
    assert observed.all()

    model = MultitaskNetworkStateLSTM(32, HIDDEN_SIZE, NUM_LAYERS, 32)
    x = torch.randn(4, SEQUENCE_LENGTH, 32)
    delta_target = torch.randn(4, 32)
    attack_target = torch.tensor([0.0, 1.0, 0.0, 1.0])
    prediction, logits = model(x)
    assert prediction.shape == (4, 32)
    assert logits.shape == (4,)
    loss = nn.MSELoss()(prediction, delta_target)
    loss = loss + nn.BCEWithLogitsLoss()(logits, attack_target)
    assert torch.isfinite(loss)
    loss.backward()
    assert model.delta_head.weight.grad is not None
    assert model.attack_head.weight.grad is not None
    print("Multitask Delta-LSTM sanity check passed (synthetic data only).")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sanity-check",
        action="store_true",
        help="Run only the synthetic model/alignment check; do not load dataset files.",
    )
    args = parser.parse_args()
    if args.sanity_check:
        _run_sanity_check()
        return

    torch.manual_seed(42)
    np.random.seed(42)
    train_files, validation_files = _files_for_train_and_validation()
    train_sessions, validation_sessions, _ = build_multisession_states(
        train_files, validation_files, []
    )

    (X_train, y_delta_train, X_validation, y_delta_validation,
     _, _, feature_columns, scaler) = prepare_multisession_delta_lstm_data(
        train_sessions,
        validation_sessions,
        [],
        sequence_length=SEQUENCE_LENGTH,
    )
    y_attack_train, observed_train = aligned_attack_targets(train_sessions)
    y_attack_validation, observed_validation = aligned_attack_targets(validation_sessions)

    if len(feature_columns) != 32:
        raise ValueError(
            f"Expected the existing 32-feature representation, got {len(feature_columns)}."
        )
    if not (len(X_train) == len(y_attack_train)
            and len(X_validation) == len(y_attack_validation)):
        raise RuntimeError("Attack labels are not aligned to next-state sequence targets.")
    if not observed_train.any() or not observed_validation.any():
        raise ValueError("Training and validation require observed next-state labels.")
    y_train_observed = y_attack_train[observed_train]
    positives = int(np.sum(y_train_observed == 1))
    negatives = int(np.sum(y_train_observed == 0))
    if positives == 0 or negatives == 0:
        raise ValueError("Training labels must contain both attack and benign windows.")

    pos_weight_value = negatives / positives
    print(f"Training class counts: benign={negatives}, attack={positives}; "
          f"BCE pos_weight={pos_weight_value:.6f}")

    model = MultitaskNetworkStateLSTM(
        input_size=len(feature_columns),
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        output_size=len(feature_columns),
    ).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    delta_criterion = nn.MSELoss()
    classification_criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(pos_weight_value, dtype=torch.float32, device=DEVICE),
        reduction="none",
    )

    train_x = torch.as_tensor(X_train, dtype=torch.float32)
    train_delta = torch.as_tensor(y_delta_train, dtype=torch.float32)
    train_attack = torch.as_tensor(y_attack_train, dtype=torch.float32)
    train_observed = torch.as_tensor(observed_train, dtype=torch.bool)
    val_x = torch.as_tensor(X_validation, dtype=torch.float32, device=DEVICE)
    val_delta = torch.as_tensor(y_delta_validation, dtype=torch.float32, device=DEVICE)
    val_attack = torch.as_tensor(y_attack_validation, dtype=torch.float32, device=DEVICE)
    val_observed = torch.as_tensor(observed_validation, dtype=torch.bool, device=DEVICE)

    best_validation_loss = float("inf")
    best_epoch = None
    best_state = None
    for epoch in range(EPOCHS):
        model.train()
        permutation = torch.randperm(len(train_x))
        for start in range(0, len(permutation), BATCH_SIZE):
            indices = permutation[start:start + BATCH_SIZE]
            batch_x = train_x[indices].to(DEVICE)
            batch_delta = train_delta[indices].to(DEVICE)
            batch_attack = train_attack[indices].to(DEVICE)
            batch_observed = train_observed[indices].to(DEVICE)

            optimizer.zero_grad()
            predicted_delta, attack_logits = model(batch_x)
            delta_loss = delta_criterion(predicted_delta, batch_delta)
            attack_loss = _classification_loss(
                attack_logits, batch_attack, batch_observed, classification_criterion
            )
            total_loss = delta_loss + CLASSIFICATION_LOSS_WEIGHT * attack_loss
            total_loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_delta_prediction, val_attack_logits = model(val_x)
            validation_delta_loss = delta_criterion(val_delta_prediction, val_delta)
            validation_attack_loss = _classification_loss(
                val_attack_logits, val_attack, val_observed, classification_criterion
            )
            validation_loss = (
                validation_delta_loss
                + CLASSIFICATION_LOSS_WEIGHT * validation_attack_loss
            ).item()

        print(
            f"Epoch {epoch + 1:02d}/{EPOCHS}: "
            f"validation delta MSE={validation_delta_loss.item():.6f}, "
            f"weighted BCE={validation_attack_loss.item():.6f}, "
            f"combined={validation_loss:.6f}"
        )
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_epoch = epoch + 1
            best_state = deepcopy(model.state_dict())

    if best_state is None:
        raise RuntimeError("No validation-selected checkpoint was produced.")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        _, validation_logits = model(val_x)
        validation_probabilities = torch.sigmoid(validation_logits).cpu().numpy()
    score_threshold = select_validation_score_threshold(
        y_attack_validation[observed_validation],
        validation_probabilities[observed_validation],
    )

    feature_order = [str(name) for name in feature_columns]
    checkpoint = {
        "model_state_dict": {
            name: tensor.detach().cpu()
            for name, tensor in best_state.items()
        },
        "model_class": "MultitaskNetworkStateLSTM",
        "input_size": len(feature_order),
        "output_size": len(feature_order),
        "hidden_size": HIDDEN_SIZE,
        "num_layers": NUM_LAYERS,
        "sequence_length": SEQUENCE_LENGTH,
        "classification_loss_weight": CLASSIFICATION_LOSS_WEIGHT,
        "training_pos_weight": pos_weight_value,
        "best_epoch": best_epoch,
        "best_validation_combined_loss": best_validation_loss,
        "validation_attack_score_threshold": score_threshold,
        "threshold_selection_rule": "highest validation threshold with recall >= 0.80",
        "feature_order": feature_order,
        "training_files": TRAIN_NAMES,
        "validation_files": VALIDATION_NAMES,
    }
    torch.save(checkpoint, CHECKPOINT_PATH)
    joblib.dump(scaler, SCALER_PATH)
    FEATURES_PATH.write_text(json.dumps(feature_order, indent=2) + "\n", encoding="utf-8")
    print(f"Saved exact selected weights and metadata: {CHECKPOINT_PATH}")
    print(f"Saved training-fitted scaler: {SCALER_PATH}")
    print(f"Saved feature ordering: {FEATURES_PATH}")
    print(f"Selected epoch: {best_epoch}; validation score threshold: {score_threshold:.8g}")


if __name__ == "__main__":
    main()
