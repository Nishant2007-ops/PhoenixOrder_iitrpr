"""Run the frozen multitask Delta-LSTM offline on one CSE-style CSV."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from src.features.attack_stage_mapper import analyze_transition
from src.features.behavioral_detector import (
    analyze_behavioral_transition,
    build_behavior_evidence,
)
from src.features.defense_recommender import generate_defense_recommendations
from src.features.transition_features import (
    build_transition_dataframe,
    get_significant_transitions,
)
from src.models.lstm_model import MultitaskNetworkStateLSTM
from src.preprocessing.normalize_cse import find_cse_files
from src.temporal.build_multisession_dataset import build_states_for_file
from src.temporal.create_sequences import create_delta_sequences
from src.temporal.prepare_delta_lstm_data import LOG_FEATURES, prepare_session_states


CHECKPOINT_PATH = Path("multitask_delta_lstm_checkpoint.pth")
SCALER_PATH = Path("multitask_delta_lstm_scaler.joblib")
FEATURES_PATH = Path("multitask_delta_lstm_features.json")
SEQUENCE_LENGTH = 12
BEHAVIOR_TRANSITION_THRESHOLD = 0.20
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def _load_artifacts(checkpoint_path, scaler_path, feature_path):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=True,
    )
    scaler = joblib.load(scaler_path)
    feature_order = json.loads(feature_path.read_text(encoding="utf-8"))

    if checkpoint.get("model_class") != "MultitaskNetworkStateLSTM":
        raise ValueError("Checkpoint is not a multitask Delta-LSTM checkpoint.")
    if checkpoint["feature_order"] != feature_order:
        raise ValueError("Feature ordering does not match checkpoint metadata.")
    if len(feature_order) != 32 or checkpoint["input_size"] != 32:
        raise ValueError("The saved model must use the trained 32-feature representation.")
    if scaler.n_features_in_ != len(feature_order):
        raise ValueError("Saved scaler dimensionality does not match feature ordering.")
    if checkpoint["sequence_length"] != SEQUENCE_LENGTH:
        raise ValueError("The saved model must use a 12-state temporal context.")

    model = MultitaskNetworkStateLSTM(
        input_size=checkpoint["input_size"],
        hidden_size=checkpoint["hidden_size"],
        num_layers=checkpoint["num_layers"],
        output_size=checkpoint["output_size"],
    ).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, scaler, feature_order, float(checkpoint[
        "validation_attack_score_threshold"
    ])


def _inverse_state(scaled_state, scaler, feature_order):
    raw_state = scaler.inverse_transform(
        np.asarray(scaled_state, dtype=np.float64).reshape(1, -1)
    )[0]
    for index, feature in enumerate(feature_order):
        if feature in LOG_FEATURES:
            raw_state[index] = np.expm1(raw_state[index])
    return raw_state


def _json_value(value):
    return json.dumps(value, ensure_ascii=False, default=str)


def infer_file(input_path, output_path, checkpoint_path, scaler_path, feature_path):
    model, scaler, feature_order, threshold = _load_artifacts(
        checkpoint_path, scaler_path, feature_path
    )
    states = build_states_for_file(input_path)
    if states.empty:
        raise ValueError(f"No valid temporal states were built from {input_path}.")

    segments = (
        states.groupby("segment_id", sort=True)
        if "segment_id" in states.columns
        else [(0, states)]
    )
    output_rows = []
    for segment_id, segment in segments:
        segment = segment.sort_values("flow_start_time").reset_index(drop=True)
        if len(segment) <= SEQUENCE_LENGTH:
            continue

        prepared = prepare_session_states(segment, feature_order)
        scaled = scaler.transform(prepared.to_numpy()).astype(np.float32)
        sequences, _ = create_delta_sequences(scaled, SEQUENCE_LENGTH)
        if not len(sequences):
            continue

        with torch.no_grad():
            sequence_tensor = torch.as_tensor(
                sequences,
                dtype=torch.float32,
                device=DEVICE,
            )
            predicted_delta, attack_logits = model(sequence_tensor)
            attack_probabilities = torch.sigmoid(attack_logits).cpu().numpy()
            predicted_states = (
                sequence_tensor[:, -1, :] + predicted_delta
            ).cpu().numpy()
            current_states = sequence_tensor[:, -1, :].cpu().numpy()

        for sequence_index, (probability, current_scaled, predicted_scaled) in enumerate(
            zip(attack_probabilities, current_states, predicted_states)
        ):
            current_index = sequence_index + SEQUENCE_LENGTH - 1
            next_index = sequence_index + SEQUENCE_LENGTH
            current_timestamp = segment.iloc[current_index]["flow_start_time"]
            next_timestamp = segment.iloc[next_index]["flow_start_time"]

            current_real = _inverse_state(current_scaled, scaler, feature_order)
            predicted_real = _inverse_state(predicted_scaled, scaler, feature_order)
            behavior = analyze_behavioral_transition(
                current_real,
                predicted_real,
                pd.Index(feature_order),
            )
            transition = build_transition_dataframe(
                current_real,
                predicted_real,
                feature_order,
            )
            significant = get_significant_transitions(
                transition,
                threshold=BEHAVIOR_TRANSITION_THRESHOLD,
            )
            behavior_evidence = build_behavior_evidence(
                behavior,
                feature_order,
            )
            stage_result = analyze_transition(transition)
            stage = stage_result["stage_analysis"]
            mitre = stage_result["mitre_mapping"]
            defense = generate_defense_recommendations(
                behavior_evidence,
                stage=stage["stage"],
                mitre_tactic=mitre["tactic"] or "Unknown",
            )

            output_rows.append({
                "source_file": input_path.name,
                "segment_id": int(segment_id),
                "sequence_index": sequence_index,
                "current_timestamp": current_timestamp,
                "timestamp": next_timestamp,
                "attack_probability": float(probability),
                "classification_threshold": threshold,
                "predicted_class": "attack" if probability >= threshold else "benign",
                "behavioral_change_score": behavior["behavior_score"],
                "significant_behavior_feature_count": behavior[
                    "significant_feature_count"
                ],
                "behavioral_change_evidence": _json_value(behavior_evidence),
                "significant_feature_changes": significant.to_json(
                    orient="records", date_format="iso"
                ),
                "attack_stage": stage["stage"],
                "stage_score": stage["score"],
                "stage_confidence": stage["confidence"],
                "stage_evidence": _json_value(stage["evidence"]),
                "mitre_tactic_id": mitre["tactic_id"],
                "mitre_tactic": mitre["tactic"],
                "mitre_techniques": _json_value(mitre["techniques"]),
                "defense_recommendations": _json_value(
                    defense["recommendations"]
                ),
            })

    if not output_rows:
        raise ValueError(
            f"No valid predictions: the input needs at least {SEQUENCE_LENGTH + 1} "
            "states within a continuous segment."
        )
    output = pd.DataFrame(output_rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    return output


def main():
    parser = argparse.ArgumentParser(
        description="Offline inference with the frozen PhoenixOrder multitask Delta-LSTM."
    )
    parser.add_argument("input_csv", type=Path, help="CSE-CIC-IDS2018-style CSV file")
    parser.add_argument(
        "--output",
        type=Path,
        help="Output CSV path (default: <input_stem>_offline_predictions.csv)",
    )
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_PATH)
    parser.add_argument("--scaler", type=Path, default=SCALER_PATH)
    parser.add_argument("--features", type=Path, default=FEATURES_PATH)
    args = parser.parse_args()
    if not args.input_csv.is_file():
        parser.error(f"Input CSV does not exist: {args.input_csv}")
    for artifact in (args.checkpoint, args.scaler, args.features):
        if not artifact.is_file():
            parser.error(f"Saved inference artifact does not exist: {artifact}")

    output_path = args.output or args.input_csv.with_name(
        f"{args.input_csv.stem}_offline_predictions.csv"
    )
    result = infer_file(
        args.input_csv,
        output_path,
        args.checkpoint,
        args.scaler,
        args.features,
    )
    print(f"Valid predictions: {len(result)}")
    print(f"Output: {output_path}")
    print("Scores are separated: attack_probability is the multitask classifier output; "
          "behavioral_change_score summarizes predicted feature changes; attack_stage "
          "is the rule-based transition-stage classification.")


if __name__ == "__main__":
    main()
