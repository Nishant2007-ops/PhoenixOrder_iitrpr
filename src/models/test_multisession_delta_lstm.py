
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

from src.preprocessing.normalize_cse import find_cse_files
from src.temporal.build_multisession_dataset import build_multisession_states
from src.temporal.prepare_delta_lstm_data import (
    prepare_multisession_delta_lstm_data,
    prepare_session_states,
    LOG_FEATURES
)
from src.temporal.create_sequences import create_delta_sequences
from src.models.lstm_model import NetworkStateLSTM
from src.features.transition_features import (
    build_transition_dataframe,
    get_significant_transitions
)
from src.features.attack_stage_mapper import (
    analyze_transition
)

from src.features.behavioral_detector import (
    analyze_behavioral_transition,
    build_behavior_evidence
)


DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

SEQUENCE_LENGTH = 12
INPUT_SIZE = 32
HIDDEN_SIZE = 64
NUM_LAYERS = 2
OUTPUT_SIZE = 32


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


def inverse_transform_state(
    scaled_state,
    scaler,
    feature_columns
):

    scaled_state = np.asarray(
        scaled_state,
        dtype=np.float64
    ).reshape(1, -1)

    transformed = scaler.inverse_transform(
        scaled_state
    )[0]

    real_state = transformed.copy()

    for i, feature in enumerate(feature_columns):

        if feature in LOG_FEATURES:

            real_state[i] = np.expm1(
                transformed[i]
            )

    return real_state


def evaluate_session(
    model,
    states,
    scaler,
    feature_columns,
    session_name
):

    model.eval()

    all_predicted = []
    all_actual = []
    all_persistence = []
    stage_rows = []

    if "segment_id" in states.columns:
        segments = states.groupby(
            "segment_id",
            sort=True
        )
    else:
        segments = [(0, states)]

    for segment_id, segment in segments:

        segment = segment.sort_values(
            "flow_start_time"
        ).reset_index(drop=True)

        prepared = prepare_session_states(
            segment,
            feature_columns
        )

        scaled = scaler.transform(
            prepared
        )

        scaled = scaled.astype(
            np.float32
        )

        X, y = create_delta_sequences(
            scaled,
            SEQUENCE_LENGTH
        )

        if len(X) == 0:
            continue

        X = torch.tensor(
            X,
            dtype=torch.float32
        ).to(DEVICE)

        y = torch.tensor(
            y,
            dtype=torch.float32
        ).to(DEVICE)

        with torch.no_grad():
            predicted_delta = model(X)

        last_state = X[:, -1, :]

        predicted_state = (
            last_state + predicted_delta
        )

        actual_next_state = (
            last_state + y
        )

        persistence_state = last_state

        all_predicted.append(predicted_state)
        all_actual.append(actual_next_state)
        all_persistence.append(persistence_state)

        for i in range(len(predicted_state)):

            current_index = (
                i + SEQUENCE_LENGTH - 1
            )

            next_index = (
                i + SEQUENCE_LENGTH
            )

            current_timestamp = (
                segment.iloc[
                    current_index
                ]["flow_start_time"]
            )

            next_timestamp = (
                segment.iloc[
                    next_index
                ]["flow_start_time"]
            )

            current_scaled = (
                last_state[i]
                .detach()
                .cpu()
                .numpy()
            )

            predicted_scaled = (
                predicted_state[i]
                .detach()
                .cpu()
                .numpy()
            )

            current_real = inverse_transform_state(
                current_scaled,
                scaler,
                feature_columns
            )

            predicted_real = inverse_transform_state(
                predicted_scaled,
                scaler,
                feature_columns
            )

            behavioral_analysis = analyze_behavioral_transition(
                current_real,
                predicted_real,
                feature_columns
            )

            behavioral_evidence = build_behavior_evidence(
                behavioral_analysis,
                feature_columns
            )

            transition_df = build_transition_dataframe(
                current_real,
                predicted_real,
                feature_columns
            )

            significant = get_significant_transitions(
                transition_df,
                threshold=0.20
            )

            analysis = analyze_transition(
                transition_df
            )

            stage_analysis = analysis[
                "stage_analysis"
            ]

            mitre_mapping = analysis[
                "mitre_mapping"
            ]

            evidence = "; ".join(
                stage_analysis["evidence"]
            )

            techniques = mitre_mapping[
                "techniques"
            ]

            technique_text = "; ".join(
                [
                    technique["id"]
                    + " - "
                    + technique["name"]
                    for technique in techniques
                ]
            )

            stage_rows.append({
                "session": session_name,
                "segment_id": segment_id,
                "sequence_index": i,
                "current_timestamp": current_timestamp,
                "next_timestamp": next_timestamp,
                "behavior_score": behavioral_analysis[
                    "behavior_score"
                ],
                "significant_feature_count": behavioral_analysis[
                    "significant_feature_count"
                ],
                "behavior_evidence": "; ".join(
                    behavioral_evidence
                ),
                "stage": stage_analysis["stage"],
                "score": stage_analysis["score"],
                "confidence": stage_analysis["confidence"],
                "evidence": evidence,
                "mitre_tactic_id": mitre_mapping[
                    "tactic_id"
                ],
                "mitre_tactic": mitre_mapping[
                    "tactic"
                ],
                "mitre_techniques": technique_text
            })

            if (
                session_name == TEST_NAMES[0]
                and segment_id == 0
                and i == 0
            ):

                transition_df.to_csv(
                    "first_transition_real_all.csv",
                    index=False
                )

                significant.to_csv(
                    "first_transition_real.csv",
                    index=False
                )

    if not all_predicted:

        empty_state = torch.empty(
            (
                0,
                len(feature_columns)
            ),
            dtype=torch.float32,
            device=DEVICE
        )

        return (
            pd.DataFrame(stage_rows),
            empty_state,
            empty_state.clone(),
            empty_state.clone()
        )

    predicted_state = torch.cat(
        all_predicted,
        dim=0
    )

    actual_next_state = torch.cat(
        all_actual,
        dim=0
    )

    persistence_state = torch.cat(
        all_persistence,
        dim=0
    )

    stage_dataframe = pd.DataFrame(
        stage_rows
    )

    return (
        stage_dataframe,
        predicted_state,
        actual_next_state,
        persistence_state
    )

def main():

    print("==============================")
    print("MULTI-SESSION DELTA-LSTM TEST")
    print("==============================")

    print(
        "Device:",
        DEVICE
    )

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

    model = NetworkStateLSTM(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        output_size=OUTPUT_SIZE
    ).to(DEVICE)

    model.load_state_dict(
        torch.load(
            "multisession_delta_lstm_model.pth",
            map_location=DEVICE
        )
    )

    model.eval()

    print("\n==============================")
    print("TEST RESULTS")
    print("==============================")

    all_predicted = []
    all_actual = []
    all_persistence = []
    all_stage_results = []

    for file, states in zip(
        test_files,
        test_sessions
    ):

        (
            stage_dataframe,
            predicted_state,
            actual_next_state,
            persistence_state
        ) = evaluate_session(
            model,
            states,
            scaler,
            feature_columns,
            file.name
        )

        all_predicted.append(
            predicted_state
        )

        all_actual.append(
            actual_next_state
        )

        all_persistence.append(
            persistence_state
        )

        all_stage_results.append(
            stage_dataframe
        )

        mse = nn.MSELoss()
        mae = nn.L1Loss()

        model_mse = mse(
            predicted_state,
            actual_next_state
        ).item()

        model_mae = mae(
            predicted_state,
            actual_next_state
        ).item()

        persistence_mse = mse(
            persistence_state,
            actual_next_state
        ).item()

        persistence_mae = mae(
            persistence_state,
            actual_next_state
        ).item()

        improvement = (
            (
                persistence_mse
                - model_mse
            )
            / persistence_mse
        ) * 100

        print("\nSession:", file.name)

        print(
            "Delta-LSTM MSE:",
            f"{model_mse:.6f}"
        )

        print(
            "Delta-LSTM MAE:",
            f"{model_mae:.6f}"
        )

        print(
            "Persistence MSE:",
            f"{persistence_mse:.6f}"
        )

        print(
            "Persistence MAE:",
            f"{persistence_mae:.6f}"
        )

        print(
            "MSE Improvement:",
            f"{improvement:.2f}%"
        )

    all_predicted = torch.cat(
        all_predicted,
        dim=0
    )

    all_actual = torch.cat(
        all_actual,
        dim=0
    )

    all_persistence = torch.cat(
        all_persistence,
        dim=0
    )

    mse = nn.MSELoss()
    mae = nn.L1Loss()

    overall_model_mse = mse(
        all_predicted,
        all_actual
    ).item()

    overall_model_mae = mae(
        all_predicted,
        all_actual
    ).item()

    overall_persistence_mse = mse(
        all_persistence,
        all_actual
    ).item()

    overall_persistence_mae = mae(
        all_persistence,
        all_actual
    ).item()

    overall_improvement = (
        (
            overall_persistence_mse
            - overall_model_mse
        )
        / overall_persistence_mse
    ) * 100

    print("\n==============================")
    print("OVERALL TEST RESULTS")
    print("==============================")

    print(
        "Delta-LSTM MSE:",
        f"{overall_model_mse:.6f}"
    )

    print(
        "Delta-LSTM MAE:",
        f"{overall_model_mae:.6f}"
    )

    print(
        "Persistence MSE:",
        f"{overall_persistence_mse:.6f}"
    )

    print(
        "Persistence MAE:",
        f"{overall_persistence_mae:.6f}"
    )

    print(
        "MSE Improvement:",
        f"{overall_improvement:.2f}%"
    )

    stage_predictions = pd.concat(
        all_stage_results,
        ignore_index=True
    )

    stage_predictions.to_csv(
        "stage_predictions.csv",
        index=False
    )

    print("\n==============================")
    print("ATTACK STAGE SUMMARY")
    print("==============================")

    print(
        stage_predictions[
            "stage"
        ].value_counts()
    )

    print("\n==============================")
    print("STAGE PREDICTIONS FILE")
    print("==============================")

    print(
        "Saved: stage_predictions.csv"
    )


if __name__ == "__main__":
    main()

