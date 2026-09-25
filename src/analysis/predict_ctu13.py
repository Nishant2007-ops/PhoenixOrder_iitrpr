import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

from src.preprocessing.normalize_cse import find_cse_files
from src.preprocessing.normalize_ctu import find_ctu_files
from src.temporal.build_multisession_dataset import build_multisession_states
from src.temporal.build_ctu_states import build_ctu_states
from src.temporal.prepare_delta_lstm_data import (
    LOG_FEATURES,
    prepare_session_states,
)
from src.temporal.create_sequences import create_delta_sequences
from src.models.lstm_model import NetworkStateLSTM
from src.features.behavioral_detector import (
    analyze_behavioral_transition,
    build_behavior_evidence,
)
from src.features.defense_recommender import (
    generate_defense_recommendations,
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


FEATURE_COLUMNS = pd.Index([
    "flow_duration_mean",
    "flow_duration_std",
    "total_packets_sum",
    "total_packets_mean",
    "total_bytes_sum",
    "total_bytes_mean",
    "forward_packets_sum",
    "forward_packets_mean",
    "backward_packets_sum",
    "backward_packets_mean",
    "forward_bytes_sum",
    "forward_bytes_mean",
    "backward_bytes_sum",
    "backward_bytes_mean",
    "bytes_per_packet_mean",
    "packet_length_mean_mean",
    "packet_length_mean_std",
    "flow_iat_mean_mean",
    "flow_iat_std_mean",
    "forward_iat_mean_mean",
    "backward_iat_mean_mean",
    "syn_count_sum",
    "ack_count_sum",
    "fin_count_sum",
    "rst_count_sum",
    "psh_count_sum",
    "urg_count_sum",
    "down_up_ratio_mean",
    "forward_packet_ratio_mean",
    "active_mean_mean",
    "idle_mean_mean",
    "flow_count",
])


def recreate_cse_scaler():

    print("Recreating CSE training scaler...")

    files = find_cse_files()

    file_map = {
        file.name: file
        for file in files
    }

    train_files = [
        file_map[name]
        for name in TRAIN_NAMES
    ]

    train_sessions, _, _ = build_multisession_states(
        train_files,
        [],
        []
    )

    prepared_sessions = []

    for states in train_sessions:

        prepared = prepare_session_states(
            states,
            FEATURE_COLUMNS
        )

        prepared_sessions.append(
            prepared
        )

    combined = np.concatenate(
        [
            states.to_numpy()
            for states in prepared_sessions
        ],
        axis=0
    )

    scaler = StandardScaler()

    scaler.fit(combined)

    print(
        "Scaler recreated using",
        combined.shape[0],
        "CSE training states."
    )

    return scaler


def inverse_transform_state(
    scaled_state,
    scaler
):

    scaled_state = np.asarray(
        scaled_state,
        dtype=np.float64
    ).reshape(1, -1)

    transformed = scaler.inverse_transform(
        scaled_state
    )[0]

    real_state = transformed.copy()

    for i, feature in enumerate(
        FEATURE_COLUMNS
    ):

        if feature in LOG_FEATURES:

            real_state[i] = np.expm1(
                transformed[i]
            )

    return real_state


def prepare_ctu_states(states):

    prepared = states.reindex(
        columns=FEATURE_COLUMNS,
        fill_value=0
    ).copy()

    for column in LOG_FEATURES:

        if column in prepared.columns:

            prepared[column] = np.log1p(
                prepared[column].clip(
                    lower=0
                )
            )

    prepared = prepared.replace(
        [np.inf, -np.inf],
        0
    ).fillna(0)

    return prepared


def main():

    print("==============================")
    print("CTU-13 EXTERNAL PREDICTION")
    print("==============================")

    print("Device:", DEVICE)

    scaler = recreate_cse_scaler()

    print()
    print("Loading model...")

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

    ctu_files = find_ctu_files()

    print(
        "CTU files found:",
        len(ctu_files)
    )

    all_predictions = []

    for file_path in ctu_files:

        print()
        print(
            "Processing:",
            file_path.name
        )

        states = build_ctu_states(
            file_path
        )

        if states.empty:
            print("No states. Skipping.")
            continue

        segments = states.groupby(
            "segment_id",
            sort=True
        )

        for segment_id, segment in segments:

            segment = segment.sort_values(
                "flow_start_time"
            ).reset_index(drop=True)

            if len(segment) <= SEQUENCE_LENGTH:

                print(
                    f"  Segment {segment_id}: "
                    f"not enough states."
                )

                continue

            prepared = prepare_ctu_states(
                segment
            )

            scaled = scaler.transform(
                prepared
            ).astype(
                np.float32
            )

            X, y = create_delta_sequences(
                scaled,
                SEQUENCE_LENGTH
            )

            if len(X) == 0:
                continue

            X_tensor = torch.tensor(
                X,
                dtype=torch.float32
            ).to(DEVICE)

            with torch.no_grad():

                predicted_delta = model(
                    X_tensor
                )

            last_state = X_tensor[:, -1, :]

            predicted_state = (
                last_state
                + predicted_delta
            )

            for i in range(
                len(predicted_state)
            ):

                current_index = (
                    i
                    + SEQUENCE_LENGTH
                    - 1
                )

                next_index = (
                    i
                    + SEQUENCE_LENGTH
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

                current_real = (
                    inverse_transform_state(
                        current_scaled,
                        scaler
                    )
                )

                predicted_real = (
                    inverse_transform_state(
                        predicted_scaled,
                        scaler
                    )
                )

                behavioral_analysis = (
                    analyze_behavioral_transition(
                        current_real,
                        predicted_real,
                        FEATURE_COLUMNS
                    )
                )

                behavioral_evidence = (
                    build_behavior_evidence(
                        behavioral_analysis,
                        FEATURE_COLUMNS
                    )
                )

                defense_result = (
                    generate_defense_recommendations(
                        behavioral_evidence
                    )
                )

                defense_recommendations = (
                    defense_result["recommendations"]
                )

                all_predictions.append({

                    "dataset": "CTU-13",

                    "session": file_path.name,

                    "segment_id": segment_id,

                    "sequence_index": i,

                    "current_timestamp":
                        current_timestamp,

                    "next_timestamp":
                        next_timestamp,

                    "behavior_score":
                        behavioral_analysis[
                            "behavior_score"
                        ],

                    "significant_feature_count":
                        behavioral_analysis[
                            "significant_feature_count"
                        ],

                    "behavior_evidence":
                        "; ".join(
                            behavioral_evidence
                        ),

                    "defense_recommendations":
                        "; ".join(
                            defense_recommendations
                        ),

                    "stage":
                        "Unclassified",

                    "confidence":
                        "N/A",

                    "evaluation_note":
                        "External CTU-13 "
                        "demonstration using "
                        "the CSE-trained model."
                })

    if not all_predictions:

        print()
        print(
            "No predictions generated."
        )
        return

    output = pd.DataFrame(
        all_predictions
    )

    output.to_csv(
        "ctu13_predictions.csv",
        index=False
    )

    print()
    print("==============================")
    print("CTU-13 COMPLETE")
    print("==============================")

    print(
        "Predictions:",
        len(output)
    )

    print(
        "Sessions:",
        output["session"].nunique()
    )

    print(
        "Output:",
        "ctu13_predictions.csv"
    )

    print()
    print(
        output[
            [
                "session",
                "current_timestamp",
                "next_timestamp",
                "behavior_score",
                "significant_feature_count"
            ]
        ].head(10)
    )


if __name__ == "__main__":
    main()
