
import numpy as np
from sklearn.preprocessing import StandardScaler

from src.temporal.create_sequences import (
    create_delta_sequences
)


LOG_FEATURES = [
    "flow_duration_mean",
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
    "active_mean_mean",
    "idle_mean_mean",
    "flow_count",
]


def prepare_session_states(
    states,
    feature_columns=None
):

    states = states.select_dtypes(
        include="number"
    ).drop(
        columns=[
            "segment_id",
            "is_attack",
            "attack_ratio",
            "attack_flows",
            "total_flows",
            "dominant_label",
        ],
        errors="ignore"
    ).copy()

    if feature_columns is not None:

        states = states.reindex(
            columns=feature_columns,
            fill_value=0
        )

    for column in LOG_FEATURES:

        if column in states.columns:

            states[column] = np.log1p(
                states[column].clip(lower=0)
            )

    return states


def create_sequences_for_segment(
    states,
    scaler,
    feature_columns,
    sequence_length=12
):

    prepared = prepare_session_states(
        states,
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
        sequence_length
    )

    return X, y


def create_sequences_for_sessions(
    sessions,
    scaler,
    feature_columns,
    sequence_length=12
):

    X_all = []
    y_all = []

    for states in sessions:

        # --------------------------------------------------
        # If segment_id exists, process every continuous
        # segment independently.
        # --------------------------------------------------

        if "segment_id" in states.columns:

            segments = states.groupby(
                "segment_id",
                sort=True
            )

            for segment_id, segment in segments:

                X, y = create_sequences_for_segment(
                    segment,
                    scaler,
                    feature_columns,
                    sequence_length
                )

                if len(X) > 0:

                    X_all.append(X)
                    y_all.append(y)

        else:

            # Fallback for datasets without segment_id.
            X, y = create_sequences_for_segment(
                states,
                scaler,
                feature_columns,
                sequence_length
            )

            if len(X) > 0:

                X_all.append(X)
                y_all.append(y)

    if not X_all:

        return (
            np.empty(
                (
                    0,
                    sequence_length,
                    len(feature_columns)
                ),
                dtype=np.float32
            ),
            np.empty(
                (
                    0,
                    len(feature_columns)
                ),
                dtype=np.float32
            )
        )

    return (
        np.concatenate(
            X_all,
            axis=0
        ),
        np.concatenate(
            y_all,
            axis=0
        )
    )


def prepare_multisession_delta_lstm_data(
    train_sessions,
    validation_sessions,
    test_sessions,
    sequence_length=12
):

    first_train = prepare_session_states(
        train_sessions[0]
    )

    feature_columns = first_train.columns

    train_prepared = [
        prepare_session_states(
            states,
            feature_columns
        )
        for states in train_sessions
    ]

    train_combined = np.concatenate(
        [
            states.to_numpy()
            for states in train_prepared
        ],
        axis=0
    )

    scaler = StandardScaler()

    scaler.fit(
        train_combined
    )

    X_train, y_train = (
        create_sequences_for_sessions(
            train_sessions,
            scaler,
            feature_columns,
            sequence_length
        )
    )

    X_validation, y_validation = (
        create_sequences_for_sessions(
            validation_sessions,
            scaler,
            feature_columns,
            sequence_length
        )
    )

    X_test, y_test = (
        create_sequences_for_sessions(
            test_sessions,
            scaler,
            feature_columns,
            sequence_length
        )
    )

    return (
        X_train,
        y_train,
        X_validation,
        y_validation,
        X_test,
        y_test,
        feature_columns,
        scaler
    )
