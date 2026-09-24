import numpy as np
from sklearn.preprocessing import StandardScaler

from src.temporal.create_sequences import create_sequences


def prepare_lstm_data(
    state_groups,
    sequence_length=12
):
    train_states = state_groups[0]
    validation_states = state_groups[1]
    test_states = state_groups[2]

    train_states = train_states.select_dtypes(
        include="number"
    ).copy()

    validation_states = validation_states.select_dtypes(
        include="number"
    ).copy()

    test_states = test_states.select_dtypes(
        include="number"
    ).copy()

    feature_columns = train_states.columns

    validation_states = validation_states.reindex(
        columns=feature_columns,
        fill_value=0
    )

    test_states = test_states.reindex(
        columns=feature_columns,
        fill_value=0
    )

    log_features = [
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

    for column in log_features:
        if column in feature_columns:
            train_states[column] = np.log1p(
                train_states[column].clip(lower=0)
            )

            validation_states[column] = np.log1p(
                validation_states[column].clip(lower=0)
            )

            test_states[column] = np.log1p(
                test_states[column].clip(lower=0)
            )

    scaler = StandardScaler()

    train_scaled = scaler.fit_transform(
        train_states
    )

    validation_scaled = scaler.transform(
        validation_states
    )

    test_scaled = scaler.transform(
        test_states
    )

    train_scaled = train_scaled.astype(
        np.float32
    )

    validation_scaled = validation_scaled.astype(
        np.float32
    )

    test_scaled = test_scaled.astype(
        np.float32
    )

    print("\nFeature count:", len(feature_columns))

    print("\nScaled data check:")

    for i, column in enumerate(feature_columns):
        train_max = np.abs(
            train_scaled[:, i]
        ).max()

        validation_max = np.abs(
            validation_scaled[:, i]
        ).max()

        test_max = np.abs(
            test_scaled[:, i]
        ).max()

        print(
            f"{column:35s} "
            f"train={train_max:10.2f} "
            f"validation={validation_max:10.2f} "
            f"test={test_max:10.2f}"
        )

    print(
        "\nTrain max:",
        np.abs(train_scaled).max()
    )

    print(
        "Validation max:",
        np.abs(validation_scaled).max()
    )

    print(
        "Test max:",
        np.abs(test_scaled).max()
    )

    X_train, y_train = create_sequences(
        train_scaled,
        sequence_length
    )

    X_validation, y_validation = create_sequences(
        validation_scaled,
        sequence_length
    )

    X_test, y_test = create_sequences(
        test_scaled,
        sequence_length
    )

    return (
        X_train,
        y_train,
        X_validation,
        y_validation,
        X_test,
        y_test,
    )

