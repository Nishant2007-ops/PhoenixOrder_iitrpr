from pathlib import Path

import pandas as pd

from src.preprocessing.normalize_ctu import (
    find_ctu_files,
    load_chunk,
    normalize_chunk,
)
from src.temporal.aggregate_states import (
    AGGREGATIONS,
    WINDOW,
)


GAP_THRESHOLD_MINUTES = 10


CTU_STATE_COLUMNS = [
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
    "bytes_per_second_mean",
    "packets_per_second_mean",
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
    "forward_byte_ratio_mean",
    "forward_packet_ratio_mean",
    "active_mean_mean",
    "idle_mean_mean",
    "flow_count",
]


def add_ctu_derived_features(data):
    data = data.copy()

    data["forward_packets"] = (
        data["total_packets"]
        * (
            data["forward_bytes"]
            / data["total_bytes"].replace(0, pd.NA)
        )
    )

    data["backward_packets"] = (
        data["total_packets"]
        - data["forward_packets"]
    )

    data["forward_byte_ratio"] = (
        data["forward_bytes"]
        / data["total_bytes"].replace(0, pd.NA)
    )

    data["forward_packet_ratio"] = (
        data["forward_packets"]
        / data["total_packets"].replace(0, pd.NA)
    )

    data["flow_iat_mean"] = (
        data["flow_start_time"]
        .sort_values()
        .diff()
        .dt.total_seconds()
    )

    data["flow_iat_std"] = data["flow_iat_mean"]

    data["forward_iat_mean"] = data["flow_iat_mean"]

    data["backward_iat_mean"] = data["flow_iat_mean"]

    data["syn_count"] = 0
    data["ack_count"] = 0
    data["fin_count"] = 0
    data["rst_count"] = 0
    data["psh_count"] = 0
    data["urg_count"] = 0

    data["active_mean"] = data["flow_duration"]
    data["idle_mean"] = 0

    return data


def aggregate_ctu_segment(segment):
    data = add_ctu_derived_features(segment)

    states = data.set_index(
        "flow_start_time"
    ).resample(
        WINDOW
    ).agg(
        {
            "flow_duration": ["mean", "std"],
            "total_packets": ["sum", "mean"],
            "total_bytes": ["sum", "mean"],
            "forward_packets": ["sum", "mean"],
            "backward_packets": ["sum", "mean"],
            "forward_bytes": ["sum", "mean"],
            "backward_bytes": ["sum", "mean"],
            "bytes_per_second": ["mean"],
            "packets_per_second": ["mean"],
            "bytes_per_packet": ["mean"],
            "packet_length_mean": ["mean", "std"],
            "flow_iat_mean": ["mean"],
            "flow_iat_std": ["mean"],
            "forward_iat_mean": ["mean"],
            "backward_iat_mean": ["mean"],
            "syn_count": ["sum"],
            "ack_count": ["sum"],
            "fin_count": ["sum"],
            "rst_count": ["sum"],
            "psh_count": ["sum"],
            "urg_count": ["sum"],
            "down_up_ratio": ["mean"],
            "forward_byte_ratio": ["mean"],
            "forward_packet_ratio": ["mean"],
            "active_mean": ["mean"],
            "idle_mean": ["mean"],
        }
    )

    states.columns = [
        f"{column}_{aggregation}"
        for column, aggregation in states.columns
    ]

    flow_counts = (
        data.set_index("flow_start_time")
        .resample(WINDOW)
        .size()
        .rename("flow_count")
    )

    states = states.join(flow_counts)

    states = states.reset_index()

    return states


def build_ctu_states(file_path):
    print(f"Processing: {file_path.name}")

    chunks = []

    for chunk in load_chunk(file_path):
        normalized = normalize_chunk(chunk)

        normalized = normalized.dropna(
            subset=["flow_start_time"]
        )

        if len(normalized) > 0:
            chunks.append(normalized)

    if not chunks:
        return pd.DataFrame()

    data = pd.concat(
        chunks,
        ignore_index=True
    )

    data = data.sort_values(
        "flow_start_time"
    ).reset_index(drop=True)

    gaps = (
        data["flow_start_time"]
        .diff()
        .dt.total_seconds()
        .div(60)
    )

    data["segment_id"] = (
        gaps.fillna(0)
        .gt(GAP_THRESHOLD_MINUTES)
        .cumsum()
    )

    all_states = []

    for segment_id, segment in data.groupby(
        "segment_id",
        sort=True
    ):
        states = aggregate_ctu_segment(segment)

        states["segment_id"] = segment_id

        all_states.append(states)

    if not all_states:
        return pd.DataFrame()

    states = pd.concat(
        all_states,
        ignore_index=True
    )

    states = states.sort_values(
        "flow_start_time"
    ).reset_index(drop=True)

    numeric_columns = states.select_dtypes(
        include="number"
    ).columns

    states[numeric_columns] = (
        states[numeric_columns]
        .replace(
            [float("inf"), float("-inf")],
            0
        )
        .fillna(0)
    )

    return states


def build_all_ctu_states():
    files = find_ctu_files()

    if not files:
        raise FileNotFoundError(
            "No CTU-13 .binetflow files found."
        )

    all_states = []

    for file_path in files:
        states = build_ctu_states(file_path)

        if not states.empty:
            states["source_file"] = file_path.name
            all_states.append(states)

    if not all_states:
        return pd.DataFrame()

    return pd.concat(
        all_states,
        ignore_index=True
    )


if __name__ == "__main__":
    states = build_all_ctu_states()

    print()
    print("CTU-13 STATES")
    print("==============================")
    print("States:", len(states))
    print(
        "Segments:",
        states["segment_id"].nunique()
    )
    print(
        "Files:",
        states["source_file"].nunique()
    )
    print()
    print("Columns:")
    print(list(states.columns))
