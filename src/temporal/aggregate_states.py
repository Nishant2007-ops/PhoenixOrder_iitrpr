import pandas as pd


WINDOW = "5min"


AGGREGATIONS = {
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


def aggregate_states(dataframe):
    data = dataframe.copy()

    data["flow_start_time"] = pd.to_datetime(
        data["flow_start_time"],
        errors="coerce"
    )

    data = data.dropna(subset=["flow_start_time"])

    data = data.sort_values("flow_start_time")

    states = data.set_index("flow_start_time").resample(WINDOW).agg(
        AGGREGATIONS
    )

    # Keep flow label statistics aligned with each state window. attack_ratio
    # retains mixed benign/attack windows instead of hiding them behind a
    # dominant label. Datasets without labels remain usable by other callers.
    if "label" in data.columns:
        labels = data["label"].astype(str).str.strip()
        normalized_labels = labels.str.lower()
        window_keys = data["flow_start_time"].dt.floor(WINDOW)
        label_groups = labels.groupby(window_keys)
        benign_counts = normalized_labels.eq("benign").groupby(window_keys).sum()
        flow_counts_by_label = labels.groupby(window_keys).size()
        attack_counts = flow_counts_by_label - benign_counts
        label_summary = pd.DataFrame({
            "attack_flows": attack_counts,
            "total_flows": flow_counts_by_label,
            "attack_ratio": attack_counts / flow_counts_by_label,
            "dominant_label": label_groups.agg(
                lambda values: values.value_counts().index[0]
            ),
        })

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
    if "label" in data.columns:
        states = states.join(label_summary, how="left")
        states["attack_ratio"] = states["attack_ratio"].fillna(0.0)
        states["attack_flows"] = states["attack_flows"].fillna(0).astype(int)
        states["total_flows"] = states["total_flows"].fillna(0).astype(int)
        states["is_attack"] = states["attack_ratio"].gt(0).astype(int)

    states = states.reset_index()

    return states
