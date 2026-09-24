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

