import pandas as pd


CSE_CANONICAL_COLUMNS = [
    "flow_start_time",
    "flow_duration",
    "protocol",
    "source_port",
    "destination_port",
    "forward_packets",
    "backward_packets",
    "forward_bytes",
    "backward_bytes",
    "total_packets",
    "total_bytes",
    "bytes_per_second",
    "packets_per_second",
    "bytes_per_packet",
    "packet_length_mean",
    "packet_length_std",
    "forward_packet_length_mean",
    "backward_packet_length_mean",
    "flow_iat_mean",
    "flow_iat_std",
    "forward_iat_mean",
    "backward_iat_mean",
    "syn_count",
    "ack_count",
    "fin_count",
    "rst_count",
    "psh_count",
    "urg_count",
    "down_up_ratio",
    "forward_byte_ratio",
    "forward_packet_ratio",
    "start_hour",
    "start_day_of_week",
    "active_mean",
    "idle_mean",
    "initial_forward_window",
    "initial_backward_window",
    "forward_header_length",
    "backward_header_length",
    "label",
]


CTU_CANONICAL_COLUMNS = [
    "flow_start_time",
    "flow_duration",
    "protocol",
    "source_address",
    "source_port",
    "traffic_direction",
    "destination_address",
    "destination_port",
    "connection_state",
    "total_packets",
    "total_bytes",
    "forward_bytes",
    "backward_bytes",
    "bytes_per_second",
    "packets_per_second",
    "bytes_per_packet",
    "packet_length_mean",
    "start_hour",
    "start_day_of_week",
    "down_up_ratio",
    "label",
]

def canonicalize_cse(dataframe):
    result = dataframe.copy()

    for column in CSE_CANONICAL_COLUMNS:
        if column not in result.columns:
            result[column] = pd.NA

    return result[CSE_CANONICAL_COLUMNS].copy()


def canonicalize_ctu(dataframe):
    missing = [
        column
        for column in CTU_CANONICAL_COLUMNS
        if column not in dataframe.columns
    ]

    if missing:
        raise ValueError(
            f"CTU missing canonical columns: {missing}"
        )

    return dataframe[CTU_CANONICAL_COLUMNS].copy()

