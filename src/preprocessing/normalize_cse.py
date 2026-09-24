from pathlib import Path
import pandas as pd


CSE_DIR = Path.home() / "datasets" / "CSE-CIC-IDS2018"


CSE_COLUMNS = {
    "Timestamp": "flow_start_time",
    "Flow Duration": "flow_duration",
    "Protocol": "protocol",
    "Src Port": "source_port",
    "Dst Port": "destination_port",
    "Tot Fwd Pkts": "forward_packets",
    "Tot Bwd Pkts": "backward_packets",
    "TotLen Fwd Pkts": "forward_bytes",
    "TotLen Bwd Pkts": "backward_bytes",
    "Flow Byts/s": "bytes_per_second",
    "Flow Pkts/s": "packets_per_second",
    "Fwd Pkts/s": "forward_packets_per_second",
    "Bwd Pkts/s": "backward_packets_per_second",
    "Pkt Len Mean": "packet_length_mean",
    "Pkt Len Std": "packet_length_std",
    "Fwd Pkt Len Mean": "forward_packet_length_mean",
    "Bwd Pkt Len Mean": "backward_packet_length_mean",
    "Flow IAT Mean": "flow_iat_mean",
    "Flow IAT Std": "flow_iat_std",
    "Fwd IAT Mean": "forward_iat_mean",
    "Bwd IAT Mean": "backward_iat_mean",
    "SYN Flag Cnt": "syn_count",
    "ACK Flag Cnt": "ack_count",
    "FIN Flag Cnt": "fin_count",
    "RST Flag Cnt": "rst_count",
    "PSH Flag Cnt": "psh_count",
    "URG Flag Cnt": "urg_count",
    "Down/Up Ratio": "down_up_ratio",
    "Active Mean": "active_mean",
    "Idle Mean": "idle_mean",
    "Init Fwd Win Byts": "initial_forward_window",
    "Init Bwd Win Byts": "initial_backward_window",
    "Fwd Header Len": "forward_header_length",
    "Bwd Header Len": "backward_header_length",
    "Label": "label",
}


def find_cse_files():
    return sorted(CSE_DIR.glob("*.csv"))


def load_chunk(file_path, chunk_size=100_000):
    columns = list(CSE_COLUMNS.keys())

    return pd.read_csv(
        file_path,
        usecols=lambda column: column in columns,
        chunksize=chunk_size,
        low_memory=False,
    )


def normalize_chunk(chunk):
    chunk = chunk.rename(columns=CSE_COLUMNS)

    chunk = chunk[
        chunk["label"].astype(str).str.strip().str.lower() != "label"
    ].copy()

    chunk["flow_start_time"] = pd.to_datetime(
        chunk["flow_start_time"],
        errors="coerce",
        dayfirst=True
    )

    valid_timestamps = chunk["flow_start_time"].dropna()

    if len(valid_timestamps) > 0:

        dominant_date = valid_timestamps.dt.date.mode().iloc[0]

        invalid_date_mask = (
            chunk["flow_start_time"].notna()
            & (
                chunk["flow_start_time"].dt.date
                != dominant_date
            )
        )

        removed = invalid_date_mask.sum()

        if removed > 0:
            chunk = chunk.loc[
                ~invalid_date_mask
            ].copy()

    chunk = chunk.dropna(
        subset=["flow_start_time"]
    )

    numeric_columns = [
        column
        for column in chunk.columns
        if column not in [
            "flow_start_time",
            "protocol",
            "label"
        ]
    ]

    for column in numeric_columns:
        chunk[column] = pd.to_numeric(
            chunk[column],
            errors="coerce"
        )

    chunk["flow_duration"] = (
        chunk["flow_duration"] / 1_000_000
    )

    chunk = chunk.replace(
        [float("inf"), float("-inf")],
        pd.NA
    )

    chunk = add_derived_features(chunk)

    return chunk


def add_derived_features(chunk):

    chunk["total_packets"] = (
        chunk["forward_packets"]
        + chunk["backward_packets"]
    )

    chunk["total_bytes"] = (
        chunk["forward_bytes"]
        + chunk["backward_bytes"]
    )

    chunk["bytes_per_packet"] = (
        chunk["total_bytes"]
        / chunk["total_packets"].replace(0, pd.NA)
    )

    chunk["source_byte_ratio"] = (
        chunk["forward_bytes"]
        / chunk["total_bytes"].replace(0, pd.NA)
    )

    chunk["forward_byte_ratio"] = (
        chunk["forward_bytes"]
        / chunk["total_bytes"].replace(0, pd.NA)
    )

    chunk["forward_packet_ratio"] = (
        chunk["forward_packets"]
        / chunk["total_packets"].replace(0, pd.NA)
    )

    chunk["start_hour"] = (
        chunk["flow_start_time"].dt.hour
    )

    chunk["start_day_of_week"] = (
        chunk["flow_start_time"].dt.dayofweek
    )

    return chunk


def validate_normalized_chunk(chunk):

    numeric_columns = chunk.select_dtypes(
        include="number"
    ).columns

    infinite_values = chunk[numeric_columns].isin(
        [float("inf"), float("-inf")]
    ).sum().sum()

    if infinite_values > 0:
        raise ValueError(
            f"Found {infinite_values} infinite values."
        )

    if chunk["flow_start_time"].isna().sum() > 0:
        print(
            "Warning: Some timestamps could not be parsed."
        )

    header_rows = (
        chunk["label"]
        .astype(str)
        .str.strip()
        .str.lower()
        == "label"
    ).sum()

    if header_rows > 0:
        raise ValueError(
            f"Found {header_rows} embedded header rows after normalization."
        )

    required_columns = [
        "flow_start_time",
        "flow_duration",
        "total_packets",
        "total_bytes",
        "bytes_per_packet",
        "start_hour",
        "start_day_of_week",
        "label",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in chunk.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    print("Validation passed.")


def main():

    files = find_cse_files()

    if not files:
        raise FileNotFoundError(
            f"No CSE-CIC-IDS2018 CSV files found in {CSE_DIR}"
        )

    file_path = files[0]

    print(
        f"Testing file: {file_path.name}"
    )

    reader = load_chunk(file_path)

    chunk = next(reader)

    normalized = normalize_chunk(chunk)

    validate_normalized_chunk(normalized)

    print(
        f"Rows processed: {len(normalized)}"
    )

    print(
        f"Features produced: {len(normalized.columns)}"
    )

    print()

    print(
        normalized.head()
    )

    print()

    print("Columns:")

    print(
        list(normalized.columns)
    )


if __name__ == "__main__":
    main()
