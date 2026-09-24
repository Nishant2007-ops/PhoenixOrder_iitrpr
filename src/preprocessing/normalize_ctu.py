from pathlib import Path
import pandas as pd


CTU_DIR = Path.home() / "datasets" / "CTU-13" / "flows"


CTU_COLUMNS = {
    "StartTime": "flow_start_time",
    "Dur": "flow_duration",
    "Proto": "protocol",
    "SrcAddr": "source_address",
    "Sport": "source_port",
    "Dir": "traffic_direction",
    "DstAddr": "destination_address",
    "Dport": "destination_port",
    "State": "connection_state",
    "TotPkts": "total_packets",
    "TotBytes": "total_bytes",
    "SrcBytes": "forward_bytes",
    "Label": "label",
}


def find_ctu_files():
    return sorted(CTU_DIR.rglob("*.binetflow"))


def load_chunk(file_path, chunk_size=100_000):
    return pd.read_csv(
        file_path,
        sep=",",
        usecols=list(CTU_COLUMNS.keys()),
        chunksize=chunk_size,
        low_memory=False,
    )


def normalize_chunk(chunk):
    chunk = chunk.rename(columns=CTU_COLUMNS)

    chunk["flow_start_time"] = pd.to_datetime(
        chunk["flow_start_time"],
        errors="coerce"
    )

    numeric_columns = [
        "flow_duration",
        "source_port",
        "destination_port",
        "total_packets",
        "total_bytes",
        "forward_bytes",
    ]

    for column in numeric_columns:
        chunk[column] = pd.to_numeric(
            chunk[column],
            errors="coerce"
        )

    chunk["backward_bytes"] = (
        chunk["total_bytes"]
        - chunk["forward_bytes"]
    )

    chunk["flow_duration"] = pd.to_numeric(
        chunk["flow_duration"],
        errors="coerce"
    )

    chunk["bytes_per_second"] = (
        chunk["total_bytes"]
        / chunk["flow_duration"].replace(0, pd.NA)
    )

    chunk["packets_per_second"] = (
        chunk["total_packets"]
        / chunk["flow_duration"].replace(0, pd.NA)
    )

    chunk["bytes_per_packet"] = (
        chunk["total_bytes"]
        / chunk["total_packets"].replace(0, pd.NA)
    )

    chunk["packet_length_mean"] = (
        chunk["total_bytes"]
        / chunk["total_packets"].replace(0, pd.NA)
    )

    chunk["start_hour"] = (
        chunk["flow_start_time"].dt.hour
    )

    chunk["start_day_of_week"] = (
        chunk["flow_start_time"].dt.dayofweek
    )

    chunk["down_up_ratio"] = (
        chunk["backward_bytes"]
        / chunk["forward_bytes"].replace(0, pd.NA)
    )

    chunk = chunk.replace(
        [float("inf"), float("-inf")],
        pd.NA
    )

    return chunk


def main():
    files = find_ctu_files()

    if not files:
        raise FileNotFoundError(
            f"No CTU-13 .binetflow files found in {CTU_DIR}"
        )

    file_path = files[0]

    print(f"Testing file: {file_path.name}")

    reader = load_chunk(file_path)

    chunk = next(reader)

    normalized = normalize_chunk(chunk)

    print(f"Rows processed: {len(normalized)}")
    print(f"Features produced: {len(normalized.columns)}")

    print()
    print(normalized.head())

    print()
    print("Columns:")
    print(list(normalized.columns))


if __name__ == "__main__":
    main()