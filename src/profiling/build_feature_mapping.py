from pathlib import Path
import pandas as pd


CSE_PATH = Path.home() / "datasets" / "CSE-CIC-IDS2018"
CTU_PATH = Path.home() / "datasets" / "CTU-13" / "flows"


def get_cse_schemas():
    schemas = {}

    for file in sorted(CSE_PATH.glob("*.csv")):
        df = pd.read_csv(file, nrows=0)
        schemas[file.name] = set(df.columns)

    return schemas


def get_ctu_schema():
    file = next(CTU_PATH.rglob("*.binetflow"))
    df = pd.read_csv(file, nrows=0)

    return set(df.columns)


def main():

    cse_schemas = get_cse_schemas()
    ctu = get_ctu_schema()

    cse_all = set.intersection(*cse_schemas.values())
    cse_any = set.union(*cse_schemas.values())

    print("\n=== FEATURE AVAILABILITY ===\n")

    mappings = {
        "flow_duration": ("Flow Duration", "Dur"),
        "protocol": ("Protocol", "Proto"),
        "destination_port": ("Dst Port", "Dport"),
        "source_port": ("Src Port", "Sport"),

        "total_forward_packets": ("Tot Fwd Pkts", None),
        "total_backward_packets": ("Tot Bwd Pkts", None),
        "total_forward_bytes": ("TotLen Fwd Pkts", "SrcBytes"),
        "total_backward_bytes": ("TotLen Bwd Pkts", None),

        "total_packets": (None, "TotPkts"),
        "total_bytes": (None, "TotBytes"),

        "timestamp": ("Timestamp", "StartTime"),

        "source_address": ("Src IP", "SrcAddr"),
        "destination_address": ("Dst IP", "DstAddr"),

        "direction": (None, "Dir"),
        "connection_state": (None, "State"),
        "source_tos": (None, "sTos"),
        "destination_tos": (None, "dTos"),

        "flow_bytes_per_second": ("Flow Byts/s", None),
        "flow_packets_per_second": ("Flow Pkts/s", None),

        "fwd_packet_length_mean": ("Fwd Pkt Len Mean", None),
        "bwd_packet_length_mean": ("Bwd Pkt Len Mean", None),
        "fwd_packet_length_max": ("Fwd Pkt Len Max", None),
        "bwd_packet_length_max": ("Bwd Pkt Len Max", None),
        "fwd_packet_length_min": ("Fwd Pkt Len Min", None),
        "bwd_packet_length_min": ("Bwd Pkt Len Min", None),

        "packet_length_mean": ("Pkt Len Mean", None),
        "packet_length_std": ("Pkt Len Std", None),

        "flow_iat_mean": ("Flow IAT Mean", None),
        "flow_iat_std": ("Flow IAT Std", None),

        "fwd_iat_mean": ("Fwd IAT Mean", None),
        "bwd_iat_mean": ("Bwd IAT Mean", None),

        "syn_count": ("SYN Flag Cnt", None),
        "ack_count": ("ACK Flag Cnt", None),
        "fin_count": ("FIN Flag Cnt", None),
        "rst_count": ("RST Flag Cnt", None),
        "psh_count": ("PSH Flag Cnt", None),
        "urg_count": ("URG Flag Cnt", None),

        "down_up_ratio": ("Down/Up Ratio", None),

        "active_mean": ("Active Mean", None),
        "idle_mean": ("Idle Mean", None),

        "initial_fwd_window": ("Init Fwd Win Byts", None),
        "initial_bwd_window": ("Init Bwd Win Byts", None),

        "fwd_header_length": ("Fwd Header Len", None),
        "bwd_header_length": ("Bwd Header Len", None),
    }

    for name, (cse_column, ctu_column) in mappings.items():

        cse_all_available = (
            cse_column is not None
            and cse_column in cse_all
        )

        cse_some_available = (
            cse_column is not None
            and cse_column in cse_any
        )

        ctu_available = (
            ctu_column is not None
            and ctu_column in ctu
        )

        if cse_all_available and ctu_available:
            status = "DIRECT_BOTH"

        elif cse_all_available:
            status = "CSE_ALL"

        elif cse_some_available and ctu_available:
            status = "CSE_SOME + CTU"

        elif cse_some_available:
            status = "CSE_SOME"

        elif ctu_available:
            status = "CTU_ONLY"

        else:
            status = "UNAVAILABLE"

        print(
            f"{name:28} "
            f"CSE_ALL={str(cse_all_available):5} "
            f"CSE_SOME={str(cse_some_available):5} "
            f"CTU={str(ctu_available):5} "
            f"{status}"
        )


if __name__ == "__main__":
    main()

