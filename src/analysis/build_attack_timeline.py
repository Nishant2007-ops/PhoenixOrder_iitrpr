from pathlib import Path
import pandas as pd

CSE_DIR = Path.home() / "datasets" / "CSE-CIC-IDS2018"

TEST_FILES = [
    "Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv",
    "Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv",
]

WINDOW = "5min"


def build_timeline(file_path):

    print(f"\nProcessing: {file_path.name}")

    all_data = []

    for chunk in pd.read_csv(
        file_path,
        usecols=["Timestamp", "Label"],
        chunksize=100000,
        low_memory=False
    ):

        chunk = chunk[
            chunk["Label"].astype(str).str.strip().str.lower()
            != "label"
        ].copy()

        chunk["Timestamp"] = pd.to_datetime(
            chunk["Timestamp"],
            errors="coerce",
            dayfirst=True
        )

        chunk = chunk.dropna(subset=["Timestamp"])

        chunk["Label"] = (
            chunk["Label"]
            .astype(str)
            .str.strip()
        )

        all_data.append(chunk)

    data = pd.concat(
        all_data,
        ignore_index=True
    )

    data = data.sort_values("Timestamp")

    dominant_date = (
        data["Timestamp"]
        .dt.date
        .mode()
        .iloc[0]
    )

    data = data[
        data["Timestamp"].dt.date == dominant_date
    ].copy()

    time_difference = (
        data["Timestamp"]
        .diff()
        .dt.total_seconds()
        / 60
    )

    data["segment_id"] = (
        time_difference
        .fillna(0)
        .gt(10)
        .cumsum()
    )

    timelines = []

    for segment_id, segment in data.groupby(
        "segment_id",
        sort=True
    ):

        segment = segment.set_index("Timestamp")

        for timestamp, window in segment.groupby(
            pd.Grouper(freq=WINDOW)
        ):

            if len(window) == 0:
                continue

            counts = window["Label"].value_counts()

            dominant_label = counts.index[0]

            total_flows = len(window)

            benign_flows = counts.get(
                "Benign",
                0
            )

            attack_flows = (
                total_flows - benign_flows
            )

            attack_ratio = (
                attack_flows / total_flows
            )

            timelines.append({
                "timestamp": timestamp,
                "segment_id": segment_id,
                "total_flows": total_flows,
                "benign_flows": benign_flows,
                "attack_flows": attack_flows,
                "attack_ratio": attack_ratio,
                "dominant_label": dominant_label
            })

    return pd.DataFrame(timelines)


def main():

    output_dir = Path("reports")
    output_dir.mkdir(exist_ok=True)

    for filename in TEST_FILES:

        file_path = CSE_DIR / filename

        timeline = build_timeline(file_path)

        output_file = (
            output_dir /
            f"{file_path.stem}_attack_timeline.csv"
        )

        timeline.to_csv(
            output_file,
            index=False
        )

        print(f"\nSaved: {output_file}")

        print("\nLabel distribution:")
        print(
            timeline["dominant_label"]
            .value_counts()
        )

        print("\nAttack windows:")

        print(
            timeline[
                timeline["attack_flows"] > 0
            ][
                [
                    "timestamp",
                    "segment_id",
                    "attack_flows",
                    "attack_ratio",
                    "dominant_label"
                ]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()
