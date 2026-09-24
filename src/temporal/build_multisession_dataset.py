
import pandas as pd

from src.preprocessing.normalize_cse import (
    load_chunk,
    normalize_chunk
)

from src.temporal.aggregate_states import (
    aggregate_states
)


GAP_THRESHOLD_MINUTES = 10


def build_states_for_file(file_path):

    print(f"Processing: {file_path.name}")

    all_chunks = []

    for chunk in load_chunk(file_path):

        normalized = normalize_chunk(chunk)

        if len(normalized) > 0:
            all_chunks.append(normalized)

    if not all_chunks:
        print("No valid data found.")
        return pd.DataFrame()

    data = pd.concat(
        all_chunks,
        ignore_index=True
    )

    data = data.sort_values(
        "flow_start_time"
    ).reset_index(
        drop=True
    )

    # Detect gaps using the ORIGINAL flow timestamps
    time_difference = (
        data["flow_start_time"]
        .diff()
        .dt.total_seconds()
        / 60
    )

    data["segment_id"] = (
        time_difference
        .fillna(0)
        .gt(GAP_THRESHOLD_MINUTES)
        .cumsum()
    )

    all_states = []

    for segment_id, segment in data.groupby(
        "segment_id"
    ):

        states = aggregate_states(
            segment
        )

        states = states.sort_values(
            "flow_start_time"
        ).reset_index(
            drop=True
        )

        states["segment_id"] = segment_id

        all_states.append(
            states
        )

    states = pd.concat(
        all_states,
        ignore_index=True
    )

    states = states.sort_values(
        "flow_start_time"
    ).reset_index(
        drop=True
    )

    numeric_columns = states.select_dtypes(
        include="number"
    ).columns

    states[numeric_columns] = (
        states[numeric_columns]
        .fillna(0)
    )

    print(
        f"States: {len(states)}"
    )

    print(
        f"Segments: "
        f"{states['segment_id'].nunique()}"
    )

    print(
        "\nSegment summary:"
    )

    for segment_id, segment in states.groupby(
        "segment_id"
    ):

        print(
            f"  Segment {segment_id}: "
            f"{len(segment)} states | "
            f"{segment['flow_start_time'].min()} "
            f"→ "
            f"{segment['flow_start_time'].max()}"
        )

    return states


def build_multisession_states(
    train_files,
    validation_files,
    test_files
):

    train_states = []

    validation_states = []

    test_states = []

    for file_path in train_files:

        train_states.append(
            build_states_for_file(
                file_path
            )
        )

    for file_path in validation_files:

        validation_states.append(
            build_states_for_file(
                file_path
            )
        )

    for file_path in test_files:

        test_states.append(
            build_states_for_file(
                file_path
            )
        )

    return (
        train_states,
        validation_states,
        test_states
    )

