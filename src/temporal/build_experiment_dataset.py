import pandas as pd

from src.preprocessing.normalize_cse import (
    load_chunk,
    normalize_chunk,
)

from src.temporal.aggregate_states import aggregate_states


def build_states_for_file(file_path):
    print(
        f"Processing: {file_path.name}"
    )

    all_chunks = []

    for chunk in load_chunk(file_path):
        normalized = normalize_chunk(chunk)
        all_chunks.append(normalized)

    data = pd.concat(
        all_chunks,
        ignore_index=True
    )

    states = aggregate_states(data)

    states = states.sort_values(
        "flow_start_time"
    ).reset_index(drop=True)

    numeric_columns = states.select_dtypes(
        include="number"
    ).columns

    states[numeric_columns] = states[
        numeric_columns
    ].fillna(0)

    print(
        f"States: {len(states)}"
    )

    return states


def build_experiment_states(file_paths):
    state_groups = []

    for file_path in file_paths:
        states = build_states_for_file(
            file_path
        )

        state_groups.append(states)

    return state_groups

