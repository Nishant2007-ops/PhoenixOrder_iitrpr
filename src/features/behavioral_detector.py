import numpy as np
import pandas as pd


EPSILON = 1e-8

BEHAVIOR_FEATURES = [
    "flow_count",
    "total_packets_sum",
    "total_bytes_sum",
    "forward_packets_sum",
    "backward_packets_sum",
    "forward_bytes_sum",
    "backward_bytes_sum",
    "syn_count_sum",
    "ack_count_sum",
    "rst_count_sum",
    "fin_count_sum",
    "psh_count_sum",
]


def calculate_normalized_change(previous_state, current_state):
    previous_state = np.asarray(previous_state, dtype=float)
    current_state = np.asarray(current_state, dtype=float)

    denominator = np.maximum(
        np.abs(previous_state),
        np.abs(current_state)
    )

    return (
        np.abs(current_state - previous_state)
        / (denominator + EPSILON)
    )


def calculate_directional_change(previous_state, current_state):
    previous_state = np.asarray(previous_state, dtype=float)
    current_state = np.asarray(current_state, dtype=float)

    denominator = np.maximum(
        np.abs(previous_state),
        np.abs(current_state)
    )

    return (
        (current_state - previous_state)
        / (denominator + EPSILON)
    )


def analyze_behavioral_transition(
    previous_state,
    current_state,
    feature_names,
    significant_threshold=0.20,
    minimum_activity=10.0,
):
    previous_state = np.asarray(previous_state, dtype=float)
    current_state = np.asarray(current_state, dtype=float)

    feature_indices = [
        feature_names.get_loc(name)
        for name in BEHAVIOR_FEATURES
        if name in feature_names
    ]

    selected_names = [
        feature_names[i]
        for i in feature_indices
    ]

    previous_selected = previous_state[feature_indices]
    current_selected = current_state[feature_indices]

    normalized_change = calculate_normalized_change(
        previous_selected,
        current_selected
    )

    directional_change = calculate_directional_change(
        previous_selected,
        current_selected
    )

    activity = np.maximum(
        np.abs(previous_selected),
        np.abs(current_selected)
    )

    result = pd.DataFrame({
        "feature": selected_names,
        "previous_value": previous_selected,
        "current_value": current_selected,
        "activity": activity,
        "normalized_change": normalized_change,
        "directional_change": directional_change,
    })

    result["direction"] = np.where(
        result["directional_change"] > 0,
        "increase",
        np.where(
            result["directional_change"] < 0,
            "decrease",
            "stable"
        )
    )

    result["significant"] = (
        (result["normalized_change"] >= significant_threshold)
        & (result["activity"] >= minimum_activity)
    )

    behavior_score = result["normalized_change"].mean()

    significant_count = int(
        result["significant"].sum()
    )

    return {
        "behavior_score": float(behavior_score),
        "significant_feature_count": significant_count,
        "feature_changes": result,
    }


def build_behavior_evidence(
    analysis,
    feature_names=None,
):
    result = analysis["feature_changes"]

    evidence = []

    for _, row in result.iterrows():

        if not row["significant"]:
            continue

        percentage = (
            row["directional_change"] * 100
        )

        evidence.append(
            f"{row['feature']} "
            f"{row['direction']} "
            f"{abs(percentage):.1f}%"
        )

    return evidence
