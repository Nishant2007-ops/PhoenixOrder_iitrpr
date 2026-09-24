
import numpy as np
import pandas as pd


EPSILON = 1e-8


def calculate_absolute_change(current_state, predicted_state):

    current_state = np.asarray(
        current_state,
        dtype=float
    )

    predicted_state = np.asarray(
        predicted_state,
        dtype=float
    )

    return predicted_state - current_state


def calculate_percentage_change(current_state, predicted_state):

    current_state = np.asarray(
        current_state,
        dtype=float
    )

    predicted_state = np.asarray(
        predicted_state,
        dtype=float
    )

    return (
        (predicted_state - current_state)
        / (np.abs(current_state) + EPSILON)
    ) * 100


def classify_change(change, threshold=0.05):

    change = np.asarray(
        change,
        dtype=float
    )

    result = np.empty(
        change.shape,
        dtype=object
    )

    result[change > threshold] = "increase"

    result[change < -threshold] = "decrease"

    result[np.abs(change) <= threshold] = "stable"

    return result


def build_transition_dataframe(
    current_state,
    predicted_state,
    feature_names,
    threshold=0.05
):

    absolute_change = calculate_absolute_change(
        current_state,
        predicted_state
    )

    percentage_change = calculate_percentage_change(
        current_state,
        predicted_state
    )

    normalized_change = (
        absolute_change
        / (
            np.abs(current_state)
            + EPSILON
        )
    )

    direction = classify_change(
        normalized_change,
        threshold
    )

    dataframe = pd.DataFrame({
        "feature": feature_names,
        "current_value": current_state,
        "predicted_value": predicted_state,
        "absolute_change": absolute_change,
        "percentage_change": percentage_change,
        "normalized_change": normalized_change,
        "direction": direction
    })

    dataframe["magnitude"] = np.abs(
        dataframe["normalized_change"]
    )

    return dataframe


def get_significant_transitions(
    transition_dataframe,
    threshold=0.20
):

    significant = transition_dataframe[
        transition_dataframe["magnitude"] >= threshold
    ].copy()

    significant = significant.sort_values(
        "magnitude",
        ascending=False
    )

    return significant.reset_index(
        drop=True
    )
