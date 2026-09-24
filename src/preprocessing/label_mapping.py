import pandas as pd


def normalize_cse_label(label):
    label = str(label).strip()

    if label.lower() == "benign":
        return "benign"

    return label.lower()


def normalize_ctu_label(label):
    label = str(label).strip()

    if label.startswith("flow="):
        label = label[5:]

    label_lower = label.lower()

    if "botnet" in label_lower:
        return "botnet"

    if "normal" in label_lower:
        return "normal"

    if "background" in label_lower:
        return "background"

    return "unknown"


def add_normalized_label(
    dataframe,
    dataset
):
    if "label" not in dataframe.columns:
        raise ValueError("Column 'label' not found.")

    if dataset == "cse":
        dataframe["normalized_label"] = (
            dataframe["label"]
            .apply(normalize_cse_label)
        )

    elif dataset == "ctu":
        dataframe["normalized_label"] = (
            dataframe["label"]
            .apply(normalize_ctu_label)
        )

    else:
        raise ValueError(
            f"Unknown dataset: {dataset}"
        )

    return dataframe

