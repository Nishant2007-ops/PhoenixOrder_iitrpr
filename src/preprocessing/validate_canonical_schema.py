from pathlib import Path
import yaml


SCHEMA_PATH = Path("config/feature_schema.yaml")


def load_schema():
    with open(SCHEMA_PATH, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def validate_schema():
    schema = load_schema()
    features = schema["features"]

    print(f"Schema features: {len(features)}")

    cse_direct = 0
    cse_derived = 0
    cse_unavailable = 0

    ctu_direct = 0
    ctu_derived = 0
    ctu_unavailable = 0

    for feature in features:
        name = feature["name"]

        cse_status = feature["cse"]["availability"]
        ctu_status = feature["ctu"]["availability"]

        if cse_status == "direct":
            cse_direct += 1
        elif cse_status == "derived":
            cse_derived += 1
        elif cse_status == "unavailable":
            cse_unavailable += 1
        else:
            raise ValueError(
                f"Invalid CSE availability for {name}: {cse_status}"
            )

        if ctu_status == "direct":
            ctu_direct += 1
        elif ctu_status == "derived":
            ctu_derived += 1
        elif ctu_status == "unavailable":
            ctu_unavailable += 1
        else:
            raise ValueError(
                f"Invalid CTU availability for {name}: {ctu_status}"
            )

    print()
    print("CSE:")
    print(f"  Direct:      {cse_direct}")
    print(f"  Derived:     {cse_derived}")
    print(f"  Unavailable: {cse_unavailable}")

    print()
    print("CTU:")
    print(f"  Direct:      {ctu_direct}")
    print(f"  Derived:     {ctu_derived}")
    print(f"  Unavailable: {ctu_unavailable}")

    total_cse = (
        cse_direct
        + cse_derived
        + cse_unavailable
    )

    total_ctu = (
        ctu_direct
        + ctu_derived
        + ctu_unavailable
    )

    if total_cse != len(features):
        raise ValueError("CSE feature accounting does not match.")

    if total_ctu != len(features):
        raise ValueError("CTU feature accounting does not match.")

    print()
    print("Canonical schema validation PASSED.")


if __name__ == "__main__":
    validate_schema()
