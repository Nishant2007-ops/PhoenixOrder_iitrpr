from pathlib import Path
import yaml


SCHEMA_PATH = Path("config/feature_schema.yaml")

VALID_AVAILABILITY = {
    "direct",
    "derived",
    "unavailable",
}


def main():
    with open(SCHEMA_PATH, "r") as file:
        schema = yaml.safe_load(file)

    features = schema.get("features", [])

    print(f"Feature count: {len(features)}")

    if len(features) != 44:
        raise ValueError(
            f"Expected 44 features, found {len(features)}"
        )

    names = [feature.get("name") for feature in features]

    if len(names) != len(set(names)):
        raise ValueError("Duplicate feature names found.")

    for feature in features:

        required_fields = {
            "name",
            "category",
            "type",
            "cse",
            "ctu",
        }

        missing = required_fields - feature.keys()

        if missing:
            raise ValueError(
                f"{feature.get('name', 'UNKNOWN')} "
                f"is missing: {missing}"
            )

        for dataset in ["cse", "ctu"]:

            availability = feature[dataset].get("availability")

            if availability not in VALID_AVAILABILITY:
                raise ValueError(
                    f"Invalid availability '{availability}' "
                    f"for {feature['name']} ({dataset})"
                )

    print("No duplicate feature names.")
    print("All required fields are present.")
    print("All availability values are valid.")
    print("Schema validation PASSED.")


if __name__ == "__main__":
    main()
