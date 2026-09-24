from pathlib import Path
import csv
import yaml


SCHEMA_PATH = Path("config/feature_schema.yaml")
REPORT_DIR = Path("reports")
REPORT_PATH = REPORT_DIR / "feature_availability_report.csv"


def main():
    with open(SCHEMA_PATH, "r") as file:
        schema = yaml.safe_load(file)

    features = schema["features"]

    REPORT_DIR.mkdir(exist_ok=True)

    rows = []

    for feature in features:

        cse = feature["cse"]
        ctu = feature["ctu"]

        rows.append({
            "feature": feature["name"],
            "category": feature["category"],
            "type": feature["type"],
            "cse_source": cse.get("source") or "",
            "cse_availability": cse["availability"],
            "ctu_source": ctu.get("source") or "",
            "ctu_availability": ctu["availability"],
        })

    with open(
        REPORT_PATH,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "feature",
                "category",
                "type",
                "cse_source",
                "cse_availability",
                "ctu_source",
                "ctu_availability",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)

    print(f"Report created: {REPORT_PATH}")
    print(f"Features documented: {len(rows)}")


if __name__ == "__main__":
    main()

