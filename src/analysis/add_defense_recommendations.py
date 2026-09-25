import pandas as pd

from src.features.defense_recommender import (
    generate_defense_recommendations
)


INPUT_FILE = "stage_predictions.csv"
OUTPUT_FILE = "stage_predictions.csv"


def main():
    df = pd.read_csv(INPUT_FILE)

    recommendations = []

    for _, row in df.iterrows():

        evidence_text = row.get(
            "behavior_evidence",
            ""
        )

        if pd.isna(evidence_text):
            evidence = []
        elif evidence_text.strip():
            evidence = [
                item.strip()
                for item in evidence_text.split(";")
                if item.strip()
            ]
        else:
            evidence = []

        result = generate_defense_recommendations(
            evidence,
            stage=row.get("stage", "Unknown"),
            mitre_tactic=row.get(
                "mitre_tactic",
                "Unknown"
            )
        )

        recommendations.append(
            "; ".join(
                result["recommendations"]
            )
        )

    df["defense_recommendations"] = recommendations

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("==============================")
    print("DEFENSE RECOMMENDATIONS ADDED")
    print("==============================")
    print(f"Rows: {len(df)}")
    print(f"Saved: {OUTPUT_FILE}")

    print("\nSample:")
    print(
        df[
            [
                "current_timestamp",
                "behavior_score",
                "behavior_evidence",
                "defense_recommendations"
            ]
        ]
        .head(5)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
