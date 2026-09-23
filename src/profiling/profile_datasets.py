from pathlib import Path
import pandas as pd


CSE_PATH = Path.home() / "datasets" / "CSE-CIC-IDS2018"
CTU_PATH = Path.home() / "datasets" / "CTU-13" / "flows"


def profile_cse():
    print("\n=== CSE-CIC-IDS2018 ===")

    files = sorted(CSE_PATH.glob("*.csv"))

    print(f"Files found: {len(files)}")

    schemas = {}

    for file in files:
        df = pd.read_csv(file, nrows=0)
        schemas[file.name] = list(df.columns)

    reference_file = files[0]
    reference_schema = schemas[reference_file.name]

    print(f"\nReference file: {reference_file.name}")
    print(f"Reference columns: {len(reference_schema)}")

    for filename, columns in schemas.items():

        extra = [column for column in columns if column not in reference_schema]
        missing = [column for column in reference_schema if column not in columns]

        print(f"\n{filename}")
        print(f"Columns: {len(columns)}")

        if extra:
            print("Extra columns:")
            for column in extra:
                print(f"  + {column}")

        if missing:
            print("Missing columns:")
            for column in missing:
                print(f"  - {column}")

        if not extra and not missing:
            print("Schema matches reference.")


def profile_ctu():
    print("\n=== CTU-13 ===")

    files = sorted(CTU_PATH.rglob("*.binetflow"))

    print(f"Files found: {len(files)}")

    schemas = {}

    for file in files:
        df = pd.read_csv(file, sep=",", nrows=0)
        schemas[file.name] = list(df.columns)

    reference_file = files[0]
    reference_schema = schemas[reference_file.name]

    print(f"\nReference file: {reference_file.name}")
    print(f"Reference columns: {len(reference_schema)}")

    all_same = True

    for filename, columns in schemas.items():

        extra = [column for column in columns if column not in reference_schema]
        missing = [column for column in reference_schema if column not in columns]

        print(f"\n{filename}")
        print(f"Columns: {len(columns)}")

        if extra:
            all_same = False
            print("Extra columns:")
            for column in extra:
                print(f"  + {column}")

        if missing:
            all_same = False
            print("Missing columns:")
            for column in missing:
                print(f"  - {column}")

        if not extra and not missing:
            print("Schema matches reference.")

    print(f"\nAll CTU schemas identical: {all_same}")


if __name__ == "__main__":
    profile_cse()
    profile_ctu()

    