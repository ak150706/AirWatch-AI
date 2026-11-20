
import pandas as pd

REQUIRED_COLUMNS = [
    "station_id",
    "pm25",
    "pm10",
    "no2",
    "temp",
    "hum",
    "label"
]

FEATURES = [
    "pm25",
    "pm10",
    "no2",
    "temp",
    "hum"
]


def prepare_data(path):

    print(f"\nLoading dataset: {path}")

    df = pd.read_csv(path)

    # Validate required columns
    missing = set(REQUIRED_COLUMNS) - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    print("Required columns validated")

    # Check missing values
    print("\nMissing values:")
    print(df[REQUIRED_COLUMNS].isnull().sum())

    # Remove exact duplicate records
    duplicate_count = df.duplicated().sum()

    print("\nExact duplicate records:", duplicate_count)

    df = df.drop_duplicates()

    # Check duplicate IDs separately
    print(
        "Duplicate station IDs:",
        df["station_id"].duplicated().sum()
    )

    # Validate target labels
    df = df.dropna(subset=["label"])

    if not set(df["label"].unique()).issubset({0, 1}):
        raise ValueError("Labels must be 0 or 1")

    # Dataset dimensions
    print("\nDataset size:", df.shape)

    # Class distribution
    print("\nTarget distribution:")
    print(df["label"].value_counts())

    # Check feature data types
    df[FEATURES] = df[FEATURES].apply(
        pd.to_numeric,
        errors="coerce"
    )

    return df
