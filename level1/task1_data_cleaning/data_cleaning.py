"""Level 1 - Task 1: Data Cleaning and Preprocessing."""

from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
RAW_FILE = BASE_DIR / "data" / "raw" / "iris.csv"
PROCESSED_FILE = BASE_DIR / "data" / "processed" / "iris.csv"
OUTPUT_FILE = BASE_DIR / "outputs" / "iris.csv"


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and standardize the Iris dataset."""
    df = df.copy()

    df.columns = [
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width",
        "species",
    ]

    numeric_columns = df.select_dtypes(include="number").columns
    categorical_columns = df.select_dtypes(exclude="number").columns

    if numeric_columns.any():
        df[numeric_columns] = df[numeric_columns].apply(
            lambda col: col.fillna(col.mean())
        )

    for column in categorical_columns:
        if df[column].isna().any():
            mode = df[column].mode()
            if not mode.empty:
                df[column] = df[column].fillna(mode.iloc[0])

    df = df.drop_duplicates().reset_index(drop=True)
    return df


def main() -> None:
    """Run the cleaning pipeline and save the results."""
    PROCESSED_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(RAW_FILE)
    print(f"Raw shape: {df.shape}")
    print(f"Missing values before cleaning:\n{df.isna().sum()}")
    print(f"Duplicate rows before cleaning: {df.duplicated().sum()}")

    cleaned = clean_data(df)

    print(f"\nCleaned shape: {cleaned.shape}")
    print(f"Missing values after cleaning:\n{cleaned.isna().sum()}")
    print(f"Duplicate rows after cleaning: {cleaned.duplicated().sum()}")
    print("\nSummary statistics:")
    print(cleaned.describe())

    cleaned.to_csv(PROCESSED_FILE, index=False)
    cleaned.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved processed data to: {PROCESSED_FILE}")
    print(f"Saved submission output to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
