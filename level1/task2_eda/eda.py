"""Level 1 - Task 2: Exploratory Data Analysis."""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = Path(__file__).resolve().parent
RAW_FILE = BASE_DIR / "data" / "raw" / "iris.csv"
PROCESSED_FILE = BASE_DIR / "data" / "processed" / "iris.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
PLOT_DIR = OUTPUT_DIR / "plots"


def load_and_clean_data() -> pd.DataFrame:
    """Load the Iris data and apply light preprocessing for EDA."""
    df = pd.read_csv(RAW_FILE)
    df.columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]

    numeric = df.select_dtypes(include="number").columns
    df[numeric] = df[numeric].apply(lambda col: col.fillna(col.mean()))

    if df["species"].isna().any():
        df["species"] = df["species"].fillna(df["species"].mode().iloc[0])

    return df.drop_duplicates().reset_index(drop=True)


def save_plot(filename: str) -> None:
    """Save the current matplotlib figure."""
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()


def run_eda(df: pd.DataFrame) -> None:
    """Create requested descriptive statistics and visualizations."""
    print("Dataset shape:", df.shape)
    print("\nData types:\n", df.dtypes)
    print("\nMissing values:\n", df.isna().sum())
    print("\nSummary statistics:\n", df.describe())
    print("\nMean:\n", df.mean(numeric_only=True))
    print("\nMedian:\n", df.median(numeric_only=True))
    print("\nMode:\n", df.mode(numeric_only=True).iloc[0])
    print("\nStandard deviation:\n", df.std(numeric_only=True))
    print("\nSpecies counts:\n", df["species"].value_counts())
    print("\nCorrelation:\n", df.corr(numeric_only=True))

    sns.pairplot(df, hue="species")
    plt.suptitle("Iris Feature Pairplot", y=1.02)
    plt.savefig(PLOT_DIR / "pairplot.png", dpi=300, bbox_inches="tight")
    plt.close("all")

    df.hist(figsize=(10, 8), bins=15)
    plt.suptitle("Histogram of Iris Features")
    save_plot("histogram_of_iris_features.png")

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df.select_dtypes(include="number"))
    plt.title("Boxplots of Numerical Features")
    save_plot("boxplots_of_numerical_features.png")

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x="sepal_length", y="petal_width", hue="species")
    plt.title("Sepal Length vs Petal Width")
    save_plot("sepal_length_vs_petal_width.png")

    plt.figure(figsize=(8, 6))
    sns.heatmap(df.corr(numeric_only=True), annot=True, fmt=".2f")
    plt.title("Correlation Heatmap")
    save_plot("correlation_heatmap.png")

    plt.figure(figsize=(8, 6))
    ax = sns.countplot(data=df, x="species", hue="species", legend=False)
    for container in ax.containers:
        ax.bar_label(container)
    plt.title("Species Distribution")
    save_plot("species_distribution.png")

    top10 = df.nlargest(10, "sepal_length").reset_index()
    plt.figure(figsize=(9, 5))
    sns.barplot(data=top10, x="index", y="sepal_length")
    plt.title("Top 10 Sepal Lengths")
    plt.xlabel("Original Row Index")
    plt.ylabel("Sepal Length")
    save_plot("top_10_sepal_lengths.png")

    grouped = (
        df.groupby("species", as_index=False)["sepal_length"]
        .max()
        .rename(columns={"sepal_length": "max_sepal_length"})
    )
    plt.figure(figsize=(8, 5))
    sns.barplot(data=grouped, x="species", y="max_sepal_length", hue="species", legend=False)
    plt.title("Maximum Sepal Length by Species")
    plt.xlabel("Species")
    plt.ylabel("Maximum Sepal Length")
    save_plot("maximum_sepal_length_by_species.png")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_and_clean_data()
    df.to_csv(PROCESSED_FILE, index=False)
    run_eda(df)
    df.to_csv(OUTPUT_DIR / "iris.csv", index=False)

    print(f"\nProcessed data: {PROCESSED_FILE}")
    print(f"Plots: {PLOT_DIR}")


if __name__ == "__main__":
    main()
