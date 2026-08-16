"""Level 2 - Task 2: K-Means Clustering."""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent
RAW_FILE = BASE_DIR / "data" / "raw" / "iris.csv"
PROCESSED_FILE = BASE_DIR / "data" / "processed" / "iris.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
PLOT_DIR = OUTPUT_DIR / "plots"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(RAW_FILE)
    df = df.drop_duplicates().reset_index(drop=True)

    numeric = df.select_dtypes(include="number").columns
    df[numeric] = df[numeric].apply(lambda col: col.fillna(col.mean()))
    df.to_csv(PROCESSED_FILE, index=False)

    features = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    X_scaled = StandardScaler().fit_transform(df[features])

    inertias = []
    for k in range(1, 11):
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        model.fit(X_scaled)
        inertias.append(model.inertia_)

    plt.figure(figsize=(7, 5))
    plt.plot(range(1, 11), inertias, marker="o")
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Inertia")
    plt.title("Elbow Method")
    plt.savefig(PLOT_DIR / "elbow_method.png", dpi=300, bbox_inches="tight")
    plt.close()

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df["cluster"] = kmeans.fit_predict(X_scaled)

    print("Cluster counts:")
    print(df["cluster"].value_counts().sort_index())
    print("\nCluster vs actual species:")
    print(pd.crosstab(df["species"], df["cluster"]))

    plt.figure(figsize=(8, 5))
    sns.scatterplot(
        data=df,
        x="sepal_length",
        y="sepal_width",
        hue="cluster",
        palette="deep",
        s=70,
    )
    plt.xlabel("Sepal Length")
    plt.ylabel("Sepal Width")
    plt.title("K-Means Clusters - Iris Dataset")
    plt.legend(title="Cluster")
    plt.savefig(PLOT_DIR / "kmeans_clusters_iris.png", dpi=300, bbox_inches="tight")
    plt.close()

    df.to_csv(OUTPUT_DIR / "iris_kmeans_clustered.csv", index=False)
    print(f"\nClustered data: {OUTPUT_DIR / 'iris_kmeans_clustered.csv'}")


if __name__ == "__main__":
    main()
