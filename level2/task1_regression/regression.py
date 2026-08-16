"""Level 2 - Task 1: Simple Linear Regression."""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent
RAW_FILE = BASE_DIR / "data" / "raw" / "house_prediction_data_set.csv"
PROCESSED_FILE = BASE_DIR / "data" / "processed" / "house_prediction_data_set.csv"
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

    X = df[["RM"]]
    y = df["MEDV"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    print("Intercept:", model.intercept_)
    print("Coefficient:", model.coef_[0])
    print(f"R-squared: {r2:.4f}")
    print(f"Mean Squared Error: {mse:.4f}")

    comparison = pd.DataFrame({
        "actual_medv": y_test.to_numpy(),
        "predicted_medv": y_pred,
    })
    comparison.to_csv(OUTPUT_DIR / "actual_vs_predicted.csv", index=False)

    plt.figure(figsize=(8, 6))
    plt.scatter(df["RM"], df["MEDV"], alpha=0.6)
    plt.xlabel("RM")
    plt.ylabel("MEDV")
    plt.title("RM vs MEDV")
    plt.savefig(PLOT_DIR / "rm_vs_medv.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.scatter(X_test["RM"], y_test, label="Actual Prices")
    sorted_idx = X_test["RM"].argsort()
    plt.plot(
        X_test["RM"].iloc[sorted_idx],
        y_pred[sorted_idx],
        label="Regression Line",
    )
    plt.xlabel("RM")
    plt.ylabel("House Price (MEDV)")
    plt.title("Simple Linear Regression")
    plt.legend()
    plt.savefig(PLOT_DIR / "simple_linear_regression.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred)
    plt.xlabel("Actual Price (MEDV)")
    plt.ylabel("Predicted Price")
    plt.title("Actual vs Predicted House Prices")
    plt.savefig(PLOT_DIR / "actual_vs_predicted.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Processed data: {PROCESSED_FILE}")
    print(f"Outputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
