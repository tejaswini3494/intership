"""Level 3 - Task 1: Customer Churn Classification."""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

BASE_DIR = Path(__file__).resolve().parent
TRAIN_FILE = BASE_DIR / "data" / "raw" / "churn-bigml-80.csv"
TEST_FILE = BASE_DIR / "data" / "raw" / "churn-bigml-20.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
PLOT_DIR = OUTPUT_DIR / "plots"


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Build a reusable preprocessing pipeline."""
    categorical = X.select_dtypes(include=["object", "string"]).columns.tolist()
    numerical = X.select_dtypes(exclude=["object", "string"]).columns.tolist()

    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numerical,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            ),
        ]
    )


def evaluate(model, X_test, y_test):
    """Return standard classification metrics."""
    pred = model.predict(X_test)
    return {
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred, zero_division=0),
        "Recall": recall_score(y_test, pred, zero_division=0),
        "F1-score": f1_score(y_test, pred, zero_division=0),
    }, pred


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(TRAIN_FILE).drop_duplicates().reset_index(drop=True)
    test = pd.read_csv(TEST_FILE).drop_duplicates().reset_index(drop=True)

    X_train = train.drop(columns="Churn")
    X_test = test.drop(columns="Churn")
    y_train = train["Churn"].astype(int)
    y_test = test["Churn"].astype(int)

    preprocessor = build_preprocessor(X_train)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
    }

    grids = {
        "Logistic Regression": {
            "model__C": [0.1, 1, 10],
            "model__solver": ["liblinear"],
        },
        "Decision Tree": {
            "model__max_depth": [3, 5, 8, None],
            "model__min_samples_split": [2, 5, 10],
        },
        "Random Forest": {
            "model__n_estimators": [100, 200],
            "model__max_depth": [None, 10, 20],
            "model__min_samples_split": [2, 5],
        },
    }

    baseline_rows = []
    tuned_rows = []
    tuned_models = {}

    for name, model in models.items():
        pipeline = Pipeline([("prep", preprocessor), ("model", model)])
        pipeline.fit(X_train, y_train)

        metrics, _ = evaluate(pipeline, X_test, y_test)
        baseline_rows.append({"Model": name, **metrics})

        grid = GridSearchCV(
            pipeline,
            grids[name],
            cv=5,
            scoring="f1",
            n_jobs=-1,
        )
        grid.fit(X_train, y_train)

        metrics, _ = evaluate(grid.best_estimator_, X_test, y_test)
        tuned_rows.append({
            "Model": name,
            **metrics,
            "CV F1": grid.best_score_,
        })
        tuned_models[name] = grid

    baseline = pd.DataFrame(baseline_rows)
    tuned = pd.DataFrame(tuned_rows)

    baseline.to_csv(OUTPUT_DIR / "baseline_model_results.csv", index=False)
    tuned.to_csv(OUTPUT_DIR / "churn_model_results.csv", index=False)

    best_name = tuned.sort_values("F1-score", ascending=False).iloc[0]["Model"]
    best_model = tuned_models[best_name]
    best_pred = best_model.predict(X_test)

    print("Baseline results:\n", baseline)
    print("\nTuned results:\n", tuned)
    print("\nBest model:", best_name)
    print("Best parameters:", best_model.best_params_)
    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            best_pred,
            target_names=["No Churn", "Churn"],
            digits=4,
            zero_division=0,
        )
    )

    cm = confusion_matrix(y_test, best_pred)
    plt.figure(figsize=(5, 4))
    plt.imshow(cm)
    plt.title(f"Confusion Matrix - {best_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks([0, 1], ["No Churn", "Churn"])
    plt.yticks([0, 1], ["No Churn", "Churn"])
    for row in range(2):
        for col in range(2):
            plt.text(col, row, cm[row, col], ha="center", va="center")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close()

    tuned.set_index("Model")[["Accuracy", "Precision", "Recall", "F1-score"]].plot(
        kind="bar", figsize=(9, 5)
    )
    plt.title("Tuned Model Comparison")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "tuned_model_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
