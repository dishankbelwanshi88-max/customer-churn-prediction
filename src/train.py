"""Train, compare and save churn-classification models."""
from pathlib import Path
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, ConfusionMatrixDisplay, RocCurveDisplay
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data_loader import load_data

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)


def prepare_data(df: pd.DataFrame):
    data = df.copy()
    data = data.drop(columns=["customerID"], errors="ignore")
    data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")
    data["Churn"] = data["Churn"].map({"Yes": 1, "No": 0})
    data = data.dropna(subset=["Churn"])
    X = data.drop(columns=["Churn"])
    y = data["Churn"].astype(int)
    return X, y


def make_pipeline(model):
    numeric = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
    categorical = [c for c in FEATURE_COLUMNS if c not in numeric]
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    preprocess = ColumnTransformer([
        ("num", numeric_pipe, numeric),
        ("cat", categorical_pipe, categorical)
    ])
    return Pipeline([("preprocess", preprocess), ("model", model)])


def main():
    global FEATURE_COLUMNS
    warnings.filterwarnings("ignore", category=FutureWarning)
    df = load_data()
    X, y = prepare_data(df)
    FEATURE_COLUMNS = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    candidates = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=2, class_weight="balanced",
            random_state=42, n_jobs=-1
        )
    }

    rows, fitted = [], {}
    for name, estimator in candidates.items():
        pipe = make_pipeline(estimator)
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        prob = pipe.predict_proba(X_test)[:, 1]
        rows.append({
            "model": name,
            "accuracy": accuracy_score(y_test, pred),
            "precision": precision_score(y_test, pred, zero_division=0),
            "recall": recall_score(y_test, pred, zero_division=0),
            "f1_score": f1_score(y_test, pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, prob)
        })
        fitted[name] = pipe
        print(f"Completed: {name}")

    metrics = pd.DataFrame(rows).sort_values("roc_auc", ascending=False)
    metrics.to_csv(OUT / "model_metrics.csv", index=False)
    print("\nHeld-out test metrics:")
    print(metrics.to_string(index=False))

    best_name = metrics.iloc[0]["model"]
    best_model = fitted[best_name]
    joblib.dump(best_model, OUT / "best_model.joblib")
    print(f"\nSaved selected model: {best_name}")

    fig, ax = plt.subplots(figsize=(7, 5))
    for name, model in fitted.items():
        RocCurveDisplay.from_estimator(model, X_test, y_test, name=name, ax=ax)
    ax.set_title("ROC Curve — Test Set")
    fig.tight_layout()
    fig.savefig(OUT / "roc_curve.png", dpi=160)
    plt.close(fig)

    pred = best_model.predict(X_test)
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_test, pred, display_labels=["Retained", "Churned"],
        cmap="Blues", ax=ax, colorbar=False
    )
    ax.set_title(f"Confusion Matrix — {best_name}")
    fig.tight_layout()
    fig.savefig(OUT / "confusion_matrix.png", dpi=160)
    plt.close(fig)

    # Permutation importance is measured on the held-out test set.
    perm = permutation_importance(
        best_model, X_test, y_test, n_repeats=8,
        random_state=42, scoring="roc_auc", n_jobs=-1
    )
    importance = pd.Series(perm.importances_mean, index=X_test.columns)
    importance = importance.sort_values().tail(12)
    fig, ax = plt.subplots(figsize=(8, 6))
    importance.plot(kind="barh", ax=ax)
    ax.set_title("Permutation Feature Importance (ROC-AUC)")
    ax.set_xlabel("Mean decrease in ROC-AUC after permutation")
    fig.tight_layout()
    fig.savefig(OUT / "feature_importance.png", dpi=160)
    plt.close(fig)

    print(f"\nOutputs saved in: {OUT}")


if __name__ == "__main__":
    main()
