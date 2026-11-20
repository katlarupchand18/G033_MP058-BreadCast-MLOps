"""Retrain model on combined train + first 70% of later data."""

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import joblib
import hashlib
from pathlib import Path
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline

from .preprocessing import (
    get_preprocessing_pipeline,
    get_feature_target_split,
    load_params,
)


def compute_data_sha256(filepath: str) -> str:
    """Compute SHA-256 of original train.csv."""
    with open(filepath, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def train_and_evaluate(X, y, model, model_name, cv=5):
    """Train model and compute cross-validation MAE."""
    kfold = KFold(n_splits=cv, shuffle=True, random_state=42)
    cv_scores = -cross_val_score(model, X, y, cv=kfold, scoring="neg_mean_absolute_error", n_jobs=-1)
    mean_mae = cv_scores.mean()
    std_mae = cv_scores.std()
    return mean_mae, std_mae


def main():
    # MLflow setup
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("bakery_sales_retrain")

    # Load combined training data
    combined_train = pd.read_csv("data/cleaned/combined_train.csv")

    # Load params
    params = load_params()
    feature_cols = params["preprocessing"]["feature_columns"]
    target_col = params["preprocessing"]["target_column"]

    # Split features and target
    X, y = get_feature_target_split(combined_train, feature_cols, target_col)

    # Get preprocessing pipeline
    preprocessor = get_preprocessing_pipeline(params)

    # Compute data SHA256
    data_sha256 = compute_data_sha256("data/raw/train.csv")

    # Define models to try (same as original training)
    models = {
        "DummyRegressor": DummyRegressor(strategy="mean"),
        "RandomForestRegressor": RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1),
        "HistGradientBoostingRegressor": HistGradientBoostingRegressor(max_depth=6, learning_rate=0.1, max_iter=200, random_state=42),
    }

    best_model = None
    best_mae = float("inf")
    best_name = None

    # Train and evaluate each model
    for name, model in models.items():
        print(f"Training {name}...")
        pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])
        mean_mae, std_mae = train_and_evaluate(X, y, pipeline, name)
        print(f"  CV MAE: {mean_mae:.4f} (+/- {std_mae:.4f})")

        if mean_mae < best_mae:
            best_mae = mean_mae
            best_model = pipeline
            best_name = name

    print(f"\nBest model: {best_name} (CV MAE: {best_mae:.4f})")

    # Fit best model on full combined data
    best_model.fit(X, y)

    # Log to MLflow
    with mlflow.start_run(run_name=f"{best_name}_retrain_v2") as run:
        mlflow.log_param("data_sha256", data_sha256)
        mlflow.log_param("model_type", best_name)
        mlflow.log_param("training_data", "train + first 70% later")
        mlflow.log_metric("cv_mae_mean", best_mae)
        mlflow.log_metric("cv_mae_std", std_mae)
        mlflow.log_metric("train_rows", len(combined_train))

        # Log model
        mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path="model",
            serialization_format="cloudpickle",
        )

        run_id = run.info.run_id
        print(f"MLflow run_id: {run_id}")

    # Register model as version 2
    client = MlflowClient(tracking_uri="sqlite:///mlflow.db")
    model_name = "bakery_sales_model"

    model_uri = f"runs:/{run_id}/model"
    mv = mlflow.register_model(model_uri, model_name)
    print(f"Registered model version: {mv.version}")

    # Evaluate on last 30% of later data
    later_clean = pd.read_csv("data/cleaned/later_cleaned.csv")
    later_clean["date"] = pd.to_datetime(later_clean["date"])
    later_sorted = later_clean.sort_values("date").reset_index(drop=True)

    n = len(later_sorted)
    split_idx = int(n * 0.7)
    later_test = later_sorted.iloc[split_idx:].copy()

    X_test, y_test = get_feature_target_split(later_test, feature_cols, target_col)
    v2_pred = best_model.predict(X_test)
    v2_mae = mean_absolute_error(y_test, v2_pred)

    print(f"\nv2 MAE on last 30%: {v2_mae:.4f}")

    # Also evaluate v1 on last 30% for comparison
    v1_model = joblib.load("models/model.joblib")
    v1_pred = v1_model.predict(X_test)
    v1_mae = mean_absolute_error(y_test, v1_pred)
    print(f"v1 MAE on last 30%: {v1_mae:.4f}")

    # Promotion rule: v2 MAE <= 0.85 * v1 MAE
    promotion_threshold = 0.85 * v1_mae
    should_promote = v2_mae <= promotion_threshold

    print(f"\nPromotion threshold (0.85 * v1 MAE): {promotion_threshold:.4f}")
    print(f"v2 MAE: {v2_mae:.4f}")
    print(f"Promote v2: {should_promote}")

    if should_promote:
        client.set_registered_model_alias(model_name, "champion", mv.version)
        print(f"Promoted v2 (version {mv.version}) to @champion")

        # Export new champion model
        joblib.dump(best_model, "models/model.joblib")
        print("Exported new champion model to models/model.joblib")
    else:
        print("Keeping v1 as @champion")

    # Save retraining summary
    summary = {
        "v1_mae_last30": v1_mae,
        "v2_mae_last30": v2_mae,
        "promotion_threshold": promotion_threshold,
        "promoted": should_promote,
        "v2_version": mv.version,
        "v2_run_id": run_id,
        "v2_model_type": best_name,
        "v2_cv_mae": best_mae,
        "combined_train_rows": len(combined_train),
        "later_train_rows": len(later_sorted) - len(later_test),
        "later_test_rows": len(later_test),
    }

    import json
    with open("reports/retraining_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\nRetraining summary saved to reports/retraining_summary.json")
    return summary


if __name__ == "__main__":
    main()