"""Model training with MLflow tracking for Bakery Bread Sales.

Implements:
- Multiple model experiments (DummyRegressor, RandomForest, HistGradientBoosting)
- MLflow tracking to sqlite:///mlflow.db
- Logging data_sha256, parameters, MAE metrics
- Model registration with champion alias
- Export champion model to models/model.joblib
"""

import hashlib
import json
from pathlib import Path
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
import joblib
import yaml

from src.bakery_mlops.preprocessing import (
    prepare_train_data,
    get_preprocessing_pipeline,
    load_params,
)


def compute_data_sha256(filepath: str) -> str:
    """Compute SHA-256 hash of a file."""
    with open(filepath, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def get_model_configs(params: dict) -> list:
    """Get model configurations from params.yaml."""
    training = params.get("training", {})
    models_config = training.get("models", [])
    return models_config


def create_model_pipeline(model_config: dict, preprocessor) -> Pipeline:
    """Create a full pipeline with preprocessing and model."""
    model_name = model_config["name"]
    model_params = model_config.get("params", {})

    if model_name == "DummyRegressor":
        model = DummyRegressor(**model_params)
    elif model_name == "RandomForestRegressor":
        model = RandomForestRegressor(**model_params)
    elif model_name == "HistGradientBoostingRegressor":
        model = HistGradientBoostingRegressor(**model_params)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model),
    ])

    return pipeline


def evaluate_model_cv(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series, cv_folds: int = 5) -> Dict[str, float]:
    """Evaluate model using cross-validation (time-series aware)."""
    # Use TimeSeriesSplit for date-based data
    tscv = TimeSeriesSplit(n_splits=cv_folds)

    scores = cross_val_score(
        pipeline, X, y,
        cv=tscv,
        scoring="neg_mean_absolute_error",
        n_jobs=-1
    )

    mae_scores = -scores
    return {
        "mae_mean": float(mae_scores.mean()),
        "mae_std": float(mae_scores.std()),
        "mae_scores": [float(s) for s in mae_scores]
    }


def train_and_log_model(
    model_config: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    preprocessor,
    data_sha256: str,
    experiment_name: str,
    run_name: str = None
) -> Tuple[str, Dict[str, float]]:
    """Train a model and log to MLflow. Returns (run_id, metrics)."""
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id

        # Create pipeline
        pipeline = create_model_pipeline(model_config, preprocessor)

        # Log parameters
        mlflow.log_param("data_sha256", data_sha256)
        mlflow.log_param("model_name", model_config["name"])
        for param_name, param_value in model_config.get("params", {}).items():
            mlflow.log_param(f"model_{param_name}", param_value)

        # Log training params
        training_params = load_params().get("training", {})
        mlflow.log_param("cv_folds", training_params.get("cv_folds", 5))
        mlflow.log_param("random_seed", training_params.get("random_seed", 42))

        # Cross-validation evaluation
        cv_results = evaluate_model_cv(
            pipeline, X_train, y_train,
            cv_folds=training_params.get("cv_folds", 5)
        )

        # Log CV metrics
        mlflow.log_metric("cv_mae_mean", cv_results["mae_mean"])
        mlflow.log_metric("cv_mae_std", cv_results["mae_std"])
        for i, score in enumerate(cv_results["mae_scores"]):
            mlflow.log_metric(f"cv_mae_fold_{i}", score)

        # Fit on full training data
        pipeline.fit(X_train, y_train)

        # Log model
        mlflow.sklearn.log_model(
            pipeline,
            "model",
            serialization_format="cloudpickle"
        )

        # Log training MAE (on full data)
        train_pred = pipeline.predict(X_train)
        train_mae = mean_absolute_error(y_train, train_pred)
        mlflow.log_metric("train_mae", train_mae)

        print(f"  Run {run_id}: {model_config['name']} - CV MAE: {cv_results['mae_mean']:.3f} ± {cv_results['mae_std']:.3f}")

        return run_id, {
            "cv_mae_mean": cv_results["mae_mean"],
            "cv_mae_std": cv_results["mae_std"],
            "train_mae": train_mae,
            "model_name": model_config["name"]
        }


def register_model(
    run_id: str,
    model_name: str,
    version: int = None
) -> int:
    """Register model in MLflow registry and return version number."""
    client = MlflowClient()

    # Register model
    model_uri = f"runs:/{run_id}/model"
    mv = mlflow.register_model(model_uri, model_name)
    registered_version = mv.version

    print(f"  Registered model '{model_name}' version {registered_version}")
    return registered_version


def set_champion_alias(model_name: str, version: int, alias: str = "champion") -> None:
    """Set the champion alias for a model version."""
    client = MlflowClient()
    client.set_registered_model_alias(model_name, alias, version)
    print(f"  Set alias '{alias}' to version {version}")


def export_champion_model(model_name: str, alias: str, output_path: str) -> None:
    """Export champion model to a joblib file for serving."""
    client = MlflowClient()

    # Get champion model version
    mv = client.get_model_version_by_alias(model_name, alias)
    version = mv.version

    # Download and save model
    model_uri = f"models:/{model_name}@{alias}"
    model = mlflow.sklearn.load_model(model_uri)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)
    print(f"  Exported champion model (v{version}) to {output_path}")


def run_training_experiment() -> Dict[str, Any]:
    """Run complete training experiment with multiple models."""
    params = load_params()
    mlflow_params = params.get("mlflow", {})
    training_params = params.get("training", {})

    # Set MLflow tracking URI
    mlflow.set_tracking_uri(mlflow_params.get("tracking_uri", "sqlite:///mlflow.db"))

    # Prepare data
    print("Preparing training data...")
    X_train, y_train, cleaned_train = prepare_train_data()

    # Get preprocessing pipeline
    preprocessor = get_preprocessing_pipeline(params)

    # Compute data SHA-256
    data_sha256 = compute_data_sha256("data/raw/train.csv")
    print(f"Data SHA-256: {data_sha256}")

    # Get model configurations
    model_configs = get_model_configs(params)

    # Run experiments
    experiment_name = mlflow_params.get("experiment_name", "bakery_sales_g033_mp058")
    results = []

    for i, model_config in enumerate(model_configs):
        print(f"\nTraining model {i+1}/{len(model_configs)}: {model_config['name']}")
        run_name = f"{model_config['name']}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"

        run_id, metrics = train_and_log_model(
            model_config,
            X_train, y_train,
            preprocessor,
            data_sha256,
            experiment_name,
            run_name
        )

        results.append({
            "run_id": run_id,
            "model_name": model_config["name"],
            "metrics": metrics
        })

    # Select best model based on CV MAE
    best_result = min(results, key=lambda r: r["metrics"]["cv_mae_mean"])
    best_run_id = best_result["run_id"]
    best_model_name = best_result["model_name"]

    print(f"\nBest model: {best_model_name} (run {best_run_id})")
    print(f"  CV MAE: {best_result['metrics']['cv_mae_mean']:.3f}")

    # Register best model
    registered_model_name = mlflow_params.get("registered_model_name", "bakery_sales_model")
    champion_alias = mlflow_params.get("champion_alias", "champion")

    version = register_model(best_run_id, registered_model_name)
    set_champion_alias(registered_model_name, version, champion_alias)

    # Export champion model
    model_output_path = params.get("output", {}).get("model_path", "models/model.joblib")
    export_champion_model(registered_model_name, champion_alias, model_output_path)

    # Save training summary
    summary = {
        "best_run_id": best_run_id,
        "best_model_name": best_model_name,
        "registered_model_name": registered_model_name,
        "champion_version": version,
        "data_sha256": data_sha256,
        "all_results": results
    }

    summary_path = "reports/training_summary.json"
    Path(summary_path).parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nTraining summary saved to {summary_path}")

    return summary


if __name__ == "__main__":
    print("=" * 60)
    print("MODEL TRAINING WITH MLFLOW")
    print("=" * 60)
    summary = run_training_experiment()
    print("\nDone!")