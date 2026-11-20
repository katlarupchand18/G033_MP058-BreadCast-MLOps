"""Predict holdout data with champion model."""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from .preprocessing import clean_holdout_data, get_preprocessing_pipeline, load_params


def main():
    # Load champion model
    model = joblib.load("models/model.joblib")

    # Load and clean holdout data
    holdout_raw = pd.read_csv("data/raw/holdout_features.csv")
    holdout_clean = clean_holdout_data(holdout_raw)

    # Save cleaned holdout
    holdout_clean.to_csv("data/cleaned/holdout_cleaned.csv", index=False)
    print(f"Holdout cleaned: {len(holdout_clean)} rows")

    # Get feature columns
    params = load_params()
    feature_cols = params["preprocessing"]["feature_columns"]

    # Prepare features
    X_holdout = holdout_clean[feature_cols]

    # Predict
    predictions = model.predict(X_holdout)
    predictions = np.maximum(predictions, 0)  # Ensure non-negative

    # Create output DataFrame
    output = pd.DataFrame({
        "record_id": holdout_clean["record_id"],
        "prediction": predictions
    })

    # Save predictions
    output_path = Path("predictions/holdout_predictions.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")
    print(f"Shape: {output.shape}")
    print(f"Prediction stats: min={predictions.min():.2f}, max={predictions.max():.2f}, mean={predictions.mean():.2f}")

    return output


if __name__ == "__main__":
    main()