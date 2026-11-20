"""Data cleaning and preprocessing pipeline for Bakery Bread Sales dataset.

This module implements reproducible cleaning based on verified data quality issues:
1. Unit mixup: price_rs for College Gate outlet (factor 100)
2. Duplicate rows: 49 exact duplicate copies
3. Target typos: 15 rows with loaves_sold 10x too large
"""

from pathlib import Path
from typing import Tuple
import pandas as pd
import numpy as np
import yaml


def load_params() -> dict:
    """Load parameters from params.yaml."""
    with open("params.yaml") as f:
        return yaml.safe_load(f)


def load_raw_train() -> pd.DataFrame:
    """Load the original raw training data."""
    return pd.read_csv("data/raw/train.csv")


def load_raw_later() -> pd.DataFrame:
    """Load the original raw later data."""
    return pd.read_csv("data/raw/later.csv")


def load_raw_holdout() -> pd.DataFrame:
    """Load the original raw holdout features data."""
    return pd.read_csv("data/raw/holdout_features.csv")


def clean_train_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all cleaning steps to training data.

    Steps (in order):
    1. Remove exact duplicate rows
    2. Fix unit mixup: price_rs for College Gate (divide by 100)
    3. Fix target typos: loaves_sold > 1000 divided by 10
    4. Reset index
    """
    df = df.copy()
    params = load_params()
    dq = params.get("data_quality", {})

    # 1. Remove exact duplicate rows
    if dq.get("duplicates", {}).get("drop_exact", True):
        before = len(df)
        df = df.drop_duplicates().reset_index(drop=True)
        removed = before - len(df)
        if removed > 0:
            print(f"  Removed {removed} exact duplicate rows")

    # 2. Fix unit mixup: price_rs for College Gate
    unit_mixup = dq.get("price_rs_unit_mixup", {})
    if unit_mixup.get("column") == "price_rs":
        affected_outlet = unit_mixup.get("affected_outlet", "College Gate")
        factor = unit_mixup.get("factor", 100)
        mask = df["outlet"] == affected_outlet
        df.loc[mask, "price_rs"] = df.loc[mask, "price_rs"] / factor
        print(f"  Fixed unit mixup: divided price_rs by {factor} for {mask.sum()} rows in '{affected_outlet}'")

    # 3. Fix target typos: loaves_sold > threshold divided by factor
    target_typos = dq.get("target_typos", {})
    if target_typos.get("column") == "loaves_sold":
        threshold = target_typos.get("threshold", 1000)
        factor = target_typos.get("factor", 10)
        mask = df["loaves_sold"] > threshold
        df.loc[mask, "loaves_sold"] = df.loc[mask, "loaves_sold"] / factor
        print(f"  Fixed target typos: divided loaves_sold by {factor} for {mask.sum()} rows > {threshold}")

    # 4. Reset index
    df = df.reset_index(drop=True)

    return df


def clean_later_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply cleaning to later data (only unit mixup if needed).

    Later data is clean per problem statement, but we apply
    the same unit correction for consistency if needed.
    """
    df = df.copy()
    params = load_params()
    dq = params.get("data_quality", {})

    # Later.csv already has consistent units (verified), but apply
    # same correction if the issue existed in later too
    unit_mixup = dq.get("price_rs_unit_mixup", {})
    if unit_mixup.get("column") == "price_rs":
        affected_outlet = unit_mixup.get("affected_outlet", "College Gate")
        factor = unit_mixup.get("factor", 100)
        mask = df["outlet"] == affected_outlet
        # Check if later data has the same issue
        if mask.any() and df.loc[mask, "price_rs"].median() > 1000:
            df.loc[mask, "price_rs"] = df.loc[mask, "price_rs"] / factor
            print(f"  Fixed unit mixup in later: divided price_rs by {factor} for {mask.sum()} rows")

    return df.reset_index(drop=True)


def clean_holdout_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply cleaning to holdout features (only unit mixup if needed)."""
    df = df.copy()
    params = load_params()
    dq = params.get("data_quality", {})

    unit_mixup = dq.get("price_rs_unit_mixup", {})
    if unit_mixup.get("column") == "price_rs":
        affected_outlet = unit_mixup.get("affected_outlet", "College Gate")
        factor = unit_mixup.get("factor", 100)
        mask = df["outlet"] == affected_outlet
        if mask.any() and df.loc[mask, "price_rs"].median() > 1000:
            df.loc[mask, "price_rs"] = df.loc[mask, "price_rs"] / factor
            print(f"  Fixed unit mixup in holdout: divided price_rs by {factor} for {mask.sum()} rows")

    return df.reset_index(drop=True)


def get_feature_target_split(
    df: pd.DataFrame,
    feature_columns: list,
    target_column: str
) -> Tuple[pd.DataFrame, pd.Series]:
    """Split dataframe into features X and target y."""
    X = df[feature_columns].copy()
    y = df[target_column].copy()
    return X, y


def get_preprocessing_pipeline(params: dict):
    """Create sklearn preprocessing pipeline for features.

    Returns a ColumnTransformer that handles:
    - Numeric features: StandardScaler
    - Categorical features: OneHotEncoder
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.pipeline import Pipeline

    preprocess = params.get("preprocessing", {})
    numeric_cols = preprocess.get("numeric_columns", [])
    categorical_cols = preprocess.get("categorical_columns", [])

    numeric_transformer = Pipeline(steps=[
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ],
        remainder="drop"  # Drop columns not specified (like date, record_id)
    )

    return preprocessor


def prepare_train_data() -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Prepare cleaned training data and return X, y, and full cleaned df."""
    raw_train = load_raw_train()
    cleaned_train = clean_train_data(raw_train)

    params = load_params()
    preprocess = params.get("preprocessing", {})
    feature_cols = preprocess.get("feature_columns", [])
    target_col = preprocess.get("target_column", "loaves_sold")

    X, y = get_feature_target_split(cleaned_train, feature_cols, target_col)
    return X, y, cleaned_train


def prepare_later_data() -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Prepare cleaned later data for monitoring/retraining."""
    raw_later = load_raw_later()
    cleaned_later = clean_later_data(raw_later)

    params = load_params()
    preprocess = params.get("preprocessing", {})
    feature_cols = preprocess.get("feature_columns", [])
    target_col = preprocess.get("target_column", "loaves_sold")

    X, y = get_feature_target_split(cleaned_later, feature_cols, target_col)
    return X, y, cleaned_later


def prepare_holdout_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Prepare cleaned holdout features for final prediction."""
    raw_holdout = load_raw_holdout()
    cleaned_holdout = clean_holdout_data(raw_holdout)

    params = load_params()
    preprocess = params.get("preprocessing", {})
    feature_cols = preprocess.get("feature_columns", [])
    id_col = preprocess.get("id_column", "record_id")

    X = cleaned_holdout[feature_cols].copy()
    record_ids = cleaned_holdout[id_col].copy()
    return X, record_ids, cleaned_holdout


def save_cleaned_data(df: pd.DataFrame, path: str) -> None:
    """Save cleaned dataframe to CSV."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Saved cleaned data to {path} ({len(df)} rows)")


if __name__ == "__main__":
    print("=" * 60)
    print("CLEANING TRAINING DATA")
    print("=" * 60)
    X_train, y_train, cleaned_train = prepare_train_data()
    print(f"\nCleaned train shape: {cleaned_train.shape}")
    print(f"Features shape: {X_train.shape}")
    print(f"Target shape: {y_train.shape}")
    print(f"Target stats: min={y_train.min():.1f}, max={y_train.max():.1f}, mean={y_train.mean():.1f}")

    print("\n" + "=" * 60)
    print("CLEANING LATER DATA")
    print("=" * 60)
    X_later, y_later, cleaned_later = prepare_later_data()
    print(f"\nCleaned later shape: {cleaned_later.shape}")
    print(f"Features shape: {X_later.shape}")
    print(f"Target shape: {y_later.shape}")

    print("\n" + "=" * 60)
    print("CLEANING HOLDOUT DATA")
    print("=" * 60)
    X_holdout, record_ids, cleaned_holdout = prepare_holdout_data()
    print(f"\nCleaned holdout shape: {cleaned_holdout.shape}")
    print(f"Features shape: {X_holdout.shape}")

    # Save cleaned data
    print("\n" + "=" * 60)
    print("SAVING CLEANED DATA")
    print("=" * 60)
    save_cleaned_data(cleaned_train, "data/cleaned/train_cleaned.csv")
    save_cleaned_data(cleaned_later, "data/cleaned/later_cleaned.csv")
    save_cleaned_data(cleaned_holdout, "data/cleaned/holdout_cleaned.csv")

    # Verify key statistics
    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)
    print(f"Train price_rs by outlet:")
    for outlet in cleaned_train["outlet"].unique():
        med = cleaned_train[cleaned_train["outlet"] == outlet]["price_rs"].median()
        print(f"  {outlet}: median={med:.1f}")
    print(f"Train loaves_sold range: {cleaned_train['loaves_sold'].min():.1f} - {cleaned_train['loaves_sold'].max():.1f}")
    print(f"Train duplicates: {cleaned_train.duplicated().sum()}")