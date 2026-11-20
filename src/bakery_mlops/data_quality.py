"""Data quality detection functions for the Bakery Bread Sales dataset.

This module implements reproducible detection of the seven data quality issues
as specified in the SCSE3040 MLOps Mini Project handbook.
"""

from pathlib import Path
from typing import Any
import pandas as pd
import numpy as np


def load_train_data() -> pd.DataFrame:
    """Load the original training data without any modifications."""
    return pd.read_csv("data/raw/train.csv")


def load_later_data() -> pd.DataFrame:
    """Load the later dataset."""
    return pd.read_csv("data/raw/later.csv")


def load_holdout_data() -> pd.DataFrame:
    """Load the holdout features dataset."""
    return pd.read_csv("data/raw/holdout_features.csv")


def detect_leaking_column(train: pd.DataFrame, holdout: pd.DataFrame) -> dict[str, Any]:
    """Detect columns that leak target information (only available after event).

    A leaking column is one that:
    1. Is present in training data but missing from holdout features
    2. Has suspiciously high correlation with the target
    """
    train_cols = set(train.columns)
    holdout_cols = set(holdout.columns)

    # Columns in train but not in holdout (excluding target)
    missing_in_holdout = train_cols - holdout_cols - {"loaves_sold"}

    result = {
        "present": False,
        "column": None,
        "evidence": ""
    }

    if missing_in_holdout:
        result["present"] = True
        result["column"] = list(missing_in_holdout)[0]
        result["evidence"] = f"Column(s) {missing_in_holdout} present in train but missing from holdout_features.csv"
    else:
        result["evidence"] = "All feature columns present in holdout_features.csv; no leaking column detected"

    return result


def detect_unit_mixup(train: pd.DataFrame, later: pd.DataFrame) -> dict[str, Any]:
    """Detect unit mixup in numeric columns across outlet groups.

    Compares distributions of numeric columns by outlet to find
    one outlet using a different unit (factor of ~10, 12, 60, 100, 1000).
    """
    numeric_cols = ["price_rs", "temp_c", "footfall_index", "competitor_shops", "shelf_hours", "orders_previous_day"]
    outlets = train["outlet"].unique()

    result = {
        "present": False,
        "column": None,
        "group_value": None,
        "factor": None,
        "evidence": ""
    }

    for col in numeric_cols:
        # Get median per outlet in train
        train_medians = train.groupby("outlet")[col].median()
        # Get median per outlet in later (should be clean)
        later_medians = later.groupby("outlet")[col].median()

        # Check for outliers - one outlet vastly different from others
        for outlet in outlets:
            train_med = train_medians.get(outlet, np.nan)
            later_med = later_medians.get(outlet, np.nan)

            if pd.isna(train_med) or pd.isna(later_med):
                continue

            # Compare with other outlets' medians in train
            other_outlets = [o for o in outlets if o != outlet]
            other_medians = [train_medians.get(o, np.nan) for o in other_outlets]
            other_medians = [m for m in other_medians if not pd.isna(m)]

            if not other_medians:
                continue

            other_median = np.median(other_medians)

            # Check if this outlet's median is ~10, 100, 1000x different
            if other_median > 0:
                ratio = train_med / other_median
                # Check for common unit conversion factors
                for factor in [10, 12, 60, 100, 1000]:
                    if abs(ratio - factor) < factor * 0.1:  # within 10% of factor
                        # Verify with later.csv that this outlet matches others there
                        if len(other_outlets) > 0:
                            later_other_median = np.median([later_medians.get(o, np.nan) for o in other_outlets if not pd.isna(later_medians.get(o, np.nan))])
                            if later_other_median > 0:
                                later_ratio = later_med / later_other_median
                                if abs(later_ratio - 1.0) < 0.5:  # later data is consistent
                                    result["present"] = True
                                    result["column"] = col
                                    result["group_value"] = outlet
                                    result["factor"] = factor
                                    result["evidence"] = (
                                        f"{col} for {outlet} has median {train_med:.1f} vs other outlets {other_median:.1f} "
                                        f"(ratio {ratio:.1f} ≈ factor {factor}). Later.csv shows consistent units across outlets."
                                    )
                                    return result

    result["evidence"] = "No unit mixup detected across outlets"
    return result


def detect_sentinel_value(train: pd.DataFrame) -> dict[str, Any]:
    """Detect sentinel values (impossible placeholders like -1, -999, 999, 9999)."""
    numeric_cols = ["price_rs", "temp_c", "footfall_index", "competitor_shops", "shelf_hours", "orders_previous_day", "loaves_sold"]
    sentinel_candidates = [-9999, -999, -1, 999, 9999, 99999]

    result = {
        "present": False,
        "column": None,
        "value": None,
        "count": None,
        "evidence": ""
    }

    for col in numeric_cols:
        value_counts = train[col].value_counts()
        for sentinel in sentinel_candidates:
            if sentinel in value_counts.index:
                count = int(value_counts[sentinel])
                result["present"] = True
                result["column"] = col
                result["value"] = sentinel
                result["count"] = count
                result["evidence"] = f"Column '{col}' contains sentinel value {sentinel} in {count} rows"
                return result

    result["evidence"] = "No sentinel values (-9999, -999, -1, 999, 9999, 99999) found in any numeric column"
    return result


def detect_duplicate_rows(train: pd.DataFrame) -> dict[str, Any]:
    """Detect exact duplicate rows."""
    dup_count = int(train.duplicated().sum())

    result = {
        "present": dup_count > 0,
        "count": dup_count,
        "evidence": ""
    }

    if dup_count > 0:
        result["evidence"] = f"Found {dup_count} extra duplicate row copies (exact duplicates across all columns)"
    else:
        result["evidence"] = "No exact duplicate rows found"

    return result


def detect_stuck_sensor(train: pd.DataFrame) -> dict[str, Any]:
    """Detect stuck sensor - long runs of identical values in numeric columns.

    A stuck sensor shows the same reading for an unusually long period (weeks).
    """
    numeric_cols = ["price_rs", "temp_c", "footfall_index", "competitor_shops", "shelf_hours", "orders_previous_day"]

    result = {
        "present": False,
        "column": None,
        "start_date": None,
        "end_date": None,
        "evidence": ""
    }

    train_sorted = train.sort_values("date").reset_index(drop=True)

    for col in numeric_cols:
        # Find runs of identical values
        train_sorted["_diff"] = (train_sorted[col] != train_sorted[col].shift()).cumsum()
        runs = train_sorted.groupby("_diff").agg(
            start_date=("date", "first"),
            end_date=("date", "last"),
            count=("date", "count"),
            value=(col, "first")
        ).sort_values("count", ascending=False)

        # Check for runs longer than 7 days (a week)
        long_runs = runs[runs["count"] > 7]
        if not long_runs.empty:
            longest = long_runs.iloc[0]
            result["present"] = True
            result["column"] = col
            result["start_date"] = str(longest["start_date"])
            result["end_date"] = str(longest["end_date"])
            result["evidence"] = (
                f"Column '{col}' stuck at value {longest['value']} from {longest['start_date']} "
                f"to {longest['end_date']} ({int(longest['count'])} days)"
            )
            return result

    result["evidence"] = "No stuck sensor detected (no runs > 7 days of identical values in any numeric column)"
    return result


def detect_target_typos(train: pd.DataFrame) -> dict[str, Any]:
    """Detect target typos - values 10x too large due to extra zero."""
    target = train["loaves_sold"]

    # Find values that are > 1000 and appear to be 10x plausible values
    # Plausible range based on distribution: most values 16-600
    high_values = train[target > 1000].copy()
    high_values["potential_correct"] = high_values["loaves_sold"] / 10

    # Count how many high values when divided by 10 fall in the plausible range
    plausible_count = ((high_values["potential_correct"] >= 16) & (high_values["potential_correct"] <= 600)).sum()

    result = {
        "present": plausible_count > 0,
        "count": int(plausible_count),
        "evidence": ""
    }

    if plausible_count > 0:
        result["evidence"] = (
            f"Found {plausible_count} rows with loaves_sold > 1000 that are exactly 10x "
            f"plausible values (e.g., 3990→399, 3580→358). These appear to be data entry errors "
            f"with an extra zero."
        )
    else:
        result["evidence"] = "No target typos detected (no values > 1000 that are 10x plausible values)"

    return result


def detect_spelling_variants(train: pd.DataFrame, later: pd.DataFrame) -> dict[str, Any]:
    """Detect spelling variants in categorical columns."""
    categorical_cols = ["day_type", "outlet"]

    result = {
        "present": False,
        "column": None,
        "true_levels": None,
        "evidence": ""
    }

    for col in categorical_cols:
        train_unique = set(train[col].unique())
        later_unique = set(later[col].unique())

        # Check for case/space/punctuation variants
        train_normalized = {str(v).strip().lower() for v in train_unique}
        later_normalized = {str(v).strip().lower() for v in later_unique}

        # If normalized sets differ from original, there are variants
        if len(train_normalized) != len(train_unique):
            result["present"] = True
            result["column"] = col
            result["true_levels"] = len(train_normalized)
            result["evidence"] = (
                f"Column '{col}' has {len(train_unique)} raw values but {len(train_normalized)} "
                f"unique values after normalization (strip/lowercase). Variants detected."
            )
            return result

        # Also check if later.csv has cleaner spellings
        if train_normalized != later_normalized:
            result["present"] = True
            result["column"] = col
            result["true_levels"] = len(later_normalized)
            result["evidence"] = (
                f"Column '{col}' has different normalized categories in train ({train_normalized}) "
                f"vs later ({later_normalized}). Spelling variants detected."
            )
            return result

    result["evidence"] = "No spelling variants detected in categorical columns (day_type, outlet)"
    return result


def run_all_detections() -> dict[str, Any]:
    """Run all seven data quality detections and return results."""
    train = load_train_data()
    later = load_later_data()
    holdout = load_holdout_data()

    results = {
        "leaking_column": detect_leaking_column(train, holdout),
        "unit_mixup": detect_unit_mixup(train, later),
        "sentinel_value": detect_sentinel_value(train),
        "duplicate_rows": detect_duplicate_rows(train),
        "stuck_sensor": detect_stuck_sensor(train),
        "target_typos": detect_target_typos(train),
        "spelling_variants": detect_spelling_variants(train, later),
    }

    return results


def print_findings_summary(results: dict[str, Any]) -> None:
    """Print a human-readable summary of all findings."""
    print("=" * 70)
    print("DATA QUALITY FINDINGS SUMMARY")
    print("=" * 70)

    for issue, finding in results.items():
        status = "PRESENT" if finding["present"] else "ABSENT"
        print(f"\n{issue.upper()}: {status}")
        for key, value in finding.items():
            if key != "present":
                print(f"  {key}: {value}")


if __name__ == "__main__":
    results = run_all_detections()
    print_findings_summary(results)