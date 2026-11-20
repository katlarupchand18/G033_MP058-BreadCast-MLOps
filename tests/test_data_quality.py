"""Tests for data quality detection functions."""

import pandas as pd
import numpy as np
import pytest
from src.bakery_mlops.data_quality import (
    detect_leaking_column,
    detect_unit_mixup,
    detect_sentinel_value,
    detect_duplicate_rows,
    detect_stuck_sensor,
    detect_target_typos,
    detect_spelling_variants,
    run_all_detections,
)


def infer_length(kwargs: dict, default: int = 3) -> int:
    """Infer array length from kwargs, default to 3."""
    for key, value in kwargs.items():
        if isinstance(value, (list, tuple)):
            return len(value)
    return default


def make_train_df(**kwargs):
    """Create a train DataFrame with all required columns."""
    n = infer_length(kwargs, 3)
    day_types = ["weekday", "weekend", "festival"]
    outlets = ["College Gate", "Mall Road", "Station Road"]
    defaults = {
        "record_id": [f"R{i+1}" for i in range(n)],
        "date": [f"2026-01-{i+1:02d}" for i in range(n)],
        "price_rs": [50.0] * n,
        "temp_c": [25.0 + i for i in range(n)],
        "footfall_index": [50 + i*10 for i in range(n)],
        "competitor_shops": [2 + i for i in range(n)],
        "shelf_hours": [8.0 + i for i in range(n)],
        "orders_previous_day": [100 + i*100 for i in range(n)],
        "day_type": [day_types[i % 3] for i in range(n)],
        "outlet": [outlets[i % 3] for i in range(n)],
        "loaves_sold": [300 + i*100 for i in range(n)]
    }
    defaults.update(kwargs)
    return pd.DataFrame(defaults)


def make_later_df(**kwargs):
    """Create a later DataFrame with all required columns."""
    n = infer_length(kwargs, 2)
    day_types = ["weekday", "weekend"]
    outlets = ["College Gate", "Mall Road"]
    defaults = {
        "record_id": [f"L{i+1}" for i in range(n)],
        "date": [f"2026-07-{i+1:02d}" for i in range(n)],
        "price_rs": [50.0] * n,
        "temp_c": [30.0 + i for i in range(n)],
        "footfall_index": [50 + i*10 for i in range(n)],
        "competitor_shops": [2 + i for i in range(n)],
        "shelf_hours": [8.0 + i for i in range(n)],
        "orders_previous_day": [100 + i*100 for i in range(n)],
        "day_type": [day_types[i % 2] for i in range(n)],
        "outlet": [outlets[i % 2] for i in range(n)],
        "loaves_sold": [300 + i*100 for i in range(n)]
    }
    defaults.update(kwargs)
    return pd.DataFrame(defaults)


def make_holdout_df(**kwargs):
    """Create a holdout DataFrame with all required columns (no target)."""
    n = infer_length(kwargs, 2)
    day_types = ["weekday", "weekend"]
    outlets = ["College Gate", "Mall Road"]
    defaults = {
        "record_id": [f"H{i+1}" for i in range(n)],
        "date": [f"2026-09-{i+1:02d}" for i in range(n)],
        "price_rs": [50.0] * n,
        "temp_c": [30.0 + i for i in range(n)],
        "footfall_index": [50 + i*10 for i in range(n)],
        "competitor_shops": [2 + i for i in range(n)],
        "shelf_hours": [8.0 + i for i in range(n)],
        "orders_previous_day": [100 + i*100 for i in range(n)],
        "day_type": [day_types[i % 2] for i in range(n)],
        "outlet": [outlets[i % 2] for i in range(n)]
    }
    defaults.update(kwargs)
    return pd.DataFrame(defaults)


class TestDataQualityDetection:
    """Test data quality detection on synthetic data."""

    def test_detect_leaking_column_absent(self):
        """Test leaking column detection when no leaking column exists."""
        train = make_train_df()
        holdout = make_holdout_df()
        result = detect_leaking_column(train, holdout)
        assert result["present"] is False
        assert result["column"] is None

    def test_detect_leaking_column_present(self):
        """Test leaking column detection when a leaking column exists."""
        train = make_train_df()
        train["leaking_col"] = train["loaves_sold"]  # Perfect correlation with target
        holdout = make_holdout_df()
        result = detect_leaking_column(train, holdout)
        assert result["present"] is True
        assert result["column"] == "leaking_col"

    def test_detect_unit_mixup_present(self):
        """Test unit mixup detection."""
        train = pd.DataFrame({
            "record_id": ["R1", "R2", "R3", "R4", "R5", "R6"],
            "date": ["2026-01-01"] * 6,
            "price_rs": [1000, 1000, 1000, 10, 10, 10],
            "temp_c": [25.0] * 6,
            "footfall_index": [50] * 6,
            "competitor_shops": [2] * 6,
            "shelf_hours": [8.0] * 6,
            "orders_previous_day": [100] * 6,
            "day_type": ["weekday"] * 6,
            "outlet": ["A", "A", "A", "B", "B", "B"],
            "loaves_sold": [300] * 6
        })
        later = pd.DataFrame({
            "record_id": ["L1", "L2", "L3", "L4", "L5", "L6"],
            "date": ["2026-07-01"] * 6,
            "price_rs": [10, 10, 10, 10, 10, 10],
            "temp_c": [30.0] * 6,
            "footfall_index": [50] * 6,
            "competitor_shops": [2] * 6,
            "shelf_hours": [8.0] * 6,
            "orders_previous_day": [100] * 6,
            "day_type": ["weekday"] * 6,
            "outlet": ["A", "A", "A", "B", "B", "B"],
            "loaves_sold": [300] * 6
        })
        result = detect_unit_mixup(train, later)
        assert result["present"] is True
        assert result["column"] == "price_rs"
        assert result["group_value"] == "A"
        assert result["factor"] == 100

    def test_detect_unit_mixup_absent(self):
        """Test unit mixup detection when no mixup exists."""
        train = make_train_df(outlet=["A", "A", "A", "B", "B", "B"])
        later = make_later_df(outlet=["A", "A", "A", "B", "B", "B"])
        result = detect_unit_mixup(train, later)
        assert result["present"] is False

    def test_detect_sentinel_value_present(self):
        """Test sentinel value detection."""
        train = make_train_df(price_rs=[1, 2, -999, 4, 5, 6])
        result = detect_sentinel_value(train)
        assert result["present"] is True
        assert result["column"] == "price_rs"
        assert result["value"] == -999
        assert result["count"] == 1

    def test_detect_sentinel_value_absent(self):
        """Test sentinel value detection when none exist."""
        train = make_train_df(record_id=["R1", "R2", "R3", "R4", "R5", "R6"])
        result = detect_sentinel_value(train)
        assert result["present"] is False

    def test_detect_duplicate_rows_present(self):
        """Test duplicate row detection."""
        train = pd.DataFrame({
            "a": [1, 2, 2, 3],
            "b": [4, 5, 5, 6]
        })
        result = detect_duplicate_rows(train)
        assert result["present"] is True
        assert result["count"] == 1  # One extra copy

    def test_detect_duplicate_rows_absent(self):
        """Test duplicate row detection when none exist."""
        train = pd.DataFrame({
            "a": [1, 2, 3],
            "b": [4, 5, 6]
        })
        result = detect_duplicate_rows(train)
        assert result["present"] is False
        assert result["count"] == 0

    def test_detect_target_typos_present(self):
        """Test target typo detection (10x values)."""
        train = make_train_df(loaves_sold=[100, 200, 3000, 400, 5000, 600])
        result = detect_target_typos(train)
        assert result["present"] == True
        assert result["count"] == 2

    def test_detect_target_typos_absent(self):
        """Test target typo detection when none exist."""
        train = make_train_df(record_id=["R1", "R2", "R3", "R4", "R5", "R6"])
        result = detect_target_typos(train)
        assert result["present"] == False
        assert result["count"] == 0

    def test_detect_spelling_variants_present(self):
        """Test spelling variant detection."""
        train = make_train_df(day_type=["weekday", "Weekday", "WEEKDAY", "weekend", "festival", "weekday"])
        later = make_later_df(day_type=["weekday", "weekend"])
        result = detect_spelling_variants(train, later)
        assert result["present"] is True
        assert result["column"] == "day_type"
        assert result["true_levels"] == 3  # weekday, weekend, festival

    def test_detect_spelling_variants_absent(self):
        """Test spelling variant detection when none exist."""
        # Create train and later with identical normalized categories for BOTH day_type and outlet
        train = make_train_df(
            record_id=["R1", "R2", "R3", "R4", "R5", "R6"],
            day_type=["weekday", "weekend", "festival", "weekday", "weekend", "festival"],
            outlet=["College Gate", "Mall Road", "Station Road", "College Gate", "Mall Road", "Station Road"]
        )
        later = make_later_df(
            day_type=["weekday", "weekend", "festival", "weekday"],
            outlet=["College Gate", "Mall Road", "Station Road", "College Gate"]
        )
        result = detect_spelling_variants(train, later)
        assert result["present"] == False

    def test_run_all_detections(self):
        """Test that all detections run without error on real data."""
        results = run_all_detections()
        assert isinstance(results, dict)
        assert len(results) == 7
        for key in ["leaking_column", "unit_mixup", "sentinel_value",
                      "duplicate_rows", "stuck_sensor", "target_typos",
                      "spelling_variants"]:
            assert key in results
            assert "present" in results[key]
            assert isinstance(results[key]["present"], (bool, np.bool_))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])