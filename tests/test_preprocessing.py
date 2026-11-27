"""Tests for data preprocessing and feature transformation."""

import pandas as pd
import numpy as np
import pytest
from src.bakery_mlops.preprocessing import (
    clean_train_data,
    clean_later_data,
    clean_holdout_data,
    get_preprocessing_pipeline,
    get_feature_target_split,
    load_params,
)


class TestPreprocessing:
    """Test data cleaning and preprocessing functions."""

    def setup_method(self):
        """Set up test data."""
        self.raw_train = pd.DataFrame({
            "record_id": ["R1", "R2", "R2", "R3", "R4", "R5"],
            "date": ["2026-01-01", "2026-01-02", "2026-01-02", "2026-01-03", "2026-01-04", "2026-01-05"],
            "price_rs": [5000.0, 5000.0, 5000.0, 50.0, 50.0, 50.0],  # College Gate = 5000, others = 50
            "temp_c": [25.0, 26.0, 26.0, 27.0, 28.0, 29.0],
            "footfall_index": [50, 60, 60, 70, 80, 90],
            "competitor_shops": [2, 3, 3, 4, 5, 1],
            "shelf_hours": [8.0, 9.0, 9.0, 10.0, 11.0, 12.0],
            "orders_previous_day": [100, 200, 200, 300, 400, 500],
            "day_type": ["weekday", "weekend", "weekend", "festival", "weekday", "weekend"],
            "outlet": ["College Gate", "College Gate", "College Gate", "Mall Road", "Mall Road", "Station Road"],
            "loaves_sold": [300, 4000, 4000, 350, 450, 550]  # 4000 is 10x typo
        })

    def test_clean_train_data_removes_duplicates(self):
        """Test that duplicate rows are removed."""
        cleaned = clean_train_data(self.raw_train)
        assert len(cleaned) == 5  # 6 - 1 duplicate
        assert cleaned.duplicated().sum() == 0

    def test_clean_train_data_fixes_unit_mixup(self):
        """Test that unit mixup is fixed for College Gate."""
        cleaned = clean_train_data(self.raw_train)
        college_gate = cleaned[cleaned["outlet"] == "College Gate"]
        mall_road = cleaned[cleaned["outlet"] == "Mall Road"]

        # After cleaning, College Gate price_rs should be ~50 (was 5000)
        assert college_gate["price_rs"].median() == pytest.approx(50.0, rel=0.1)
        assert mall_road["price_rs"].median() == 50.0

    def test_clean_train_data_fixes_target_typos(self):
        """Test that target typos (10x) are fixed."""
        cleaned = clean_train_data(self.raw_train)
        # The 4000 value should become 400
        assert cleaned["loaves_sold"].max() < 1000
        assert 400 in cleaned["loaves_sold"].values

    def test_clean_later_data(self):
        """Test later data cleaning."""
        raw_later = pd.DataFrame({
            "record_id": ["L1", "L2"],
            "date": ["2026-07-01", "2026-07-02"],
            "price_rs": [5000.0, 50.0],
            "temp_c": [30.0, 31.0],
            "footfall_index": [50, 60],
            "competitor_shops": [2, 3],
            "shelf_hours": [8.0, 9.0],
            "orders_previous_day": [100, 200],
            "day_type": ["weekday", "weekend"],
            "outlet": ["College Gate", "Mall Road"],
            "loaves_sold": [300, 400]
        })
        cleaned = clean_later_data(raw_later)
        college_gate = cleaned[cleaned["outlet"] == "College Gate"]
        # Later data should already be clean, but if not, fix it
        assert college_gate["price_rs"].iloc[0] == pytest.approx(50.0, rel=0.1)

    def test_clean_holdout_data(self):
        """Test holdout data cleaning."""
        raw_holdout = pd.DataFrame({
            "record_id": ["H1", "H2"],
            "date": ["2026-09-01", "2026-09-02"],
            "price_rs": [5000.0, 50.0],
            "temp_c": [30.0, 31.0],
            "footfall_index": [50, 60],
            "competitor_shops": [2, 3],
            "shelf_hours": [8.0, 9.0],
            "orders_previous_day": [100, 200],
            "day_type": ["weekday", "weekend"],
            "outlet": ["College Gate", "Mall Road"]
        })
        cleaned = clean_holdout_data(raw_holdout)
        college_gate = cleaned[cleaned["outlet"] == "College Gate"]
        assert college_gate["price_rs"].iloc[0] == pytest.approx(50.0, rel=0.1)

    def test_get_feature_target_split(self):
        """Test feature/target splitting."""
        cleaned = clean_train_data(self.raw_train)
        params = load_params()
        feature_cols = params["preprocessing"]["feature_columns"]
        target_col = params["preprocessing"]["target_column"]

        X, y = get_feature_target_split(cleaned, feature_cols, target_col)

        assert len(X) == len(cleaned)
        assert len(y) == len(cleaned)
        assert list(X.columns) == feature_cols
        assert y.name == target_col

    def test_get_preprocessing_pipeline(self):
        """Test that preprocessing pipeline is created correctly."""
        params = load_params()
        preprocessor = get_preprocessing_pipeline(params)

        # Check it's a ColumnTransformer
        from sklearn.compose import ColumnTransformer
        assert isinstance(preprocessor, ColumnTransformer)

        # Check transformers exist (use transformers attribute, not transformers_)
        transformer_names = [name for name, _, _ in preprocessor.transformers]
        assert "num" in transformer_names
        assert "cat" in transformer_names

    def test_preprocessing_pipeline_fit_transform(self):
        """Test that preprocessing pipeline works on data."""
        cleaned = clean_train_data(self.raw_train)
        params = load_params()
        preprocessor = get_preprocessing_pipeline(params)
        feature_cols = params["preprocessing"]["feature_columns"]

        X = cleaned[feature_cols]
        X_transformed = preprocessor.fit_transform(X)

        # Should produce numeric array
        assert isinstance(X_transformed, np.ndarray)
        assert X_transformed.shape[0] == len(X)
        assert X_transformed.shape[1] > 0  # Some features after encoding

    def test_clean_train_data_preserves_original(self):
        """Ensure cleaning does not modify the original data."""
        original = self.raw_train.copy(deep=True)

        clean_train_data(self.raw_train)

        pd.testing.assert_frame_equal(self.raw_train, original)

    def test_clean_data_preserves_correct_prices(self):
        """Ensure correct prices are not converted again."""
        data = pd.DataFrame({
            "outlet": ["College Gate", "Mall Road"],
            "price_rs": [50.0, 60.0]
        })

        later = clean_later_data(data)
        holdout = clean_holdout_data(data)

        assert later["price_rs"].tolist() == [50.0, 60.0]
        assert holdout["price_rs"].tolist() == [50.0, 60.0]

    def test_pipeline_handles_unseen_category(self):
        """Ensure the pipeline handles unknown outlet categories."""
        cleaned = clean_train_data(self.raw_train)
        params = load_params()

        feature_cols = params["preprocessing"]["feature_columns"]
        X = cleaned[feature_cols]

        preprocessor = get_preprocessing_pipeline(params)
        X_train = preprocessor.fit_transform(X)

        new_data = X.iloc[[0]].copy()
        new_data["outlet"] = "New Outlet"

        transformed = preprocessor.transform(new_data)

        assert transformed.shape[0] == 1
        assert transformed.shape[1] == X_train.shape[1]
        assert np.isfinite(transformed).all()




if __name__ == "__main__":
    pytest.main([__file__, "-v"])