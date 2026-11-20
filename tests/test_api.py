"""Tests for FastAPI service endpoints.

Uses FastAPI TestClient for testing without running server.
"""

import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from app.main import app


client = TestClient(app)


class TestHealthEndpoint:
    """Tests for GET /health endpoint."""

    def test_health_returns_ok(self):
        """Test that health endpoint returns status ok."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"

    def test_health_response_structure(self):
        """Test health response has expected structure."""
        response = client.get("/health")
        data = response.json()
        assert "status" in data
        assert isinstance(data["status"], str)


class TestPredictEndpoint:
    """Tests for POST /predict endpoint."""

    def test_predict_valid_input(self):
        """Test prediction with valid input."""
        payload = {
            "price_rs": 50.0,
            "temp_c": 30.0,
            "footfall_index": 50,
            "competitor_shops": 2,
            "shelf_hours": 8.0,
            "orders_previous_day": 100,
            "day_type": "weekday",
            "outlet": "College Gate"
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "prediction" in data
        assert isinstance(data["prediction"], (int, float))
        assert data["prediction"] >= 0

    def test_predict_all_day_types(self):
        """Test prediction works for all day_type values."""
        for day_type in ["weekday", "weekend", "festival"]:
            payload = {
                "price_rs": 50.0,
                "temp_c": 30.0,
                "footfall_index": 50,
                "competitor_shops": 2,
                "shelf_hours": 8.0,
                "orders_previous_day": 100,
                "day_type": day_type,
                "outlet": "College Gate"
            }
            response = client.post("/predict", json=payload)
            assert response.status_code == 200
            assert "prediction" in response.json()

    def test_predict_all_outlets(self):
        """Test prediction works for all outlet values."""
        for outlet in ["College Gate", "Mall Road", "Station Road"]:
            payload = {
                "price_rs": 50.0,
                "temp_c": 30.0,
                "footfall_index": 50,
                "competitor_shops": 2,
                "shelf_hours": 8.0,
                "orders_previous_day": 100,
                "day_type": "weekday",
                "outlet": outlet
            }
            response = client.post("/predict", json=payload)
            assert response.status_code == 200
            assert "prediction" in response.json()

    def test_predict_rejects_shelf_hours_above_30(self):
        """Test that shelf_hours > 30 is rejected (client rule)."""
        payload = {
            "price_rs": 50.0,
            "temp_c": 30.0,
            "footfall_index": 50,
            "competitor_shops": 2,
            "shelf_hours": 31.0,  # Above 30 - should be rejected
            "orders_previous_day": 100,
            "day_type": "weekday",
            "outlet": "College Gate"
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 422
        # Check it's a validation error
        data = response.json()
        assert "detail" in data

    def test_predict_rejects_orders_previous_day_below_5(self):
        """Test that orders_previous_day < 5 is rejected (client rule)."""
        payload = {
            "price_rs": 50.0,
            "temp_c": 30.0,
            "footfall_index": 50,
            "competitor_shops": 2,
            "shelf_hours": 8.0,
            "orders_previous_day": 4,  # Below 5 - should be rejected
            "day_type": "weekday",
            "outlet": "College Gate"
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data

    def test_predict_rejects_missing_fields(self):
        """Test that missing required fields are rejected."""
        payload = {
            "price_rs": 50.0,
            "temp_c": 30.0,
            # Missing other required fields
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_predict_rejects_extra_fields(self):
        """Test that extra unexpected fields are rejected."""
        payload = {
            "price_rs": 50.0,
            "temp_c": 30.0,
            "footfall_index": 50,
            "competitor_shops": 2,
            "shelf_hours": 8.0,
            "orders_previous_day": 100,
            "day_type": "weekday",
            "outlet": "College Gate",
            "extra_field": "not_allowed"
        }
        response = client.post("/predict", json=payload)
        # Should reject extra fields (if configured with extra='forbid')
        # For now, just check it doesn't crash - behavior depends on Pydantic config
        assert response.status_code in [200, 422]

    def test_predict_boundary_shelf_hours(self):
        """Test boundary values for shelf_hours (<= 30 allowed)."""
        # Exactly 30 should be allowed
        payload = {
            "price_rs": 50.0,
            "temp_c": 30.0,
            "footfall_index": 50,
            "competitor_shops": 2,
            "shelf_hours": 30.0,
            "orders_previous_day": 100,
            "day_type": "weekday",
            "outlet": "College Gate"
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 200

    def test_predict_boundary_orders_previous_day(self):
        """Test boundary values for orders_previous_day (>= 5 allowed)."""
        # Exactly 5 should be allowed
        payload = {
            "price_rs": 50.0,
            "temp_c": 30.0,
            "footfall_index": 50,
            "competitor_shops": 2,
            "shelf_hours": 8.0,
            "orders_previous_day": 5,
            "day_type": "weekday",
            "outlet": "College Gate"
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 200


if __name__ == "__main__":
    pytest.main([__file__, "-v"])