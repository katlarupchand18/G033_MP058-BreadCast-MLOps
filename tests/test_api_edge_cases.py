"""Additional edge-case tests for the BreadCast API."""

import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


@pytest.fixture
def valid_payload():
    return {
        "price_rs": 50.0,
        "temp_c": 30.0,
        "footfall_index": 50,
        "competitor_shops": 2,
        "shelf_hours": 8.0,
        "orders_previous_day": 100,
        "day_type": "weekday",
        "outlet": "College Gate"
    }


@pytest.mark.parametrize("field,value", [
    ("day_type", "holiday"),
    ("outlet", "Unknown Outlet"),
    ("footfall_index", -1),
    ("competitor_shops", -1),
    ("shelf_hours", 0),
    ("shelf_hours", -1),
    ("price_rs", "invalid"),
    ("temp_c", "invalid"),
])
def test_invalid_input_values(valid_payload, field, value):
    """Verify that invalid input values are rejected."""
    valid_payload[field] = value

    response = client.post("/predict", json=valid_payload)

    assert response.status_code == 422
    assert "detail" in response.json()
    assert any(
        field in error["loc"]
        for error in response.json()["detail"]
    )


def test_zero_footfall_allowed(valid_payload):
    """Verify that zero footfall is accepted."""
    valid_payload["footfall_index"] = 0

    response = client.post("/predict", json=valid_payload)

    assert response.status_code == 200
    assert "prediction" in response.json()


def test_zero_competitors_allowed(valid_payload):
    """Verify that zero competitor shops is accepted."""
    valid_payload["competitor_shops"] = 0

    response = client.post("/predict", json=valid_payload)

    assert response.status_code == 200
    assert "prediction" in response.json()