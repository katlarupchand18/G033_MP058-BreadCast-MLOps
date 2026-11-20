"""FastAPI service for Bakery Bread Sales prediction.

Endpoints:
- GET /health: Returns {"status": "ok"}
- POST /predict: Accepts 8 input fields, returns prediction

Validation rules (client requirements):
- shelf_hours <= 30
- orders_previous_day >= 5
"""

import logging
import json
from typing import Any
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
import joblib
import numpy as np
import pandas as pd


# Configure JSON logging
class JSONFormatter(logging.Formatter):
    """Format log records as JSON."""

    def format(self, record: logging.LogRecord) -> str:
        log_obj = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
        }
        # Add extra fields if present
        for key, value in record.__dict__.items():
            if key not in ["name", "msg", "args", "created", "filename", "funcName",
                           "levelname", "levelno", "lineno", "module", "msecs",
                           "message", "name", "pathname", "process", "processName",
                           "relativeCreated", "thread", "threadName", "exc_info",
                           "exc_text", "stack_info", "getMessage"]:
                log_obj[key] = value
        return json.dumps(log_obj)


# Set up logger
logger = logging.getLogger("bakery_api")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(JSONFormatter())
logger.addHandler(handler)
logger.propagate = False


# Global model variable
model = None
model_loaded = False


def load_model():
    """Load the champion model from disk."""
    global model, model_loaded
    model_path = Path("models/model.joblib")
    if model_path.exists():
        model = joblib.load(model_path)
        model_loaded = True
        logger.info("Model loaded successfully", extra={"model_path": str(model_path)})
    else:
        logger.warning("Model file not found", extra={"model_path": str(model_path)})


# Load model at import time (for tests and direct usage)
load_model()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Model already loaded at import time
    yield
    # Cleanup if needed


app = FastAPI(
    title="Bakery Bread Sales Prediction API",
    description="Predict daily loaves sold for Indore bakery outlets",
    version="1.0.0",
    lifespan=lifespan
)


# Pydantic request/response models
class PredictionRequest(BaseModel):
    """Request schema for POST /predict with client validation rules."""

    price_rs: float = Field(..., description="Price of a loaf in rupees")
    temp_c: float = Field(..., description="Maximum temperature in Celsius")
    footfall_index: int = Field(..., description="Street footfall index", ge=0)
    competitor_shops: int = Field(..., description="Number of competitor bakeries within 500m", ge=0)
    shelf_hours: float = Field(..., description="Hours outlet was open", le=30, gt=0)
    orders_previous_day: int = Field(..., description="Loaves sold previous day", ge=5)
    day_type: str = Field(..., description="Type of day: weekday, weekend, or festival")
    outlet: str = Field(..., description="Outlet name: College Gate, Mall Road, or Station Road")

    @field_validator("day_type")
    @classmethod
    def validate_day_type(cls, v: str) -> str:
        allowed = {"weekday", "weekend", "festival"}
        if v not in allowed:
            raise ValueError(f"day_type must be one of {allowed}")
        return v

    @field_validator("outlet")
    @classmethod
    def validate_outlet(cls, v: str) -> str:
        allowed = {"College Gate", "Mall Road", "Station Road"}
        if v not in allowed:
            raise ValueError(f"outlet must be one of {allowed}")
        return v

    model_config = {"extra": "forbid"}  # Reject unexpected fields


class PredictionResponse(BaseModel):
    """Response schema for POST /predict."""

    prediction: float = Field(..., description="Predicted loaves sold")


class HealthResponse(BaseModel):
    """Response schema for GET /health."""

    status: str = "ok"


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(status="ok")


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest) -> PredictionResponse:
    """Predict loaves sold for given input features.

    Validates input against client rules:
    - shelf_hours <= 30 (enforced by Pydantic Field)
    - orders_previous_day >= 5 (enforced by Pydantic Field)
    - day_type and outlet must be valid categories
    - No extra fields allowed (extra='forbid')

    Returns 422 for invalid input before model inference.
    """
    if not model_loaded or model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Convert request to DataFrame for model
        input_df = pd.DataFrame([request.model_dump()])

        # Ensure column order matches training
        feature_order = [
            "price_rs", "temp_c", "footfall_index", "competitor_shops",
            "shelf_hours", "orders_previous_day", "day_type", "outlet"
        ]
        input_df = input_df[feature_order]

        # Make prediction
        prediction = model.predict(input_df)[0]

        # Ensure non-negative
        prediction = max(0.0, float(prediction))

        # Log prediction (structured JSON)
        logger.info(
            "Prediction made",
            extra={
                "input": request.model_dump(),
                "prediction": prediction
            }
        )

        return PredictionResponse(prediction=prediction)

    except Exception as e:
        logger.error("Prediction failed", extra={"error": str(e)})
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)