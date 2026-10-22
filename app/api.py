"""ChurnGuard REST API.

Run locally:
    uvicorn app.api:app --reload
Interactive docs: http://localhost:8000/docs
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from churnguard import __version__
from churnguard.config import PROJECT_ROOT
from churnguard.data import SchemaError
from churnguard.predict import ChurnPredictor

logger = logging.getLogger("churnguard.api")

YesNo = Literal["Yes", "No"]
InternetAddon = Literal["Yes", "No", "No internet service"]

EXAMPLE_CUSTOMER = {
    "customerID": "7590-VHVEG",
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 1,
    "PhoneService": "No",
    "MultipleLines": "No phone service",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 29.85,
    "TotalCharges": 29.85,
}


class Customer(BaseModel):
    """One customer. Invalid values are rejected with a 422 error before reaching the model."""

    model_config = ConfigDict(json_schema_extra={"example": EXAMPLE_CUSTOMER})

    customerID: str | None = Field(default=None, max_length=64)
    gender: Literal["Female", "Male"]
    SeniorCitizen: Literal[0, 1]
    Partner: YesNo
    Dependents: YesNo
    tenure: int = Field(ge=0, le=120, description="Months with the company")
    PhoneService: YesNo
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: InternetAddon
    OnlineBackup: InternetAddon
    DeviceProtection: InternetAddon
    TechSupport: InternetAddon
    StreamingTV: InternetAddon
    StreamingMovies: InternetAddon
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: YesNo
    PaymentMethod: Literal[
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ]
    MonthlyCharges: float = Field(gt=0, le=1000)
    TotalCharges: float = Field(ge=0)


class BatchRequest(BaseModel):
    customers: list[Customer] = Field(min_length=1, max_length=1000)


class Reason(BaseModel):
    feature: str
    value: str
    impact: float = Field(description="SHAP contribution in log-odds; positive raises churn risk")
    direction: Literal["increases_risk", "decreases_risk"]


class Prediction(BaseModel):
    customerID: str | None
    churn_probability: float
    will_churn: bool
    risk_level: Literal["low", "medium", "high"]
    threshold: float
    top_reasons: list[Reason]
    suggested_action: str


def model_path() -> Path:
    path = Path(os.getenv("MODEL_PATH", "models/churn_model.joblib"))
    return path if path.is_absolute() else PROJECT_ROOT / path


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once at startup instead of on every request."""
    path = model_path()
    try:
        app.state.predictor = ChurnPredictor.from_path(path)
        logger.info("Loaded model from %s", path)
    except FileNotFoundError:
        app.state.predictor = None
        logger.error("Model file not found at %s. Run `python -m churnguard.train` first.", path)
    yield


app = FastAPI(
    title="ChurnGuard API",
    description="Predicts customer churn, explains each prediction with SHAP and suggests "
    "a retention action.",
    version=__version__,
    lifespan=lifespan,
)


def get_predictor() -> ChurnPredictor:
    predictor = getattr(app.state, "predictor", None)
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    return predictor


def run_prediction(customers: list[Customer]) -> list[dict]:
    predictor = get_predictor()
    frame = pd.DataFrame([c.model_dump() for c in customers])
    try:
        return predictor.predict(frame)
    except SchemaError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/health")
def health() -> dict:
    predictor = getattr(app.state, "predictor", None)
    return {
        "status": "ok" if predictor else "degraded",
        "model_loaded": predictor is not None,
        "model": predictor.metadata.get("best_model") if predictor else None,
        "version": __version__,
    }


@app.get("/model/info")
def model_info() -> dict:
    predictor = get_predictor()
    return {"threshold": predictor.threshold, **predictor.metadata}


@app.post("/predict", response_model=Prediction)
def predict(customer: Customer) -> dict:
    return run_prediction([customer])[0]


@app.post("/predict/batch", response_model=list[Prediction])
def predict_batch(request: BatchRequest) -> list[dict]:
    return run_prediction(request.customers)
