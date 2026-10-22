"""Shared test fixtures.

Tests use a small synthetic dataset with the same schema as the real one, so they run
in seconds on CI without downloading anything.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from churnguard.data import CATEGORY_VALUES, FEATURE_COLUMNS, clean, split_xy
from churnguard.features import build_pipeline
from churnguard.models import make_model
from churnguard.predict import ChurnPredictor, save_artifact


def make_raw_customers(n: int = 400, seed: int = 0) -> pd.DataFrame:
    """Raw-format rows (all text, like the CSV) where churn depends on contract and tenure."""
    rng = np.random.default_rng(seed)
    rows = {col: rng.choice(values, size=n) for col, values in CATEGORY_VALUES.items()}
    df = pd.DataFrame(rows)
    df["SeniorCitizen"] = rng.integers(0, 2, size=n)
    df["tenure"] = rng.integers(0, 73, size=n)
    df["MonthlyCharges"] = rng.uniform(18, 120, size=n).round(2)
    df["TotalCharges"] = (df["MonthlyCharges"] * df["tenure"]).round(2)

    logit = (
        -1.0
        + 1.8 * (df["Contract"] == "Month-to-month")
        - 0.04 * df["tenure"]
        + 0.6 * (df["PaymentMethod"] == "Electronic check")
    )
    churn = rng.random(n) < 1 / (1 + np.exp(-logit))
    df["Churn"] = np.where(churn, "Yes", "No")
    df.insert(0, "customerID", [f"TEST-{i:04d}" for i in range(n)])

    df = df.astype(str)
    df.loc[df["tenure"] == "0", "TotalCharges"] = " "  # mimic the blank values in the real file
    return df


@pytest.fixture(scope="session")
def raw_customers() -> pd.DataFrame:
    return make_raw_customers()


@pytest.fixture(scope="session")
def clean_customers(raw_customers) -> pd.DataFrame:
    return clean(raw_customers)


@pytest.fixture(scope="session")
def model_path(tmp_path_factory, clean_customers):
    """A small XGBoost model artifact trained on the synthetic data."""
    X, y = split_xy(clean_customers)
    pipeline = build_pipeline(make_model("xgboost", {"n_estimators": 60}))
    pipeline.fit(X, y)
    path = tmp_path_factory.mktemp("model") / "churn_model.joblib"
    save_artifact(path, pipeline, threshold=0.5, background=X.head(50),
                  metadata={"best_model": "xgboost", "version": "test"})
    return path


@pytest.fixture(scope="session")
def predictor(model_path) -> ChurnPredictor:
    return ChurnPredictor.from_path(model_path)


@pytest.fixture
def one_customer(raw_customers) -> dict:
    row = raw_customers.iloc[0][["customerID", *FEATURE_COLUMNS]].to_dict()
    row.update({"SeniorCitizen": 0, "tenure": 2, "MonthlyCharges": 85.5, "TotalCharges": 171.0,
                "Contract": "Month-to-month"})
    return row
