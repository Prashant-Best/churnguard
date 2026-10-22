"""Feature engineering and preprocessing, packaged as scikit-learn objects.

Putting everything inside one Pipeline means the exact same transformations run during
cross-validation, final training and live prediction. That prevents data leakage and
"training/serving skew", two of the most common bugs in real ML systems.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churnguard.data import BINARY_COLUMNS, CATEGORY_VALUES, NUMERIC_COLUMNS

SERVICE_COLUMNS = [
    "PhoneService",
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

TENURE_BINS = [-1, 12, 24, 48, 72, np.inf]
TENURE_LABELS = ["0-1y", "1-2y", "2-4y", "4-6y", "6y+"]

ENGINEERED_NUMERIC = ["num_services", "avg_monthly_spend", "charge_increase", "auto_payment"]
ENGINEERED_CATEGORICAL = ["tenure_bucket"]

NUMERIC_FEATURES = NUMERIC_COLUMNS + BINARY_COLUMNS + ENGINEERED_NUMERIC
CATEGORICAL_FEATURES = list(CATEGORY_VALUES) + ENGINEERED_CATEGORICAL


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Adds business-meaningful features. It is stateless, so fit() learns nothing."""

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        # How many paid services the customer uses (more services = more "locked in").
        X["num_services"] = (X[SERVICE_COLUMNS] == "Yes").sum(axis=1) + (
            X["InternetService"] != "No"
        ).astype(int)
        # Average historical bill. clip(lower=1) avoids dividing by zero for brand-new customers.
        X["avg_monthly_spend"] = X["TotalCharges"] / X["tenure"].clip(lower=1)
        # Positive when the customer now pays more than their historical average (a price rise).
        X["charge_increase"] = X["MonthlyCharges"] - X["avg_monthly_spend"]
        X["auto_payment"] = X["PaymentMethod"].str.contains("automatic", regex=False).astype(int)
        X["tenure_bucket"] = pd.cut(X["tenure"], bins=TENURE_BINS, labels=TENURE_LABELS).astype(
            str
        )
        X["SeniorCitizen"] = X["SeniorCitizen"].astype(float)
        return X


def build_preprocessor() -> ColumnTransformer:
    """Scale numbers and one-hot encode categories. Unknown categories are ignored, not fatal."""
    categories = [CATEGORY_VALUES[c] for c in CATEGORY_VALUES] + [TENURE_LABELS]
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(
                    categories=categories, handle_unknown="ignore", sparse_output=False
                ),
                CATEGORICAL_FEATURES,
            ),
        ],
        verbose_feature_names_out=False,
    )


def build_pipeline(model) -> Pipeline:
    """Full pipeline: raw cleaned columns in, churn probability out."""
    return Pipeline(
        steps=[
            ("features", FeatureEngineer()),
            ("preprocess", build_preprocessor()),
            ("model", model),
        ]
    )


def transformed_feature_groups(preprocessor: ColumnTransformer) -> list[str]:
    """For each column the model sees, return the original feature it came from.

    One-hot encoding turns "Contract" into three columns. To explain a prediction in
    business language we add their contributions back together under "Contract".
    """
    groups: list[str] = list(NUMERIC_FEATURES)
    encoder: OneHotEncoder = preprocessor.named_transformers_["cat"]
    for col, cats in zip(CATEGORICAL_FEATURES, encoder.categories_, strict=True):
        groups.extend([col] * len(cats))
    return groups
