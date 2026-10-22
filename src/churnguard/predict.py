"""Loading a trained model artifact and turning predictions into business actions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from churnguard.data import FEATURE_COLUMNS, ID_COLUMN, clean, validate
from churnguard.explain import ChurnExplainer

# Rule-based playbook: the strongest risk factor decides which retention offer to make.
RETENTION_PLAYBOOK: dict[str, str] = {
    "Contract": "Offer a discounted upgrade to a 1-year or 2-year contract.",
    "tenure": "Enrol the customer in the new-customer onboarding and loyalty programme.",
    "tenure_bucket": "Enrol the customer in the new-customer onboarding and loyalty programme.",
    "PaymentMethod": "Offer a bill credit for switching to automatic payment.",
    "auto_payment": "Offer a bill credit for switching to automatic payment.",
    "InternetService": "Run a service-quality check and review the internet plan price.",
    "MonthlyCharges": "Offer a personalised plan review or a bundle discount.",
    "avg_monthly_spend": "Offer a personalised plan review or a bundle discount.",
    "charge_increase": "Explain the recent price change and offer a loyalty discount.",
    "TotalCharges": "Offer a personalised plan review or a bundle discount.",
    "OnlineSecurity": "Offer a free 3-month trial of the online security and tech-support bundle.",
    "TechSupport": "Offer a free 3-month trial of the online security and tech-support bundle.",
    "OnlineBackup": "Offer a free 3-month trial of online backup and device protection.",
    "DeviceProtection": "Offer a free 3-month trial of online backup and device protection.",
    "PaperlessBilling": "Send a clear e-bill summary and check for billing confusion.",
    "num_services": "Offer a discounted bundle that adds one more service.",
}
DEFAULT_ACTION = "Ask the retention team to call the customer and understand their concerns."
NO_ACTION = "Low risk. No retention action needed; keep in regular engagement campaigns."

RISK_BANDS = [(0.6, "high"), (0.3, "medium"), (0.0, "low")]


def risk_level(probability: float) -> str:
    for cutoff, label in RISK_BANDS:
        if probability >= cutoff:
            return label
    return "low"


def suggest_action(reasons: list[dict], probability: float) -> str:
    if risk_level(probability) == "low":
        return NO_ACTION
    for reason in reasons:
        if reason["direction"] == "increases_risk" and reason["feature"] in RETENTION_PLAYBOOK:
            return RETENTION_PLAYBOOK[reason["feature"]]
    return DEFAULT_ACTION


def save_artifact(path: str | Path, pipeline, threshold: float, background: pd.DataFrame,
                  metadata: dict[str, Any]) -> Path:
    """Bundle everything needed at prediction time into one file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "pipeline": pipeline,
        "threshold": float(threshold),
        "background": background.reset_index(drop=True),
        "metadata": metadata,
    }
    joblib.dump(artifact, path, compress=3)
    return path


class ChurnPredictor:
    """The single entry point used by the API and the dashboard."""

    def __init__(self, artifact: dict[str, Any]):
        self.pipeline = artifact["pipeline"]
        self.threshold = artifact["threshold"]
        self.metadata = artifact.get("metadata", {})
        self.explainer = ChurnExplainer(self.pipeline, artifact["background"])

    @classmethod
    def from_path(cls, path: str | Path) -> ChurnPredictor:
        return cls(joblib.load(path))

    @staticmethod
    def prepare(df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate raw customer rows (the same steps used in training)."""
        data = clean(df.astype({c: str for c in df.columns if c in FEATURE_COLUMNS}))
        validate(data)
        return data

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        data = self.prepare(df)
        return self.pipeline.predict_proba(data[FEATURE_COLUMNS])[:, 1]

    def predict(self, df: pd.DataFrame, k: int = 3) -> list[dict[str, Any]]:
        """Probability, decision, risk band, top reasons and suggested action per customer."""
        data = self.prepare(df)
        X = data[FEATURE_COLUMNS]
        proba = self.pipeline.predict_proba(X)[:, 1]
        reasons = self.explainer.top_reasons(X, k=k)
        ids = data[ID_COLUMN] if ID_COLUMN in data.columns else pd.Series([None] * len(data))

        results = []
        for cid, p, r in zip(ids, proba, reasons, strict=True):
            results.append(
                {
                    "customerID": None if cid is None or pd.isna(cid) else str(cid),
                    "churn_probability": round(float(p), 4),
                    "will_churn": bool(p >= self.threshold),
                    "risk_level": risk_level(float(p)),
                    "threshold": round(self.threshold, 4),
                    "top_reasons": r,
                    "suggested_action": suggest_action(r, float(p)),
                }
            )
        return results

    def contributions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Per-feature SHAP contributions for every row (used by the dashboard charts)."""
        data = self.prepare(df)
        grouped, _ = self.explainer.explain(data[FEATURE_COLUMNS])
        return grouped
