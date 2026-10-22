"""SHAP explanations, grouped back to business-level features.

SHAP splits a prediction into contributions from each feature: base value + contributions
= the model's raw output (log-odds of churn). Positive values push towards churn,
negative values push towards staying.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from churnguard.features import transformed_feature_groups


class ChurnExplainer:
    """Explains a fitted pipeline: features -> preprocess -> model."""

    def __init__(self, pipeline: Pipeline, background: pd.DataFrame):
        self.preprocess = Pipeline(pipeline.steps[:-1])
        self.model = pipeline.steps[-1][1]
        self.groups = transformed_feature_groups(pipeline.named_steps["preprocess"])
        self.feature_names = list(dict.fromkeys(self.groups))
        self._kind = type(self.model).__name__
        self._shap_explainer = None

        if self._kind != "XGBClassifier":
            import shap  # imported lazily: only needed for non-XGBoost models

            bg = self.preprocess.transform(background)
            if self._kind == "LogisticRegression":
                self._shap_explainer = shap.LinearExplainer(self.model, bg)
            else:
                self._shap_explainer = shap.TreeExplainer(self.model)

    def _raw_contributions(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """SHAP values per transformed column and the base value, in log-odds."""
        Z = self.preprocess.transform(X)
        if self._kind == "XGBClassifier":
            # XGBoost computes exact TreeSHAP values natively; the last column is the bias.
            import xgboost as xgb

            contribs = self.model.get_booster().predict(xgb.DMatrix(Z), pred_contribs=True)
            return contribs[:, :-1], contribs[:, -1]

        values = self._shap_explainer.shap_values(Z)
        base = self._shap_explainer.expected_value
        if isinstance(values, list):  # older SHAP returns one array per class
            values, base = values[1], base[1]
        values = np.asarray(values)
        if values.ndim == 3:  # (rows, features, classes)
            values = values[:, :, 1]
            base = np.asarray(base)[1]
        return values, np.full(len(Z), float(np.asarray(base).ravel()[-1]))

    def explain(self, X: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
        """Return (contributions grouped by original feature, base values)."""
        values, base = self._raw_contributions(X)
        frame = pd.DataFrame(values, columns=self.groups, index=X.index)
        grouped = frame.T.groupby(level=0, sort=False).sum().T
        return grouped[self.feature_names], base

    def global_importance(self, X: pd.DataFrame) -> pd.Series:
        """Mean absolute contribution of each feature across many customers."""
        grouped, _ = self.explain(X)
        return grouped.abs().mean().sort_values(ascending=False)

    def top_reasons(self, X: pd.DataFrame, k: int = 3) -> list[list[dict]]:
        """The k strongest drivers for each row, as plain dictionaries."""
        engineered = self.preprocess.named_steps["features"].transform(X)
        grouped, _ = self.explain(X)
        results = []
        for idx in grouped.index:
            row = grouped.loc[idx]
            strongest = row.reindex(row.abs().sort_values(ascending=False).index)[:k]
            reasons = []
            for feature, impact in strongest.items():
                value = engineered.at[idx, feature]
                if isinstance(value, (float, np.floating)):
                    value = round(float(value), 2)
                reasons.append(
                    {
                        "feature": feature,
                        "value": str(value),
                        "impact": round(float(impact), 4),
                        "direction": "increases_risk" if impact > 0 else "decreases_risk",
                    }
                )
            results.append(reasons)
        return results
