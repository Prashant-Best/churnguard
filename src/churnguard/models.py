"""Model definitions and the Optuna search space."""

from __future__ import annotations

from typing import Any

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

RANDOM_STATE = 42
CANDIDATE_MODELS = ["logistic_regression", "random_forest", "xgboost"]


def make_model(name: str, params: dict[str, Any] | None = None, scale_pos_weight: float = 1.0):
    """Create an untrained model.

    Every model compensates for class imbalance (about 26% of customers churn), so it
    does not simply learn to predict "no churn" for everyone.
    """
    params = dict(params or {})
    if name == "logistic_regression":
        return LogisticRegression(
            C=params.get("C", 1.0), class_weight="balanced", max_iter=2000
        )
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=params.get("n_estimators", 400),
            min_samples_leaf=params.get("min_samples_leaf", 5),
            max_features=params.get("max_features", "sqrt"),
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
    if name in {"xgboost", "xgboost_tuned"}:
        defaults = {
            "n_estimators": 300,
            "learning_rate": 0.05,
            "max_depth": 4,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 1.0,
            "reg_lambda": 1.0,
        }
        defaults.update(params)
        return XGBClassifier(
            **defaults,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
    raise ValueError(f"Unknown model: {name}")


def xgboost_search_space(trial) -> dict[str, Any]:
    """Hyperparameters Optuna is allowed to try for XGBoost."""
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 800, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_depth": trial.suggest_int("max_depth", 2, 7),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 10.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
    }
