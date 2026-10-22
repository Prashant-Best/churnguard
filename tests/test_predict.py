import numpy as np
import pandas as pd

from churnguard.data import FEATURE_COLUMNS
from churnguard.predict import NO_ACTION, RETENTION_PLAYBOOK, risk_level, suggest_action


def test_probabilities_are_valid(predictor, raw_customers):
    proba = predictor.predict_proba(raw_customers.head(30))
    assert proba.shape == (30,)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_predict_returns_reasons_and_action(predictor, raw_customers):
    results = predictor.predict(raw_customers.head(5), k=3)
    assert len(results) == 5
    for r in results:
        assert len(r["top_reasons"]) == 3
        assert r["risk_level"] in {"low", "medium", "high"}
        assert r["suggested_action"]
        assert r["customerID"].startswith("TEST-")


def test_shap_values_add_up_to_model_output(predictor, raw_customers):
    """SHAP's key property: base value + contributions = model log-odds."""
    data = predictor.prepare(raw_customers.head(20))
    X = data[FEATURE_COLUMNS]
    grouped, base = predictor.explainer.explain(X)
    proba = predictor.pipeline.predict_proba(X)[:, 1]
    log_odds = np.log(proba / (1 - proba))
    np.testing.assert_allclose(grouped.sum(axis=1).to_numpy() + base, log_odds, atol=1e-3)


def test_contract_drives_risk_in_synthetic_data(predictor, raw_customers):
    importance = predictor.explainer.global_importance(
        predictor.prepare(raw_customers)[FEATURE_COLUMNS]
    )
    assert "Contract" in importance.head(3).index or "tenure" in importance.head(3).index


def test_predict_without_customer_id(predictor, raw_customers):
    rows = raw_customers.head(2).drop(columns=["customerID", "Churn"])
    results = predictor.predict(rows)
    assert results[0]["customerID"] is None


def test_risk_levels():
    assert risk_level(0.9) == "high"
    assert risk_level(0.45) == "medium"
    assert risk_level(0.05) == "low"


def test_action_uses_first_risk_increasing_reason():
    reasons = [
        {"feature": "tenure", "direction": "decreases_risk"},
        {"feature": "Contract", "direction": "increases_risk"},
    ]
    assert suggest_action(reasons, 0.8) == RETENTION_PLAYBOOK["Contract"]
    assert suggest_action(reasons, 0.1) == NO_ACTION


def test_numeric_inputs_are_accepted(predictor, one_customer):
    results = predictor.predict(pd.DataFrame([one_customer]))
    assert 0 <= results[0]["churn_probability"] <= 1
