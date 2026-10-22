from typing import get_args

import pytest
from fastapi.testclient import TestClient

from churnguard.data import CATEGORY_VALUES


@pytest.fixture(scope="module")
def client(model_path):
    import os

    os.environ["MODEL_PATH"] = str(model_path)
    from app.api import app

    with TestClient(app) as test_client:
        yield test_client


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_predict_single(client, one_customer):
    response = client.post("/predict", json=one_customer)
    assert response.status_code == 200
    body = response.json()
    assert 0 <= body["churn_probability"] <= 1
    assert len(body["top_reasons"]) == 3
    assert body["risk_level"] in {"low", "medium", "high"}


def test_predict_batch(client, one_customer):
    second = {**one_customer, "customerID": "B-2", "Contract": "Two year", "tenure": 60}
    response = client.post("/predict/batch", json={"customers": [one_customer, second]})
    assert response.status_code == 200
    assert [r["customerID"] for r in response.json()] == [one_customer["customerID"], "B-2"]


def test_unknown_category_is_rejected(client, one_customer):
    response = client.post("/predict", json={**one_customer, "Contract": "Lifetime"})
    assert response.status_code == 422


def test_negative_tenure_is_rejected(client, one_customer):
    response = client.post("/predict", json={**one_customer, "tenure": -3})
    assert response.status_code == 422


def test_missing_field_is_rejected(client, one_customer):
    body = dict(one_customer)
    body.pop("Contract")
    assert client.post("/predict", json=body).status_code == 422


def test_empty_batch_is_rejected(client):
    assert client.post("/predict/batch", json={"customers": []}).status_code == 422


def test_model_info(client):
    body = client.get("/model/info").json()
    assert "threshold" in body


def test_api_schema_matches_data_schema():
    """The API's allowed values must stay in sync with the training data schema."""
    from app.api import Customer

    for column, allowed in CATEGORY_VALUES.items():
        annotation = Customer.model_fields[column].annotation
        assert set(get_args(annotation)) == set(allowed), column
