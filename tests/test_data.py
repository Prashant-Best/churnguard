import pandas as pd
import pytest

from churnguard.data import FEATURE_COLUMNS, SchemaError, clean, split_xy, validate


def test_blank_total_charges_become_zero_for_new_customers(raw_customers):
    df = clean(raw_customers)
    new = df[df["tenure"] == 0]
    assert len(new) > 0
    assert (new["TotalCharges"] == 0).all()
    assert df["TotalCharges"].notna().all()


def test_target_is_mapped_to_integers(clean_customers):
    assert set(clean_customers["Churn"].unique()) <= {0, 1}


def test_whitespace_is_stripped():
    raw = pd.DataFrame({"customerID": ["A"], "tenure": [" 5 "], "MonthlyCharges": ["10"],
                        "SeniorCitizen": ["0"], "TotalCharges": ["50"], "Contract": [" One year "]})
    df = clean(raw)
    assert df.loc[0, "Contract"] == "One year"
    assert df.loc[0, "tenure"] == 5


def test_validate_accepts_clean_data(clean_customers):
    validate(clean_customers, require_target=True)


def test_validate_rejects_unknown_category(clean_customers):
    bad = clean_customers.copy()
    bad.loc[0, "Contract"] = "Lifetime"
    with pytest.raises(SchemaError, match="Contract"):
        validate(bad)


def test_validate_rejects_missing_column(clean_customers):
    with pytest.raises(SchemaError, match="Missing columns"):
        validate(clean_customers.drop(columns=["tenure"]))


def test_validate_rejects_negative_numbers(clean_customers):
    bad = clean_customers.copy()
    bad.loc[0, "MonthlyCharges"] = -5
    with pytest.raises(SchemaError, match="negative"):
        validate(bad)


def test_duplicate_customers_are_dropped(raw_customers):
    doubled = pd.concat([raw_customers, raw_customers.head(5)])
    assert len(clean(doubled)) == len(raw_customers)


def test_split_xy_returns_only_feature_columns(clean_customers):
    X, y = split_xy(clean_customers)
    assert list(X.columns) == FEATURE_COLUMNS
    assert len(X) == len(y)
