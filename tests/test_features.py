import numpy as np
import pandas as pd

from churnguard.data import split_xy
from churnguard.features import (
    FeatureEngineer,
    build_preprocessor,
    transformed_feature_groups,
)


def test_num_services_counts_yes_values_and_internet(clean_customers):
    X, _ = split_xy(clean_customers)
    out = FeatureEngineer().transform(X)
    row = out.iloc[0]
    services = ["PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
                "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
    expected = sum(row[c] == "Yes" for c in services) + int(row["InternetService"] != "No")
    assert row["num_services"] == expected


def test_average_spend_has_no_division_by_zero(clean_customers):
    X, _ = split_xy(clean_customers)
    out = FeatureEngineer().transform(X)
    assert np.isfinite(out["avg_monthly_spend"]).all()
    assert (out.loc[out["tenure"] == 0, "avg_monthly_spend"] == 0).all()


def test_tenure_buckets_cover_all_rows(clean_customers):
    X, _ = split_xy(clean_customers)
    out = FeatureEngineer().transform(X)
    assert set(out["tenure_bucket"]) <= {"0-1y", "1-2y", "2-4y", "4-6y", "6y+"}


def test_auto_payment_flag(clean_customers):
    X, _ = split_xy(clean_customers)
    out = FeatureEngineer().transform(X)
    automatic = out["PaymentMethod"].str.contains("automatic")
    assert (out["auto_payment"] == automatic.astype(int)).all()


def test_preprocessor_ignores_unseen_categories(clean_customers):
    X, _ = split_xy(clean_customers)
    engineered = FeatureEngineer().transform(X)
    pre = build_preprocessor().fit(engineered)
    odd = engineered.head(1).copy()
    odd["Contract"] = "Something new"
    out = pre.transform(odd)
    assert out.shape[1] == pre.transform(engineered.head(1)).shape[1]


def test_feature_groups_match_transformed_width(clean_customers):
    X, _ = split_xy(clean_customers)
    engineered = FeatureEngineer().transform(X)
    pre = build_preprocessor().fit(engineered)
    width = pre.transform(engineered).shape[1]
    groups = transformed_feature_groups(pre)
    assert len(groups) == width
    assert groups.count("Contract") == 3


def test_feature_engineer_does_not_modify_input(clean_customers):
    X, _ = split_xy(clean_customers)
    before = X.copy()
    FeatureEngineer().transform(X)
    pd.testing.assert_frame_equal(X, before)
