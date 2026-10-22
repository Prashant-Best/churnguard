"""Loading, cleaning and validating the Telco churn data.

The raw file has a few known problems that are fixed here, in one place, so that training,
the API and the dashboard all see data in exactly the same shape:

* ``TotalCharges`` is stored as text and is blank for 11 brand-new customers (tenure 0).
* ``Churn`` is "Yes"/"No" and needs to become 1/0.
* Text values can carry stray whitespace.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

TARGET = "Churn"
ID_COLUMN = "customerID"

YES_NO = ["No", "Yes"]
INTERNET_ADDON = ["No", "Yes", "No internet service"]

# Every categorical column and the values it is allowed to take.
# The API schema and the validation below are both checked against this single source of truth.
CATEGORY_VALUES: dict[str, list[str]] = {
    "gender": ["Female", "Male"],
    "Partner": YES_NO,
    "Dependents": YES_NO,
    "PhoneService": YES_NO,
    "MultipleLines": ["No", "Yes", "No phone service"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": INTERNET_ADDON,
    "OnlineBackup": INTERNET_ADDON,
    "DeviceProtection": INTERNET_ADDON,
    "TechSupport": INTERNET_ADDON,
    "StreamingTV": INTERNET_ADDON,
    "StreamingMovies": INTERNET_ADDON,
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": YES_NO,
    "PaymentMethod": [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ],
}

NUMERIC_COLUMNS = ["tenure", "MonthlyCharges", "TotalCharges"]
BINARY_COLUMNS = ["SeniorCitizen"]
FEATURE_COLUMNS = list(CATEGORY_VALUES) + BINARY_COLUMNS + NUMERIC_COLUMNS


class SchemaError(ValueError):
    """Raised when incoming data does not match the expected columns or values."""


def load_raw(path: str | Path) -> pd.DataFrame:
    """Read the raw CSV. Everything is read as text first so cleaning is explicit."""
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy of the data with correct types."""
    df = df.copy()

    for col in df.columns:
        if pd.api.types.is_string_dtype(df[col]) or df[col].dtype == object:
            df[col] = df[col].astype(str).str.strip()

    df["tenure"] = pd.to_numeric(df["tenure"], errors="coerce")
    df["MonthlyCharges"] = pd.to_numeric(df["MonthlyCharges"], errors="coerce")
    df["SeniorCitizen"] = pd.to_numeric(df["SeniorCitizen"], errors="coerce").astype("Int64")

    # Blank TotalCharges only happens for customers in their first month (tenure 0):
    # they have not been billed yet, so the true value is 0. Any other gap is
    # estimated from the monthly charge.
    total = pd.to_numeric(df["TotalCharges"].replace("", None), errors="coerce")
    new_customer = total.isna() & (df["tenure"] == 0)
    total = total.mask(new_customer, 0.0)
    total = total.fillna(df["MonthlyCharges"] * df["tenure"])
    df["TotalCharges"] = total.astype(float)

    if TARGET in df.columns:
        df[TARGET] = df[TARGET].map({"Yes": 1, "No": 0, 1: 1, 0: 0, "1": 1, "0": 0})

    if ID_COLUMN in df.columns:
        df = df.drop_duplicates(subset=ID_COLUMN)

    return df.reset_index(drop=True)


def validate(df: pd.DataFrame, require_target: bool = False) -> None:
    """Fail loudly if columns are missing, values are unknown, or numbers are invalid."""
    required = FEATURE_COLUMNS + ([TARGET] if require_target else [])
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SchemaError(f"Missing columns: {missing}")

    problems = []
    for col, allowed in CATEGORY_VALUES.items():
        unknown = set(df[col].dropna().unique()) - set(allowed)
        if unknown:
            problems.append(f"{col} has unknown values {sorted(unknown)}")

    for col in NUMERIC_COLUMNS + BINARY_COLUMNS:
        if df[col].isna().any():
            problems.append(f"{col} has missing or non-numeric values")
        elif (df[col] < 0).any():
            problems.append(f"{col} has negative values")

    if require_target and not df[TARGET].isin([0, 1]).all():
        problems.append(f"{TARGET} must be Yes/No")

    if problems:
        raise SchemaError("; ".join(problems))


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separate model inputs from the target."""
    return df[FEATURE_COLUMNS].copy(), df[TARGET].astype(int)
