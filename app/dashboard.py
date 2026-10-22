"""ChurnGuard dashboard for retention teams.

Run locally:
    streamlit run app/dashboard.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))  # lets Streamlit Cloud import the package without install

from churnguard.data import CATEGORY_VALUES, SchemaError  # noqa: E402
from churnguard.predict import ChurnPredictor  # noqa: E402

MODEL_PATH = Path(os.getenv("MODEL_PATH", ROOT / "models" / "churn_model.joblib"))
SAMPLE_PATH = ROOT / "data" / "sample" / "sample_customers.csv"
REPORTS = ROOT / "reports"
RISK_COLORS = {"high": "#e03131", "medium": "#f08c00", "low": "#2f9e44"}

st.set_page_config(page_title="ChurnGuard", page_icon="📉", layout="wide")


@st.cache_resource
def load_predictor() -> ChurnPredictor:
    return ChurnPredictor.from_path(MODEL_PATH)


@st.cache_data
def score(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    predictor = load_predictor()
    results = predictor.predict(df)
    contributions = predictor.contributions(df)
    scored = df.copy().reset_index(drop=True)
    scored["churn_probability"] = [r["churn_probability"] for r in results]
    scored["risk_level"] = [r["risk_level"] for r in results]
    scored["will_churn"] = [r["will_churn"] for r in results]
    scored["suggested_action"] = [r["suggested_action"] for r in results]
    scored["top_reason"] = [
        next((x["feature"] for x in r["top_reasons"] if x["direction"] == "increases_risk"),
             "none")
        for r in results
    ]
    return scored, contributions.reset_index(drop=True)


def contribution_chart(contrib: pd.Series, title: str):
    top = contrib.reindex(contrib.abs().sort_values(ascending=False).index).head(8)[::-1]
    frame = pd.DataFrame({"feature": top.index, "impact": top.values})
    frame["effect"] = frame["impact"].map(lambda v: "raises risk" if v > 0 else "lowers risk")
    fig = px.bar(frame, x="impact", y="feature", orientation="h", color="effect",
                 color_discrete_map={"raises risk": "#e03131", "lowers risk": "#2f9e44"},
                 title=title)
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=40, b=10), yaxis_title=None,
                      xaxis_title="SHAP impact on churn (log-odds)", legend_title=None)
    return fig


# ---------------------------------------------------------------- sidebar
st.sidebar.title("📉 ChurnGuard")
st.sidebar.caption("Find customers likely to leave, see why, and act before they do.")

if not MODEL_PATH.exists():
    st.error("No trained model found. Run `python -m churnguard.train` first.")
    st.stop()

predictor = load_predictor()
uploaded = st.sidebar.file_uploader("Upload customers CSV", type="csv")
if uploaded is not None:
    customers = pd.read_csv(uploaded, dtype=str, keep_default_na=False)
    source = uploaded.name
else:
    customers = pd.read_csv(SAMPLE_PATH, dtype=str, keep_default_na=False)
    source = "sample customers (held-out test set)"
st.sidebar.write(f"**Data:** {source}")
st.sidebar.write(f"**Model:** {predictor.metadata.get('best_model', 'unknown')}")
st.sidebar.write(f"**Decision threshold:** {predictor.threshold:.2f}")

try:
    scored, contributions = score(customers)
except SchemaError as exc:
    st.error(f"The file does not match the expected format: {exc}")
    st.stop()

tab_overview, tab_customer, tab_whatif, tab_model = st.tabs(
    ["Risk overview", "Customer drill-down", "What-if simulator", "Model performance"]
)

# ---------------------------------------------------------------- overview
with tab_overview:
    at_risk = scored[scored["will_churn"]]
    monthly_at_risk = pd.to_numeric(at_risk["MonthlyCharges"]).sum()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Customers scored", f"{len(scored):,}")
    c2.metric("Predicted to churn", f"{len(at_risk):,}", f"{len(at_risk) / len(scored):.0%}",
              delta_color="off")
    c3.metric("High risk", f"{(scored['risk_level'] == 'high').sum():,}")
    c4.metric("Monthly revenue at risk", f"${monthly_at_risk:,.0f}")

    left, right = st.columns([3, 2])
    with left:
        fig = px.histogram(scored, x="churn_probability", color="risk_level", nbins=30,
                           color_discrete_map=RISK_COLORS, title="Churn risk distribution")
        fig.add_vline(x=predictor.threshold, line_dash="dash",
                      annotation_text="decision threshold")
        fig.update_layout(height=340, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, width="stretch")
    with right:
        drivers = at_risk["top_reason"].value_counts().head(8).reset_index()
        drivers.columns = ["main driver", "customers"]
        fig = px.bar(drivers, x="customers", y="main driver", orientation="h",
                     title="Main churn driver among at-risk customers")
        fig.update_layout(height=340, margin=dict(l=10, r=10, t=40, b=10),
                          yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, width="stretch")

    st.subheader("Customers ranked by churn risk")
    columns = ["customerID", "churn_probability", "risk_level", "Contract", "tenure",
               "MonthlyCharges", "top_reason", "suggested_action"]
    table = scored.sort_values("churn_probability", ascending=False)[columns]
    st.dataframe(
        table, width="stretch", hide_index=True, height=380,
        column_config={"churn_probability": st.column_config.ProgressColumn(
            "churn probability", min_value=0.0, max_value=1.0, format="%.2f")},
    )
    st.download_button("Download predictions as CSV", table.to_csv(index=False),
                       "churn_predictions.csv", "text/csv")

# ---------------------------------------------------------------- drill-down
with tab_customer:
    ranked = scored.sort_values("churn_probability", ascending=False)

    def label(i: int) -> str:
        r = ranked.loc[i]
        return f"{r['customerID']} - {r['churn_probability']:.0%} ({r['risk_level']})"

    choice = st.selectbox("Choose a customer (sorted by risk)", ranked.index, format_func=label)
    row = scored.loc[choice]
    c1, c2, c3 = st.columns(3)
    c1.metric("Churn probability", f"{row['churn_probability']:.0%}")
    c2.metric("Risk level", row["risk_level"].title())
    c3.metric("Prediction", "Will churn" if row["will_churn"] else "Will stay")
    st.info(f"**Suggested action:** {row['suggested_action']}")
    left, right = st.columns([3, 2])
    with left:
        st.plotly_chart(contribution_chart(contributions.loc[choice], "Why this prediction?"),
                        width="stretch")
    with right:
        details = customers.reset_index(drop=True).loc[choice].drop(labels=["Churn"],
                                                                   errors="ignore")
        st.dataframe(details.rename("value"), width="stretch", height=360)

# ---------------------------------------------------------------- what-if
with tab_whatif:
    st.write("Change a customer's details and see how the churn risk responds.")
    base = customers.reset_index(drop=True).loc[choice].to_dict()
    with st.form("whatif"):
        c1, c2, c3 = st.columns(3)
        contract = c1.selectbox("Contract", CATEGORY_VALUES["Contract"],
                                index=CATEGORY_VALUES["Contract"].index(base["Contract"]))
        payment = c1.selectbox("Payment method", CATEGORY_VALUES["PaymentMethod"],
                               index=CATEGORY_VALUES["PaymentMethod"].index(base["PaymentMethod"]))
        internet = c2.selectbox("Internet service", CATEGORY_VALUES["InternetService"],
                                index=CATEGORY_VALUES["InternetService"].index(
                                    base["InternetService"]))
        support = c2.selectbox("Tech support", CATEGORY_VALUES["TechSupport"],
                               index=CATEGORY_VALUES["TechSupport"].index(base["TechSupport"]))
        tenure = c3.slider("Tenure (months)", 0, 72, int(float(base["tenure"])))
        monthly = c3.slider("Monthly charges ($)", 18.0, 120.0,
                            float(min(max(float(base["MonthlyCharges"]), 18.0), 120.0)))
        submitted = st.form_submit_button("Recalculate risk")

    changed = {**base, "Contract": contract, "PaymentMethod": payment,
               "InternetService": internet, "TechSupport": support, "tenure": tenure,
               "MonthlyCharges": monthly, "TotalCharges": round(monthly * tenure, 2)}
    if internet == "No":
        for col in ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
                    "StreamingTV", "StreamingMovies"]:
            changed[col] = "No internet service"
    elif changed["TechSupport"] == "No internet service":
        changed["TechSupport"] = "No"
    before = row["churn_probability"]
    after = predictor.predict(pd.DataFrame([changed]))[0]
    c1, c2 = st.columns(2)
    c1.metric("Original risk", f"{before:.0%}")
    c2.metric("New risk", f"{after['churn_probability']:.0%}",
              f"{(after['churn_probability'] - before) * 100:+.1f} pts", delta_color="inverse")
    st.info(f"**Suggested action:** {after['suggested_action']}")

# ---------------------------------------------------------------- model
with tab_model:
    metrics_file = REPORTS / "metrics.json"
    if metrics_file.exists():
        summary = json.loads(metrics_file.read_text())
        test = summary["test_metrics"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Test ROC-AUC", f"{test['roc_auc']:.3f}")
        c2.metric("Recall (churners caught)", f"{test['recall']:.0%}")
        c3.metric("Precision", f"{test['precision']:.0%}")
        c4.metric(f"Churners in top {test['top_fraction']:.0%}",
                  f"{test['gain_top_fraction']:.0%}")
    comparison = REPORTS / "model_comparison.csv"
    if comparison.exists():
        st.subheader("Model comparison")
        st.table(pd.read_csv(comparison).set_index("model").round(4))
    figures = REPORTS / "figures"
    left, right = st.columns(2)
    for i, name in enumerate(["roc_pr_curves", "cumulative_gain", "shap_importance",
                              "confusion_matrix"]):
        path = figures / f"{name}.png"
        if path.exists():
            (left if i % 2 == 0 else right).image(str(path), width="stretch")
