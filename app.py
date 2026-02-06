import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="Cash Sales Velocity - Prediction", layout="wide")
st.title("Cash Sales Velocity — Prediction Dashboard")
st.caption("Predict probability of selling within ≤30 days / ≤60 days based on pricing multiple + property context.")

# -----------------------------
# Paths (repo layout)
# -----------------------------
DATA_PATH = "data/ai_stats_clean_for_velocity.csv"
CAL30_PATH = "artifacts/cal30.pkl"
CAL60_PATH = "artifacts/cal60.pkl"
FEATURES_PATH = "artifacts/features.json"

# -----------------------------
# Load artifacts
# -----------------------------
@st.cache_resource
def load_models():
    cal30 = joblib.load(CAL30_PATH)
    cal60 = joblib.load(CAL60_PATH)
    with open(FEATURES_PATH, "r") as f:
        meta = json.load(f)
    return cal30, cal60, meta

@st.cache_data
def load_reference_data():
    df = pd.read_csv(DATA_PATH)

    # parse / standardize
    df["PURCHASE DATE"] = pd.to_datetime(df["PURCHASE DATE"], errors="coerce")
    df["SALE DATE - start"] = pd.to_datetime(df["SALE DATE - start"], errors="coerce")

    df["total_cost"] = pd.to_numeric(df["Total Purchase Price"], errors="coerce")
    df["sale_price"] = pd.to_numeric(df["Cash Sales Price - amount"], errors="coerce")
    df["acres"] = pd.to_numeric(df["Acres"], errors="coerce")

    df["days_to_sale"] = (df["SALE DATE - start"] - df["PURCHASE DATE"]).dt.days
    df["multiple"] = df["sale_price"] / df["total_cost"]

    df["County, State"] = df["County, State"].fillna("Unknown")
    df["Property Location or City"] = df["Property Location or City"].fillna("Unknown")

    # filter basic
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["days_to_sale", "multiple", "total_cost", "sale_price"])
    df = df[df["days_to_sale"] >= 0].copy()

    # aggregates that were used in training
    df["county_avg_days"] = df.groupby("County, State")["days_to_sale"].transform("mean")
    df["city_avg_days"] = df.groupby("Property Location or City")["days_to_sale"].transform("mean")
    df["county_count"] = df.groupby("County, State")["days_to_sale"].transform("count")
    df["city_count"] = df.groupby("Property Location or City")["days_to_sale"].transform("count")

    # Create lookup tables for Streamlit runtime
    county_stats = df.groupby("County, State").agg(
        county_avg_days=("days_to_sale", "mean"),
        county_count=("days_to_sale", "count")
    ).reset_index()

    city_stats = df.groupby("Property Location or City").agg(
        city_avg_days=("days_to_sale", "mean"),
        city_count=("days_to_sale", "count")
    ).reset_index()

    # global fallback values if a county/city isn't in history
    global_avg_days = float(df["days_to_sale"].mean())
    global_count = int(df.shape[0])

    return df, county_stats, city_stats, global_avg_days, global_count

cal30, cal60, meta = load_models()
FEATURES = meta["features"]

df_ref, county_stats, city_stats, global_avg_days, global_count = load_reference_data()

county_list = sorted(df_ref["County, State"].dropna().unique().tolist())
city_list = sorted(df_ref["Property Location or City"].dropna().unique().tolist())

# -----------------------------
# Feature engineering (MUST match training)
# -----------------------------
def build_features(county_state, city, acres, total_cost, multiple):
    acres = float(acres)
    total_cost = float(total_cost)
    multiple = float(multiple)

    # derived money values
    sale_price = multiple * total_cost

    log_cost = np.log1p(total_cost)
    cost_per_acre = total_cost / (acres if acres > 0 else np.nan)
    sale_per_acre = sale_price / (acres if acres > 0 else np.nan)
    profit_pct = (sale_price - total_cost) / total_cost if total_cost > 0 else np.nan

    # lookup county/city aggregates
    c_row = county_stats[county_stats["County, State"] == county_state]
    if len(c_row):
        county_avg_days = float(c_row["county_avg_days"].iloc[0])
        county_count = int(c_row["county_count"].iloc[0])
    else:
        county_avg_days = global_avg_days
        county_count = 0

    ct_row = city_stats[city_stats["Property Location or City"] == city]
    if len(ct_row):
        city_avg_days = float(ct_row["city_avg_days"].iloc[0])
        city_count = int(ct_row["city_count"].iloc[0])
    else:
        city_avg_days = global_avg_days
        city_count = 0

    # month is unknown at prediction time → let user choose it
    # (handled outside)
    return {
        "multiple": multiple,
        "log_cost": log_cost,
        "acres": acres,
        "cost_per_acre": cost_per_acre,
        "sale_per_acre": sale_per_acre,
        "profit_pct": profit_pct,
        "county_avg_days": county_avg_days,
        "city_avg_days": city_avg_days,
        "county_count": county_count,
        "city_count": city_count,
    }

# -----------------------------
# Sidebar inputs
# -----------------------------
st.sidebar.header("Property Inputs")

county_state = st.sidebar.selectbox("County, State", options=county_list, index=0)
city = st.sidebar.selectbox("City", options=city_list, index=0)

acres = st.sidebar.number_input("Acres", min_value=0.0, value=2.5, step=0.1)
total_cost = st.sidebar.number_input("Total Purchase Price (Cost)", min_value=100.0, value=6000.0, step=100.0)

purchase_month = st.sidebar.slider("Purchase month (1–12)", 1, 12, 10)

st.sidebar.subheader("Pricing Scan (Multiple)")
m_min = st.sidebar.number_input("Min multiple", min_value=0.5, value=1.5, step=0.1)
m_max = st.sidebar.number_input("Max multiple", min_value=0.6, value=6.0, step=0.1)
m_step = st.sidebar.number_input("Step", min_value=0.05, value=0.1, step=0.05)

st.sidebar.subheader("Decision Targets")
target_p60 = st.sidebar.slider("Target P(sell ≤60d)", 0.0, 1.0, 0.60, 0.01)
target_p30 = st.sidebar.slider("Target P(sell ≤30d)", 0.0, 1.0, 0.45, 0.01)
use_30_constraint = st.sidebar.checkbox("Also enforce 30-day target", value=False)

# -----------------------------
# Prediction scan
# -----------------------------
multiples = np.arange(m_min, m_max + 1e-9, m_step)

rows = []
for m in multiples:
    feats = build_features(county_state, city, acres, total_cost, m)
    feats["purchase_month"] = int(purchase_month)
    rows.append(feats)

X_scan = pd.DataFrame(rows)

# Ensure same column order as training
for col in FEATURES:
    if col not in X_scan.columns:
        X_scan[col] = np.nan
X_scan = X_scan[FEATURES].fillna(0)

p30 = cal30.predict_proba(X_scan)[:, 1]
p60 = cal60.predict_proba(X_scan)[:, 1]

pred = pd.DataFrame({
    "multiple": multiples,
    "sale_price_est": multiples * float(total_cost),
    "P_sell_30d": p30,
    "P_sell_60d": p60,
})

# -----------------------------
# Recommend best multiple under constraints
# -----------------------------
mask = pred["P_sell_60d"] >= target_p60
if use_30_constraint:
    mask &= pred["P_sell_30d"] >= target_p30

feasible = pred[mask].copy()
rec = feasible.sort_values("multiple").iloc[-1] if len(feasible) else None

# -----------------------------
# KPIs
# -----------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Cost", f"${total_cost:,.0f}")
c2.metric("Acres", f"{acres:,.2f}")
c3.metric("County avg days (history)", f"{build_features(county_state, city, acres, total_cost, m_min)['county_avg_days']:.1f}")
c4.metric("City avg days (history)", f"{build_features(county_state, city, acres, total_cost, m_min)['city_avg_days']:.1f}")

st.divider()

if rec is None:
    st.warning("No pricing multiple meets your probability target(s) in this scan range. Try lowering targets or scanning lower multiples.")
else:
    st.success(
        f"Recommended max multiple: **{rec['multiple']:.2f}x** "
        f"(sale ≈ **${rec['sale_price_est']:,.0f}**) "
        f"| P30=**{rec['P_sell_30d']:.2f}** P60=**{rec['P_sell_60d']:.2f}**"
    )

# -----------------------------
# Plots
# -----------------------------
left, right = st.columns(2)

with left:
    fig = px.line(
        pred,
        x="multiple",
        y="P_sell_60d",
        title="P(sell ≤60 days) vs multiple",
        labels={"multiple": "Markup multiple", "P_sell_60d": "Probability"},
    )
    fig.update_yaxes(range=[0, 1])
    st.plotly_chart(fig, use_container_width=True)

with right:
    fig = px.line(
        pred,
        x="multiple",
        y="P_sell_30d",
        title="P(sell ≤30 days) vs multiple",
        labels={"multiple": "Markup multiple", "P_sell_30d": "Probability"},
    )
    fig.update_yaxes(range=[0, 1])
    st.plotly_chart(fig, use_container_width=True)

st.markdown("### Prediction scan table")
st.dataframe(pred.round(3), use_container_width=True)

st.download_button(
    "Download prediction scan CSV",
    data=pred.to_csv(index=False).encode("utf-8"),
    file_name="prediction_scan.csv",
    mime="text/csv",
)

st.caption(
    "These are probabilistic predictions for pricing guidance (not guarantees). "
    "Model uses historical patterns (location speed averages + economics + pricing multiple)."
)
