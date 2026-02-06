import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold

import lightgbm as lgb

# =========================
# Page
# =========================
st.set_page_config(page_title="Cash Sales Velocity — Prediction", layout="wide")
st.title("Cash Sales Velocity — Prediction Dashboard")
st.caption("This app trains the model on startup (cached) and predicts P(sell ≤30d) and P(sell ≤60d). No model files needed.")

DATA_PATH = "ai_stats_clean_for_velocity.csv"  # file in same repo as app.py

# =========================
# Load + prepare data
# =========================
@st.cache_data
def load_data(path):
    df = pd.read_csv(path)

    # basic columns
    df["PURCHASE DATE"] = pd.to_datetime(df["PURCHASE DATE"], errors="coerce")
    df["SALE DATE - start"] = pd.to_datetime(df["SALE DATE - start"], errors="coerce")

    df["total_cost"] = pd.to_numeric(df["Total Purchase Price"], errors="coerce")
    df["sale_price"] = pd.to_numeric(df["Cash Sales Price - amount"], errors="coerce")
    df["acres"] = pd.to_numeric(df["Acres"], errors="coerce")

    df["days_to_sale"] = (df["SALE DATE - start"] - df["PURCHASE DATE"]).dt.days
    df["multiple"] = df["sale_price"] / df["total_cost"]

    df["county_state"] = df["County, State"].fillna("Unknown").astype(str)
    df["city"] = df["Property Location or City"].fillna("Unknown").astype(str)

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["days_to_sale", "multiple", "total_cost", "sale_price"])
    df = df[df["days_to_sale"] >= 0].copy()

    # targets
    df["sell_30d"] = (df["days_to_sale"] <= 30).astype(int)
    df["sell_60d"] = (df["days_to_sale"] <= 60).astype(int)

    # feature engineering (numeric-only so LightGBM is happy)
    df["log_cost"] = np.log1p(df["total_cost"])
    df["sale_price_implied"] = df["multiple"] * df["total_cost"]
    df["cost_per_acre"] = df["total_cost"] / df["acres"].replace(0, np.nan)
    df["sale_per_acre"] = df["sale_price_implied"] / df["acres"].replace(0, np.nan)
    df["profit_pct"] = (df["sale_price_implied"] - df["total_cost"]) / df["total_cost"]

    # frequency encoding for categories (turn text into numeric)
    county_freq = df["county_state"].value_counts(normalize=True)
    city_freq = df["city"].value_counts(normalize=True)
    df["county_freq"] = df["county_state"].map(county_freq).fillna(0.0)
    df["city_freq"] = df["city"].map(city_freq).fillna(0.0)

    # county/city historical speed (numeric aggregates)
    county_avg = df.groupby("county_state")["days_to_sale"].mean()
    city_avg = df.groupby("city")["days_to_sale"].mean()
    df["county_avg_days"] = df["county_state"].map(county_avg).fillna(df["days_to_sale"].mean())
    df["city_avg_days"] = df["city"].map(city_avg).fillna(df["days_to_sale"].mean())

    # fill numeric NaNs
    num_cols = [
        "multiple","log_cost","acres","cost_per_acre","sale_per_acre","profit_pct",
        "county_freq","city_freq","county_avg_days","city_avg_days"
    ]
    df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)
    df[num_cols] = df[num_cols].fillna(0)

    features = num_cols
    return df, features, county_freq, city_freq, county_avg, city_avg, float(df["days_to_sale"].mean())

df, FEATURES, county_freq, city_freq, county_avg, city_avg, global_avg_days = load_data(DATA_PATH)

# =========================
# Train models (cached)
# =========================
@st.cache_resource
def train_models(df, features):
    X = df[features].copy()

    # time split: last 20% test by purchase date
    df_sorted = df.sort_values("PURCHASE DATE").reset_index(drop=True)
    X = df_sorted[features].copy()
    y30 = df_sorted["sell_30d"].values
    y60 = df_sorted["sell_60d"].values

    split = int(len(df_sorted) * 0.80)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y30_train, y30_test = y30[:split], y30[split:]
    y60_train, y60_test = y60[:split], y60[split:]

    def fit_calibrated(Xtr, ytr, Xte, yte, seed=42):
        pos = ytr.mean()
        spw = (1 - pos) / max(pos, 1e-6)

        base = lgb.LGBMClassifier(
            n_estimators=3000,
            learning_rate=0.03,
            num_leaves=64,
            min_child_samples=30,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.2,
            reg_lambda=0.6,
            random_state=seed,
            n_jobs=-1,
            scale_pos_weight=spw
        )

        base.fit(
            Xtr, ytr,
            eval_set=[(Xte, yte)],
            eval_metric="auc",
            callbacks=[lgb.early_stopping(200, verbose=False)]
        )

        cal = CalibratedClassifierCV(
            estimator=base,
            method="sigmoid",
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        )
        cal.fit(Xtr, ytr)

        p = cal.predict_proba(Xte)[:, 1]
        metrics = {
            "AUC": float(roc_auc_score(yte, p)),
            "AP": float(average_precision_score(yte, p)),
            "Brier": float(brier_score_loss(yte, p)),
            "base_rate": float(yte.mean())
        }
        return cal, metrics

    cal30, m30 = fit_calibrated(X_train, y30_train, X_test, y30_test, seed=42)
    cal60, m60 = fit_calibrated(X_train, y60_train, X_test, y60_test, seed=43)

    return cal30, cal60, m30, m60

cal30, cal60, m30, m60 = train_models(df, FEATURES)

# =========================
# Show model metrics
# =========================
st.subheader("Model performance (holdout last 20%)")
c1, c2, c3, c4 = st.columns(4)
c1.metric("≤30d AUC", f"{m30['AUC']:.3f}")
c2.metric("≤30d AP", f"{m30['AP']:.3f}")
c3.metric("≤60d AUC", f"{m60['AUC']:.3f}")
c4.metric("≤60d AP", f"{m60['AP']:.3f}")

st.divider()

# =========================
# Sidebar inputs
# =========================
st.sidebar.header("Property Inputs")

county_options = sorted(df["county_state"].unique())
city_options = sorted(df["city"].unique())

county_state = st.sidebar.selectbox("County, State", county_options, index=0)
city = st.sidebar.selectbox("City", city_options, index=0)

acres = st.sidebar.number_input("Acres", min_value=0.0, value=2.5, step=0.1)
total_cost = st.sidebar.number_input("Total Purchase Price (Cost)", min_value=100.0, value=6000.0, step=100.0)

st.sidebar.subheader("Pricing Scan (multiple)")
m_min = st.sidebar.number_input("Min multiple", min_value=0.5, value=1.5, step=0.1)
m_max = st.sidebar.number_input("Max multiple", min_value=0.6, value=6.0, step=0.1)
m_step = st.sidebar.number_input("Step", min_value=0.05, value=0.1, step=0.05)

target_p60 = st.sidebar.slider("Target P(sell ≤60d)", 0.0, 1.0, 0.70, 0.01)
target_p30 = st.sidebar.slider("Target P(sell ≤30d)", 0.0, 1.0, 0.60, 0.01)
enforce_30 = st.sidebar.checkbox("Also enforce ≤30d target", value=False)

# =========================
# Build prediction rows
# =========================
def make_features(county_state, city, acres, total_cost, multiple):
    total_cost = float(total_cost)
    acres = float(acres)
    multiple = float(multiple)

    sale_price = multiple * total_cost

    log_cost = np.log1p(total_cost)
    cost_per_acre = total_cost / (acres if acres > 0 else np.nan)
    sale_per_acre = sale_price / (acres if acres > 0 else np.nan)
    profit_pct = (sale_price - total_cost) / total_cost if total_cost > 0 else 0.0

    county_f = float(county_freq.get(county_state, 0.0))
    city_f = float(city_freq.get(city, 0.0))

    county_avg_days_val = float(county_avg.get(county_state, global_avg_days))
    city_avg_days_val = float(city_avg.get(city, global_avg_days))

    row = {
        "multiple": multiple,
        "log_cost": log_cost,
        "acres": acres,
        "cost_per_acre": 0.0 if np.isnan(cost_per_acre) else float(cost_per_acre),
        "sale_per_acre": 0.0 if np.isnan(sale_per_acre) else float(sale_per_acre),
        "profit_pct": float(profit_pct),
        "county_freq": county_f,
        "city_freq": city_f,
        "county_avg_days": county_avg_days_val,
        "city_avg_days": city_avg_days_val,
    }
    return row

multiples = np.arange(m_min, m_max + 1e-9, m_step)
rows = [make_features(county_state, city, acres, total_cost, m) for m in multiples]
X_scan = pd.DataFrame(rows)[FEATURES].fillna(0)

p30 = cal30.predict_proba(X_scan)[:, 1]
p60 = cal60.predict_proba(X_scan)[:, 1]

pred = pd.DataFrame({
    "multiple": multiples,
    "sale_price_est": multiples * float(total_cost),
    "P_sell_30d": p30,
    "P_sell_60d": p60
})

# Recommendation
mask = pred["P_sell_60d"] >= target_p60
if enforce_30:
    mask &= pred["P_sell_30d"] >= target_p30

feasible = pred[mask].copy()
rec = feasible.sort_values("multiple").iloc[-1] if len(feasible) else None

# =========================
# Output
# =========================
k1, k2, k3, k4 = st.columns(4)
k1.metric("Cost", f"${total_cost:,.0f}")
k2.metric("Acres", f"{acres:,.2f}")
k3.metric("County avg days", f"{float(county_avg.get(county_state, global_avg_days)):.1f}")
k4.metric("City avg days", f"{float(city_avg.get(city, global_avg_days)):.1f}")

if rec is None:
    st.warning("No multiple in your scan meets the probability target(s). Try lowering targets or scanning lower multiples.")
else:
    st.success(
        f"Recommended MAX multiple: **{rec['multiple']:.2f}x** "
        f"(sale ≈ **${rec['sale_price_est']:,.0f}**) "
        f"| P30=**{rec['P_sell_30d']:.2f}** P60=**{rec['P_sell_60d']:.2f}**"
    )

left, right = st.columns(2)
with left:
    fig = px.line(pred, x="multiple", y="P_sell_60d", title="P(sell ≤60d) vs multiple")
    fig.update_yaxes(range=[0, 1])
    st.plotly_chart(fig, use_container_width=True)

with right:
    fig = px.line(pred, x="multiple", y="P_sell_30d", title="P(sell ≤30d) vs multiple")
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
