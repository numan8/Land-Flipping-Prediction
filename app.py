import os, re, json
import requests, joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

# ============================================================
# CONFIG (put your Drive share links here — no sidebar pasting)
# ============================================================
CAL30_LINK = "https://drive.google.com/file/d/1LmJ1Av5CEvMsJ4I85DrpxCeXO0tc4dN4/view?usp=sharing"
CAL60_LINK = "https://drive.google.com/file/d/1U6IfGFEAtOQXI0tHtAcWDHGM3p29eqjf/view?usp=sharing"
FEATURES_LINK = "https://drive.google.com/file/d/1H6hkyzbfe4aGkpWH4y8FJgcLp_kYLSo2/view?usp=sharing"

# This is only for dropdown options (county/city) and optional “typical” stats.
# If you don’t want any dataset in the app, you can remove this and use text inputs instead.
DATA_PATH = "ai_stats_clean_for_velocity.csv"

# ============================================================
# PAGE
# ============================================================
st.set_page_config(page_title="Cash Sales Velocity — Pricing Decision", layout="wide")

# Premium-ish CSS (client-facing look)
st.markdown(
    """
<style>
/* Background */
.stApp {
  background: radial-gradient(1200px 600px at 20% -10%, rgba(99,102,241,0.18), transparent 55%),
              radial-gradient(1200px 600px at 85% 0%, rgba(14,165,233,0.14), transparent 55%),
              linear-gradient(180deg, #fbfdff 0%, #f6f8ff 55%, #fbfdff 100%);
}

/* Title spacing */
.block-container { padding-top: 1.1rem; padding-bottom: 1.5rem; }

/* Metric cards */
[data-testid="stMetric"] {
  background: rgba(255,255,255,0.72);
  border: 1px solid rgba(15,23,42,0.08);
  padding: 14px 14px;
  border-radius: 16px;
  box-shadow: 0 10px 25px rgba(2,6,23,0.06);
}

/* Dataframe container */
[data-testid="stDataFrame"] {
  background: rgba(255,255,255,0.72);
  border-radius: 16px;
  border: 1px solid rgba(15,23,42,0.08);
  box-shadow: 0 10px 25px rgba(2,6,23,0.06);
}

/* Expander */
.streamlit-expanderHeader {
  background: rgba(255,255,255,0.72);
  border: 1px solid rgba(15,23,42,0.08);
  border-radius: 14px;
}
</style>
""",
    unsafe_allow_html=True
)

st.title("Cash Sales Velocity — Pricing Decision Dashboard")
st.caption(
    "Goal: pick a cash list price ceiling that maximizes **probability of selling fast (≤30 / ≤60 days)** while staying profitable and increasing compounding cycles."
)

# ============================================================
# HELPERS — Drive download
# ============================================================
def extract_file_id(url: str) -> str:
    m = re.search(r"/d/([a-zA-Z0-9_-]+)", url)
    if m:
        return m.group(1)
    m = re.search(r"id=([a-zA-Z0-9_-]+)", url)
    if m:
        return m.group(1)
    raise ValueError(f"Could not extract file id from: {url}")

CAL30_ID = extract_file_id(CAL30_LINK)
CAL60_ID = extract_file_id(CAL60_LINK)
FEATURES_ID = extract_file_id(FEATURES_LINK)

ART_DIR = "artifacts"
CAL30_PATH = os.path.join(ART_DIR, "cal30.pkl")
CAL60_PATH = os.path.join(ART_DIR, "cal60.pkl")
FEATURES_PATH = os.path.join(ART_DIR, "features.json")

def gdrive_download(file_id: str, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        return

    URL = "https://drive.google.com/uc?export=download"
    s = requests.Session()
    r = s.get(URL, params={"id": file_id}, stream=True)
    r.raise_for_status()

    # If HTML returned => token confirm or permissions issue
    if "text/html" in (r.headers.get("Content-Type", "")):
        token = None
        for k, v in r.cookies.items():
            if k.startswith("download_warning"):
                token = v
                break
        if token:
            r = s.get(URL, params={"id": file_id, "confirm": token}, stream=True)
            r.raise_for_status()
        else:
            raise RuntimeError(
                "Google Drive returned HTML (permissions issue). "
                "Set sharing to: Anyone with link → Viewer."
            )

    with open(out_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)

@st.cache_resource
def load_models():
    gdrive_download(CAL30_ID, CAL30_PATH)
    gdrive_download(CAL60_ID, CAL60_PATH)
    gdrive_download(FEATURES_ID, FEATURES_PATH)

    cal30 = joblib.load(CAL30_PATH)
    cal60 = joblib.load(CAL60_PATH)

    with open(FEATURES_PATH, "r") as f:
        meta = json.load(f)
    FEATURES = meta["features"] if isinstance(meta, dict) and "features" in meta else meta

    return cal30, cal60, FEATURES

cal30, cal60, FEATURES = load_models()

# ============================================================
# DATA FOR DROPDOWNS (County/City)
# ============================================================
@st.cache_data
def load_location_options(data_path: str):
    if not os.path.exists(data_path):
        # If you didn't ship the dataset, return empty lists (fallback)
        return [], [], None

    df = pd.read_csv(data_path)
    # Expecting these columns from your training renames
    # If your CSV still has "County, State" and "Property Location or City",
    # update here accordingly.
    possible_county_cols = ["county_state", "County, State"]
    possible_city_cols = ["city", "Property Location or City"]

    county_col = next((c for c in possible_county_cols if c in df.columns), None)
    city_col = next((c for c in possible_city_cols if c in df.columns), None)

    if county_col is None or city_col is None:
        return [], [], None

    df[county_col] = df[county_col].astype("string")
    df[city_col] = df[city_col].astype("string")

    counties = sorted(df[county_col].dropna().unique().tolist())
    cities = sorted(df[city_col].dropna().unique().tolist())
    return counties, cities, (df, county_col, city_col)

counties, cities, loc_pack = load_location_options(DATA_PATH)

# ============================================================
# SIDEBAR INPUTS (business inputs only)
# ============================================================
st.sidebar.header("Deal Inputs")

# County/City dropdowns (preferred)
if loc_pack is not None and len(counties) > 0:
    loc_df, county_col, city_col = loc_pack

    county_state = st.sidebar.selectbox("County, State", counties, index=0)

    filtered_cities = sorted(
        loc_df.loc[loc_df[county_col] == county_state, city_col]
        .dropna()
        .unique()
        .tolist()
    )
    if len(filtered_cities) == 0:
        filtered_cities = cities

    city = st.sidebar.selectbox("City", filtered_cities, index=0)

else:
    # Fallback if dataset not present
    st.sidebar.info("Dropdowns disabled (dataset not found). Using manual inputs.")
    county_state = st.sidebar.text_input("County, State", value="Los Angeles, CA")
    city = st.sidebar.text_input("City", value="Lancaster")

acres = st.sidebar.number_input("Acres", min_value=0.0, value=2.5, step=0.1)
total_cost = st.sidebar.number_input("Total Purchase Price (All-in)", min_value=1.0, value=6000.0, step=100.0)

st.sidebar.divider()
st.sidebar.subheader("Speed target")

mode = st.sidebar.radio("Target window", ["≤30 days", "≤60 days"], index=0)
target_prob = st.sidebar.slider("Target probability", 0.30, 0.95, 0.70, 0.01)

st.sidebar.divider()
st.sidebar.subheader("Pricing scan range")

m_min = st.sidebar.number_input("Min multiple", min_value=0.5, value=1.0, step=0.1)
m_max = st.sidebar.number_input("Max multiple", min_value=0.6, value=8.0, step=0.1)
m_step = st.sidebar.number_input("Step", min_value=0.05, value=0.1, step=0.05)

# ============================================================
# FINANCIAL ASSUMPTIONS (client-friendly)
# ============================================================
with st.sidebar.expander("Assumptions (commissions)"):
    AFFILIATE_PCT_SALE = st.number_input("Affiliate / Listing (% of Sale)", min_value=0.0, max_value=0.30, value=0.10, step=0.01)
    AGENT_PCT_PROFIT = st.number_input("Agents Total (% of Profit) (4%+4%)", min_value=0.0, max_value=0.30, value=0.08, step=0.01)

def estimate_net_profit(sale_price: float, total_cost: float) -> float:
    affiliate = AFFILIATE_PCT_SALE * sale_price
    profit_before_agents = sale_price - total_cost - affiliate
    agent_comm = AGENT_PCT_PROFIT * max(profit_before_agents, 0.0)
    net_profit = profit_before_agents - agent_comm
    return net_profit

# ============================================================
# MODEL FEATURE ROWS (must match training feature names)
# ============================================================
def make_feature_row(multiple: float) -> dict:
    return {
        "multiple": float(multiple),
        "total_cost": float(total_cost),
        "acres": float(acres),
        "county_state": str(county_state),
        "city": str(city),
    }

multiples = np.arange(m_min, m_max + 1e-9, m_step)
X = pd.DataFrame([make_feature_row(m) for m in multiples])

# Ensure all expected features exist
for col in FEATURES:
    if col not in X.columns:
        X[col] = np.nan
X = X[FEATURES].copy()

# Predict probabilities
p30 = cal30.predict_proba(X)[:, 1]
p60 = cal60.predict_proba(X)[:, 1]

pred = pd.DataFrame({
    "multiple": multiples,
    "sale_price": multiples * float(total_cost),
    "P_sell_30d": p30,
    "P_sell_60d": p60,
})

pred["net_profit"] = pred["sale_price"].apply(lambda s: estimate_net_profit(float(s), float(total_cost)))
pred["net_roi"] = np.where(float(total_cost) > 0, pred["net_profit"] / float(total_cost), np.nan)

# ============================================================
# DECISION LOGIC
# ============================================================
if mode == "≤30 days":
    use_prob_col = "P_sell_30d"
    days_label = "≤30d"
    target_days = 30
else:
    use_prob_col = "P_sell_60d"
    days_label = "≤60d"
    target_days = 60

feasible = pred[(pred[use_prob_col] >= target_prob)].copy()
rec = feasible.sort_values("multiple").iloc[-1] if len(feasible) else None

def decision_label(prob: float) -> str:
    if prob >= target_prob:
        return "GO"
    if prob >= max(0.50, target_prob - 0.10):
        return "CAUTION"
    return "NO-GO"

pred["decision"] = pred[use_prob_col].apply(decision_label)

# ============================================================
# HEADER: CLIENT-FACING DECISION OUTPUT
# ============================================================
st.subheader("Recommended price ceiling (decision output)")

k1, k2, k3, k4, k5 = st.columns(5)

if rec is None:
    k1.metric("Recommended max multiple", "—")
    k2.metric("Recommended list price", "—")
    k3.metric(f"P(sell {days_label})", "—")
    k4.metric("Est. net profit (after commissions)", "—")
    k5.metric("Est. net ROI", "—")
    st.warning(
        f"No price in your scan meets target **P(sell {days_label}) ≥ {target_prob:.0%}**. "
        f"Try scanning lower multiples, lowering target probability, or switching to the {('≤60 days' if mode=='≤30 days' else '≤30 days')} window."
    )
else:
    k1.metric("Recommended max multiple", f"{rec['multiple']:.2f}x")
    k2.metric("Recommended list price", f"${rec['sale_price']:,.0f}")
    k3.metric(f"P(sell {days_label})", f"{rec[use_prob_col]:.0%}")
    k4.metric("Est. net profit (after commissions)", f"${rec['net_profit']:,.0f}")
    k5.metric("Est. net ROI", f"{rec['net_roi']:.0%}")

st.caption(
    f"Decision rule: choose the **highest** multiple that still achieves **P(sell {days_label}) ≥ {target_prob:.0%}**. "
    "This maximizes price while protecting velocity/compounding."
)

st.divider()

# ============================================================
# CHARTS (clean + interpretable)
# ============================================================
c1, c2 = st.columns(2)

with c1:
    fig = px.line(
        pred, x="multiple", y=use_prob_col,
        title=f"Probability of selling {days_label} vs markup multiple"
    )
    fig.update_yaxes(range=[0, 1])
    st.plotly_chart(fig, use_container_width=True)

with c2:
    fig = px.line(
        pred, x="multiple", y="net_profit",
        title="Estimated net profit vs markup multiple (after commissions)"
    )
    st.plotly_chart(fig, use_container_width=True)

# Optional: “tradeoff curve” in one chart
st.markdown("### Speed vs Profit tradeoff")
trade = pred.copy()
trade["prob_target_window"] = trade[use_prob_col]
fig = px.scatter(
    trade,
    x="net_profit",
    y="prob_target_window",
    color="decision",
    hover_data=["multiple", "sale_price", "net_roi", "P_sell_30d", "P_sell_60d"],
    labels={"net_profit": "Net Profit ($)", "prob_target_window": f"P(sell {days_label})"},
    title="Pick a point that balances speed (probability) and profit"
)
fig.update_yaxes(range=[0, 1])
st.plotly_chart(fig, use_container_width=True)

# ============================================================
# DECISION TABLE (client-facing)
# ============================================================
st.subheader("Pricing options (decision table)")

show = pred.copy()
show["P_sell_30d"] = (show["P_sell_30d"] * 100).round(1)
show["P_sell_60d"] = (show["P_sell_60d"] * 100).round(1)
show["net_profit"] = show["net_profit"].round(0)
show["net_roi"] = (show["net_roi"] * 100).round(1)
show["sale_price"] = show["sale_price"].round(0)

# Show top rows around recommended point for readability
if rec is not None:
    rec_idx = int(np.argmin(np.abs(show["multiple"].values - float(rec["multiple"]))))
    start = max(0, rec_idx - 10)
    end = min(len(show), rec_idx + 15)
    st.dataframe(
        show.iloc[start:end][["multiple","sale_price","decision","P_sell_30d","P_sell_60d","net_profit","net_roi"]],
        use_container_width=True
    )
else:
    st.dataframe(
        show[["multiple","sale_price","decision","P_sell_30d","P_sell_60d","net_profit","net_roi"]].head(40),
        use_container_width=True
    )

st.download_button(
    "Download pricing scan CSV",
    data=pred.to_csv(index=False).encode("utf-8"),
    file_name="pricing_decision_scan.csv",
    mime="text/csv",
)

with st.expander("How to read this (client-friendly)"):
    st.markdown(
        f"""
**What this dashboard does**
- You enter the deal context (**county/city/acres/total cost**).
- We scan price points as **multiples** (Sale Price / Total Cost).
- For each multiple we estimate:
  - **P(sell {days_label})** (speed/velocity)
  - **Net profit** after commissions (10% of sale + 8% of profit by default)
  - **Net ROI**

**Recommended price ceiling**
- The app chooses the **highest** multiple that still meets your speed requirement:
  - **P(sell {days_label}) ≥ {target_prob:.0%}**
- That’s your “max list price” that still preserves fast compounding cycles.

**GO / CAUTION / NO-GO**
- **GO**: meets your target probability.
- **CAUTION**: close to target (within ~10 points).
- **NO-GO**: likely too slow.
"""
    )
