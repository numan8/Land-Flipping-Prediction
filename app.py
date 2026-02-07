import os, re, json
import requests, joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

# -----------------------------
# Page
# -----------------------------
st.set_page_config(page_title="Cash Sales Velocity — Pricing Decision", layout="wide")
st.title("Cash Sales Velocity — Pricing Decision Dashboard")
st.caption("Goal: choose a cash list price that maximizes probability of selling fast (≤30 / ≤60 days) to increase compounding cycles.")

# =========================
# 1) Paste Google Drive SHARE LINKS here
# =========================
CAL30_LINK = "https://drive.google.com/file/d/1LmJ1Av5CEvMsJ4I85DrpxCeXO0tc4dN4/view?usp=sharing"
CAL60_LINK = "https://drive.google.com/file/d/1U6IfGFEAtOQXI0tHtAcWDHGM3p29eqjf/view?usp=sharing"
FEATURES_LINK = "https://drive.google.com/file/d/1H6hkyzbfe4aGkpWH4y8FJgcLp_kYLSo2/view?usp=sharing"

def extract_file_id(url: str) -> str:
    m = re.search(r"/d/([a-zA-Z0-9_-]+)", url)
    if m: return m.group(1)
    m = re.search(r"id=([a-zA-Z0-9_-]+)", url)
    if m: return m.group(1)
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

    # if HTML returned -> need confirm token OR permissions issue
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
            raise RuntimeError("Drive returned HTML. Set sharing to: Anyone with link → Viewer.")

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

# =========================
# 2) Sidebar inputs (business inputs only)
# =========================
st.sidebar.header("Deal Inputs")

county_state = st.sidebar.text_input("County, State", value="Los Angeles, CA")
city = st.sidebar.text_input("City", value="Lancaster")
acres = st.sidebar.number_input("Acres", min_value=0.0, value=2.5, step=0.1)

total_cost = st.sidebar.number_input("Total Purchase Price (All-in)", min_value=1.0, value=6000.0, step=100.0)

st.sidebar.divider()
st.sidebar.subheader("Target speed")

mode = st.sidebar.radio("Target window", ["≤30 days", "≤60 days"], index=1)
target_prob = st.sidebar.slider("Target probability", 0.30, 0.95, 0.70, 0.01)

st.sidebar.divider()
st.sidebar.subheader("Pricing scan range")

m_min = st.sidebar.number_input("Min multiple", min_value=0.5, value=1.5, step=0.1)
m_max = st.sidebar.number_input("Max multiple", min_value=0.6, value=6.0, step=0.1)
m_step = st.sidebar.number_input("Step", min_value=0.05, value=0.1, step=0.05)

# =========================
# 3) Financial assumptions (for decision output)
# =========================
AFFILIATE_PCT_SALE = 0.10  # 10% of sale price
AGENT_PCT_PROFIT = 0.08    # 4% + 4% of profit

def estimate_net_profit(sale_price: float, total_cost: float) -> float:
    affiliate = AFFILIATE_PCT_SALE * sale_price
    profit_before_agents = sale_price - total_cost - affiliate
    agent_comm = AGENT_PCT_PROFIT * max(profit_before_agents, 0)
    net_profit = profit_before_agents - agent_comm
    return net_profit

# =========================
# 4) Build feature rows and predict
# =========================
def make_feature_row(multiple: float) -> dict:
    # must match training feature names
    return {
        "multiple": float(multiple),
        "total_cost": float(total_cost),
        "acres": float(acres),
        "county_state": str(county_state),
        "city": str(city),
    }

multiples = np.arange(m_min, m_max + 1e-9, m_step)
X = pd.DataFrame([make_feature_row(m) for m in multiples])

# Ensure ALL expected features exist
for col in FEATURES:
    if col not in X.columns:
        X[col] = np.nan
X = X[FEATURES].copy()

p30 = cal30.predict_proba(X)[:, 1]
p60 = cal60.predict_proba(X)[:, 1]

pred = pd.DataFrame({
    "multiple": multiples,
    "sale_price": multiples * float(total_cost),
    "P_sell_30d": p30,
    "P_sell_60d": p60,
})
pred["net_profit"] = pred["sale_price"].apply(lambda s: estimate_net_profit(float(s), float(total_cost)))
pred["net_roi"] = pred["net_profit"] / float(total_cost)
pred["cycles_per_year_est"] = np.where(
    mode == "≤30 days",
    pred["P_sell_30d"] * (365/30),  # rough proxy: probability × cycles/year if 30d target
    pred["P_sell_60d"] * (365/60),  # rough proxy for 60d target
)

# =========================
# 5) Decision logic
# =========================
if mode == "≤30 days":
    use_prob_col = "P_sell_30d"
    days_label = "≤30d"
else:
    use_prob_col = "P_sell_60d"
    days_label = "≤60d"

# Highest multiple that still meets target probability
feasible = pred[pred[use_prob_col] >= target_prob].copy()
rec = feasible.sort_values("multiple").iloc[-1] if len(feasible) else None

# traffic-light label
def label(prob):
    if prob >= target_prob:
        return "GO"
    if prob >= max(0.50, target_prob - 0.10):
        return "CAUTION"
    return "NO-GO"

pred["decision"] = pred[use_prob_col].apply(label)

# =========================
# 6) Clean KPI output (client-facing)
# =========================
st.subheader("Recommended price ceiling (decision output)")

k1, k2, k3, k4, k5 = st.columns(5)

if rec is None:
    k1.metric("Recommended max multiple", "—")
    k2.metric("Recommended list price", "—")
    k3.metric(f"Probability sell {days_label}", "—")
    k4.metric("Est. net profit (after commissions)", "—")
    k5.metric("Est. net ROI", "—")
    st.warning(
        f"No price in your scan meets target P(sell {days_label}) ≥ {target_prob:.0%}. "
        "Try scanning lower multiples or reduce the target probability."
    )
else:
    k1.metric("Recommended max multiple", f"{rec['multiple']:.2f}x")
    k2.metric("Recommended list price", f"${rec['sale_price']:,.0f}")
    k3.metric(f"P(sell {days_label})", f"{rec[use_prob_col]:.0%}")
    k4.metric("Est. net profit (after commissions)", f"${rec['net_profit']:,.0f}")
    k5.metric("Est. net ROI", f"{rec['net_roi']:.0%}")

st.caption(
    f"Decision rule: pick the **highest** multiple that still keeps P(sell {days_label}) ≥ {target_prob:.0%}. "
    "This maximizes price while maintaining speed."
)

# =========================
# 7) Charts (simple + interpretable)
# =========================
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

# =========================
# 8) Simple decision table
# =========================
st.subheader("Pricing options (decision table)")
show = pred.copy()
show["P_sell_30d"] = (show["P_sell_30d"] * 100).round(1)
show["P_sell_60d"] = (show["P_sell_60d"] * 100).round(1)
show["net_profit"] = show["net_profit"].round(0)
show["net_roi"] = (show["net_roi"] * 100).round(1)

st.dataframe(
    show[["multiple","sale_price","decision","P_sell_30d","P_sell_60d","net_profit","net_roi"]],
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
**What you’re seeing**
- We scan markup multiples (Sale Price / Total Cost).
- For each multiple, the model estimates the probability that the property sells **{days_label}**.
- We also estimate **net profit** after commissions (10% of sale + 8% of profit).

**Recommended ceiling**
- The dashboard chooses the **highest multiple** that still meets your target probability
  (**P(sell {days_label}) ≥ {target_prob:.0%}**).
- This supports your “velocity of money” strategy: keep sales fast to maximize compounding cycles.
"""
    )
