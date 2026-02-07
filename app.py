import os, re, json, hashlib
import requests, joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

# -----------------------------
# Page setup (modern UI)
# -----------------------------
st.set_page_config(page_title="Pricing Decision Engine", layout="wide")

st.markdown("""
<style>
/* Background */
.stApp {
  background: radial-gradient(1200px 600px at 20% -10%, rgba(99,102,241,0.20), transparent 55%),
              radial-gradient(1200px 600px at 85% 0%, rgba(14,165,233,0.18), transparent 55%),
              linear-gradient(180deg, #f8fafc 0%, #eef2ff 55%, #f8fafc 100%);
}

/* Title */
h1, h2, h3 { letter-spacing: -0.02em; }

/* Cards */
.card {
  background: rgba(255,255,255,0.75);
  border: 1px solid rgba(15, 23, 42, 0.08);
  box-shadow: 0 10px 30px rgba(2, 6, 23, 0.06);
  border-radius: 18px;
  padding: 16px 18px;
}
.badge {
  display:inline-block;
  padding: 4px 10px;
  border-radius: 999px;
  font-weight: 700;
  font-size: 12px;
  border: 1px solid rgba(0,0,0,0.08);
}
.badge-go { background: rgba(34,197,94,0.15); color:#166534; }
.badge-caution { background: rgba(234,179,8,0.18); color:#854d0e; }
.badge-nogo { background: rgba(239,68,68,0.14); color:#7f1d1d; }

/* Sidebar */
section[data-testid="stSidebar"] {
  background: rgba(255,255,255,0.80);
  border-right: 1px solid rgba(15, 23, 42, 0.08);
}
</style>
""", unsafe_allow_html=True)

st.title("Cash Sales Velocity — Pricing Decision Engine")
st.caption("Choose a cash list price that maximizes the probability of selling fast (≤30 / ≤60 days) while staying profitable and increasing compounding cycles.")

# =========================
# 1) Put Google Drive SHARE LINKS here (hard-coded)
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

    # If HTML returned -> confirm token or permissions issue
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
            raise RuntimeError("Google Drive returned HTML. Fix sharing: Anyone with link → Viewer.")

    with open(out_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)

@st.cache_resource
def load_models():
    with st.spinner("Loading pricing models..."):
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
# 2) Sidebar (CLIENT inputs only)
# =========================
st.sidebar.header("Deal Inputs")

county_state = st.sidebar.text_input("County, State", value="Los Angeles, CA")
city = st.sidebar.text_input("City", value="Lancaster")
acres = st.sidebar.number_input("Acres", min_value=0.0, value=2.5, step=0.1)
total_cost = st.sidebar.number_input("Total Purchase Price (All-in)", min_value=1.0, value=6000.0, step=100.0)

st.sidebar.divider()
st.sidebar.subheader("Sell-fast target")

mode = st.sidebar.radio("Target window", ["≤30 days", "≤60 days"], index=1)
target_prob = st.sidebar.slider("Minimum probability target", 0.40, 0.95, 0.70, 0.01)

st.sidebar.divider()
st.sidebar.subheader("Scan pricing range")

m_min = st.sidebar.number_input("Min markup multiple", min_value=0.5, value=1.2, step=0.1)
m_max = st.sidebar.number_input("Max markup multiple", min_value=0.6, value=6.0, step=0.1)
m_step = st.sidebar.number_input("Step", min_value=0.05, value=0.1, step=0.05)

st.sidebar.divider()
st.sidebar.subheader("Commission assumptions")
AFFILIATE_PCT_SALE = st.sidebar.number_input("Affiliate/List commission (% of sale)", min_value=0.0, max_value=0.5, value=0.10, step=0.01)
AGENT_PCT_PROFIT = st.sidebar.number_input("Agents total (% of profit)", min_value=0.0, max_value=0.5, value=0.08, step=0.01)

# =========================
# 3) Finance helpers
# =========================
def estimate_net_profit(sale_price: float, total_cost: float) -> float:
    affiliate = AFFILIATE_PCT_SALE * sale_price
    profit_before_agents = sale_price - total_cost - affiliate
    agent_comm = AGENT_PCT_PROFIT * max(profit_before_agents, 0)
    return profit_before_agents - agent_comm

# =========================
# 4) Feature building (handles common training schemas)
# =========================
def stable_hash_to_int(s: str, mod: int = 10_000_000) -> int:
    if s is None: s = ""
    h = hashlib.md5(s.encode("utf-8")).hexdigest()
    return int(h[:8], 16) % mod

def make_feature_row(multiple: float) -> dict:
    row = {
        "multiple": float(multiple),
        "total_cost": float(total_cost),
        "acres": float(acres),
        "county_state": str(county_state),
        "city": str(city),
        "county_state_hash": stable_hash_to_int(str(county_state)),
        "city_hash": stable_hash_to_int(str(city)),
    }
    return row

multiples = np.arange(m_min, m_max + 1e-9, m_step)
X_raw = pd.DataFrame([make_feature_row(m) for m in multiples])

# Ensure all expected features exist
for col in FEATURES:
    if col not in X_raw.columns:
        X_raw[col] = np.nan

X = X_raw[FEATURES].copy()

# Coerce types safely
for c in X.columns:
    # If the model expects numeric-only, force numeric where possible
    if X[c].dtype == "object":
        # Try numeric first, else hash
        tmp = pd.to_numeric(X[c], errors="coerce")
        if tmp.notna().any():
            X[c] = tmp
        else:
            X[c] = X[c].astype(str).apply(lambda v: stable_hash_to_int(v)).astype(float)

# =========================
# 5) Predict probabilities
# =========================
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

# Compounding proxy:
# expected cycles/year ≈ probability * (365/target_days)
target_days = 30 if mode == "≤30 days" else 60
use_prob_col = "P_sell_30d" if mode == "≤30 days" else "P_sell_60d"
pred["cycles_year_proxy"] = pred[use_prob_col] * (365.0 / target_days)

# =========================
# 6) Decision logic (client-friendly)
# =========================
feasible = pred[pred[use_prob_col] >= target_prob].copy()
rec = feasible.sort_values("multiple").iloc[-1] if len(feasible) else None

def decision_label(prob):
    if prob >= target_prob:
        return "GO"
    if prob >= max(0.50, target_prob - 0.10):
        return "CAUTION"
    return "NO-GO"

pred["decision"] = pred[use_prob_col].apply(decision_label)

def decision_badge(x: str) -> str:
    if x == "GO":
        return '<span class="badge badge-go">GO</span>'
    if x == "CAUTION":
        return '<span class="badge badge-caution">CAUTION</span>'
    return '<span class="badge badge-nogo">NO-GO</span>'

# =========================
# 7) Executive summary (TOP section)
# =========================
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Recommended price ceiling (decision output)")

k1, k2, k3, k4, k5, k6 = st.columns(6)

if rec is None:
    k1.metric("Recommended max multiple", "—")
    k2.metric("Recommended list price", "—")
    k3.metric(f"P(sell {mode})", "—")
    k4.metric("Est. net profit", "—")
    k5.metric("Est. net ROI", "—")
    k6.metric("Cycles/year (proxy)", "—")

    st.warning(
        f"No price in your scan meets target **P(sell {mode}) ≥ {target_prob:.0%}**.\n\n"
        "Try one of these:\n"
        "• scan **lower multiples** (reduce price)\n"
        "• lower the probability target a bit (e.g., 65%)\n"
        "• widen scan range (e.g., start from 1.0x)"
    )
else:
    k1.metric("Recommended max multiple", f"{rec['multiple']:.2f}x")
    k2.metric("Recommended list price", f"${rec['sale_price']:,.0f}")
    k3.metric(f"P(sell {mode})", f"{rec[use_prob_col]:.0%}")
    k4.metric("Est. net profit", f"${rec['net_profit']:,.0f}")
    k5.metric("Est. net ROI", f"{rec['net_roi']:.0%}")
    k6.metric("Cycles/year (proxy)", f"{rec['cycles_year_proxy']:.2f}")

st.caption(
    f"Rule used: choose the **highest** multiple that still keeps **P(sell {mode}) ≥ {target_prob:.0%}** — maximizing price while protecting velocity."
)
st.markdown('</div>', unsafe_allow_html=True)

st.write("")

# =========================
# 8) Charts (client interpretable)
# =========================
c1, c2 = st.columns(2)

with c1:
    fig = px.line(
        pred, x="multiple", y=use_prob_col,
        markers=True,
        title=f"Probability of selling {mode} vs markup multiple"
    )
    fig.update_yaxes(range=[0, 1])
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, use_container_width=True)

with c2:
    fig = px.line(
        pred, x="multiple", y="net_profit",
        markers=True,
        title="Estimated net profit vs markup multiple (after commissions)"
    )
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, use_container_width=True)

# =========================
# 9) Decision table (GO/CAUTION/NO-GO)
# =========================
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Pricing options (decision table)")

show = pred.copy()
show["sale_price"] = show["sale_price"].round(0)
show["P_sell_30d"] = (show["P_sell_30d"] * 100).round(1)
show["P_sell_60d"] = (show["P_sell_60d"] * 100).round(1)
show["net_profit"] = show["net_profit"].round(0)
show["net_roi"] = (show["net_roi"] * 100).round(1)
show["cycles_year_proxy"] = show["cycles_year_proxy"].round(2)

# Make a clean table for client
client_table = show[["multiple","sale_price","decision","P_sell_30d","P_sell_60d","net_profit","net_roi","cycles_year_proxy"]].copy()
client_table = client_table.rename(columns={
    "sale_price":"list_price",
    "net_profit":"est_net_profit",
    "net_roi":"est_net_roi_%"
})

st.dataframe(client_table, use_container_width=True, hide_index=True)

st.download_button(
    "Download decision scan CSV",
    data=pred.to_csv(index=False).encode("utf-8"),
    file_name="pricing_decision_scan.csv",
    mime="text/csv",
)

with st.expander("How to read this (simple)"):
    st.markdown(f"""
- **multiple** = Sale Price / Total Cost  
- **P(sell {mode})** is the model’s estimate you sell within the target window  
- **Recommended ceiling** = highest multiple that still meets your probability target  
- **Cycles/year (proxy)** = probability × (365 / target_days)  
  - Higher cycles/year means faster redeployment / more compounding
""")

st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# Safety check (optional, hidden-ish)
# -----------------------------
# If curve is totally flat, it usually means the model isn't "seeing" the multiple feature.
# We don't show this to the client; we show as a subtle note only to operator.
flat_check = np.std(pred[use_prob_col].values)
if flat_check < 1e-6:
    st.info("Operator note: Probability curve is flat. This usually means the model feature schema differs from this app. Confirm your FEATURES match training.")
