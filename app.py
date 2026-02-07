import os
import re
import json
import requests
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

# =========================
# PAGE CONFIG + PREMIUM UI
# =========================
st.set_page_config(page_title="Cash Sales Velocity — Pricing Decision", layout="wide")

st.markdown("""
<style>
/* Background + typography */
.stApp {
  background: radial-gradient(1200px 600px at 20% -10%, rgba(99,102,241,0.18), transparent 55%),
              radial-gradient(1200px 600px at 85% 0%, rgba(14,165,233,0.15), transparent 55%),
              linear-gradient(180deg, #ffffff 0%, #f8fafc 55%, #ffffff 100%);
}
.block-container { padding-top: 1.2rem; padding-bottom: 1.2rem; }

h1, h2, h3 { letter-spacing: -0.3px; }
.small-note { color: rgba(15,23,42,0.65); font-size: 0.92rem; }

.kpi-card {
  border-radius: 18px;
  padding: 14px 16px;
  background: rgba(255,255,255,0.75);
  border: 1px solid rgba(148,163,184,0.35);
  box-shadow: 0 10px 30px rgba(2,6,23,0.06);
}
.kpi-label { font-size: 0.85rem; color: rgba(15,23,42,0.65); }
.kpi-value { font-size: 1.35rem; font-weight: 700; color: rgba(2,6,23,0.92); }

.badge {
  display: inline-block;
  padding: 6px 10px;
  border-radius: 999px;
  font-weight: 600;
  font-size: 0.85rem;
}
.badge-go { background: rgba(16,185,129,0.12); color: rgb(5,150,105); border: 1px solid rgba(16,185,129,0.25); }
.badge-nogo { background: rgba(244,63,94,0.10); color: rgb(225,29,72); border: 1px solid rgba(244,63,94,0.25); }

.hr { height: 1px; background: rgba(148,163,184,0.35); margin: 12px 0 18px 0; }

</style>
""", unsafe_allow_html=True)

st.title("Cash Sales Velocity — Pricing Decision Dashboard")
st.caption("Goal: choose a cash list price that maximizes probability of selling fast (≤30 / ≤60 days) while staying profitable and increasing compounding cycles.")

# =========================
# GOOGLE DRIVE DOWNLOAD
# =========================
ART_DIR = "artifacts"
CAL30_PATH = os.path.join(ART_DIR, "cal30.pkl")
CAL60_PATH = os.path.join(ART_DIR, "cal60.pkl")
FEATURES_PATH = os.path.join(ART_DIR, "features.json")

def extract_gdrive_file_id(x: str) -> str:
    """
    Accepts:
    - raw file id: 1LmJ1Av5CEvMsJ4I85DrpxCeXO0tc4dN4
    - share link:  https://drive.google.com/file/d/<ID>/view?usp=sharing
    - uc link:     https://drive.google.com/uc?export=download&id=<ID>
    Returns: ID
    """
    if x is None:
        return ""
    x = x.strip()
    # already an id?
    if re.fullmatch(r"[A-Za-z0-9_-]{20,}", x):
        return x
    m = re.search(r"/file/d/([A-Za-z0-9_-]{20,})", x)
    if m:
        return m.group(1)
    m = re.search(r"[?&]id=([A-Za-z0-9_-]{20,})", x)
    if m:
        return m.group(1)
    return ""

def gdrive_direct_url(file_id: str) -> str:
    return f"https://drive.google.com/uc?export=download&id={file_id}"

def download_from_gdrive(file_id_or_link: str, out_path: str):
    file_id = extract_gdrive_file_id(file_id_or_link)
    if not file_id:
        raise ValueError("Invalid Google Drive ID/link. Paste the share link or file ID.")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Skip if already downloaded
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        return

    url = gdrive_direct_url(file_id)
    with requests.Session() as s:
        r = s.get(url, stream=True)
        r.raise_for_status()

        # Handle Google Drive confirmation token (sometimes needed)
        token = None
        for k, v in r.cookies.items():
            if k.startswith("download_warning"):
                token = v
                break
        if token:
            r = s.get(url + f"&confirm={token}", stream=True)
            r.raise_for_status()

        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

@st.cache_resource
def load_models_from_gdrive(cal30_link: str, cal60_link: str, features_link: str):
    with st.spinner("Loading prediction models…"):
        download_from_gdrive(cal30_link, CAL30_PATH)
        download_from_gdrive(cal60_link, CAL60_PATH)
        download_from_gdrive(features_link, FEATURES_PATH)

        cal30 = joblib.load(CAL30_PATH)
        cal60 = joblib.load(CAL60_PATH)

        with open(FEATURES_PATH, "r") as f:
            meta = json.load(f)
        FEATURES = meta["features"] if isinstance(meta, dict) and "features" in meta else meta

    return cal30, cal60, FEATURES

# =========================
# SIDEBAR INPUTS (CLIENT)
# =========================
st.sidebar.header("1) Load Models")

# ✅ put your drive links or IDs here as defaults (user can change)
CAL30_LINK = st.sidebar.text_input("cal30.pkl (Drive link or ID)", value="")
CAL60_LINK = st.sidebar.text_input("cal60.pkl (Drive link or ID)", value="")
FEAT_LINK  = st.sidebar.text_input("features.json (Drive link or ID)", value="")

if not (CAL30_LINK and CAL60_LINK and FEAT_LINK):
    st.info("Paste your Google Drive links (or file IDs) for cal30.pkl, cal60.pkl, and features.json in the sidebar.")
    st.stop()

try:
    cal30, cal60, FEATURES = load_models_from_gdrive(CAL30_LINK, CAL60_LINK, FEAT_LINK)
except Exception as e:
    st.error(f"Could not load models. Error: {e}")
    st.stop()

st.sidebar.header("2) Property Inputs")
county_state = st.sidebar.text_input("County, State", value="Los Angeles, CA")
city = st.sidebar.text_input("City", value="Lancaster")
acres = st.sidebar.number_input("Acres", min_value=0.0, value=2.5, step=0.1)
total_cost = st.sidebar.number_input("Total purchase price ($)", min_value=1.0, value=6000.0, step=100.0)

st.sidebar.header("3) Pricing Strategy")
goal_mode = st.sidebar.radio("Speed goal", ["≤30 days priority", "≤60 days priority"], index=1)
target_p30 = st.sidebar.slider("Target P(sell ≤30d)", 0.0, 1.0, 0.60, 0.01)
target_p60 = st.sidebar.slider("Target P(sell ≤60d)", 0.0, 1.0, 0.65, 0.01)

min_net_profit = st.sidebar.number_input("Min net profit ($)", value=0.0, step=100.0)
min_net_roi = st.sidebar.slider("Min net ROI (after commissions)", 0.0, 5.0, 0.10, 0.05)  # 0.10 = 10%

st.sidebar.header("4) Scan Multiples")
m_min = st.sidebar.number_input("Min multiple", min_value=0.5, value=1.5, step=0.1)
m_max = st.sidebar.number_input("Max multiple", min_value=0.6, value=6.0, step=0.1)
m_step = st.sidebar.number_input("Step", min_value=0.05, value=0.1, step=0.05)

# =========================
# FEATURE BUILDER (MUST MATCH TRAINING)
# =========================
def make_feature_row(multiple: float) -> dict:
    return {
        "multiple": float(multiple),
        "total_cost": float(total_cost),
        "acres": float(acres),
        "county_state": str(county_state),
        "city": str(city),
    }

multiples = np.arange(m_min, m_max + 1e-9, m_step)
X_new = pd.DataFrame([make_feature_row(m) for m in multiples])

# Ensure ALL expected feature columns exist
for c in FEATURES:
    if c not in X_new.columns:
        X_new[c] = np.nan
X_new = X_new[FEATURES].copy()

# Convert object cols to string (safe for encoders that expect it)
for c in X_new.columns:
    if X_new[c].dtype == "object":
        X_new[c] = X_new[c].astype("string")

# =========================
# PREDICT
# =========================
p30 = cal30.predict_proba(X_new)[:, 1]
p60 = cal60.predict_proba(X_new)[:, 1]

pred = pd.DataFrame({
    "multiple": multiples,
    "sale_price": multiples * float(total_cost),
    "P_sell_30d": p30,
    "P_sell_60d": p60,
})

# =========================
# PROFIT MODEL (DECISION OUTPUT)
# (using your stated commission assumptions)
# - Affiliate/Listing: 10% of sale price
# - Agents: 8% of profit_before_agents, only if profit positive
# =========================
pred["affiliate_commission"] = 0.10 * pred["sale_price"]
pred["profit_before_agents"] = pred["sale_price"] - float(total_cost) - pred["affiliate_commission"]
pred["agent_commission"] = 0.08 * pred["profit_before_agents"].clip(lower=0)
pred["net_profit"] = pred["profit_before_agents"] - pred["agent_commission"]
pred["net_roi"] = pred["net_profit"] / float(total_cost)

# Optional: simple “estimated cycle days” proxy from probabilities (client-friendly heuristic)
# Interpretation:
# - high P30 -> lower expected days
# - low P60 -> higher expected days
pred["est_days"] = (
    25 * pred["P_sell_30d"] +
    45 * (pred["P_sell_60d"] - pred["P_sell_30d"]).clip(lower=0) +
    120 * (1 - pred["P_sell_60d"])
).clip(lower=10, upper=365)

pred["est_cycles_year"] = 365 / pred["est_days"]

# =========================
# DECISION RULE (GO/NO-GO)
# =========================
if goal_mode == "≤30 days priority":
    speed_mask = (pred["P_sell_30d"] >= target_p30) & (pred["P_sell_60d"] >= target_p60)
else:
    speed_mask = (pred["P_sell_60d"] >= target_p60)

profit_mask = (pred["net_profit"] >= float(min_net_profit)) & (pred["net_roi"] >= float(min_net_roi))
pred["decision"] = np.where(speed_mask & profit_mask, "GO", "NO-GO")

feasible = pred[(pred["decision"] == "GO")].copy()

# Recommendation: pick the HIGHEST multiple still “GO”
rec = feasible.sort_values("multiple").iloc[-1] if len(feasible) else None

# =========================
# MAIN OUTPUT (CLIENT-FACING)
# =========================
st.markdown("## Recommended price ceiling (decision output)")
st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

k1, k2, k3, k4, k5, k6 = st.columns(6)

def kpi(col, label, value):
    col.markdown(f"""
    <div class="kpi-card">
      <div class="kpi-label">{label}</div>
      <div class="kpi-value">{value}</div>
    </div>
    """, unsafe_allow_html=True)

if rec is None:
    kpi(k1, "Recommended max multiple", "—")
    kpi(k2, "Recommended list price", "—")
    kpi(k3, "P(sell ≤30d)", "—")
    kpi(k4, "P(sell ≤60d)", "—")
    kpi(k5, "Est. net profit", "—")
    kpi(k6, "Est. cycles/year", "—")

    st.warning(
        "No price in your scan meets the selected speed + profit targets. "
        "Try scanning lower multiples, lowering probability targets, or reducing ROI constraints."
    )
else:
    kpi(k1, "Recommended max multiple", f"{rec['multiple']:.2f}x")
    kpi(k2, "Recommended list price", f"${rec['sale_price']:,.0f}")
    kpi(k3, "P(sell ≤30d)", f"{rec['P_sell_30d']*100:.1f}%")
    kpi(k4, "P(sell ≤60d)", f"{rec['P_sell_60d']*100:.1f}%")
    kpi(k5, "Est. net profit", f"${rec['net_profit']:,.0f}")
    kpi(k6, "Est. cycles/year", f"{rec['est_cycles_year']:.1f}")

    st.markdown(
        f"<span class='badge badge-go'>GO</span> "
        f"<span class='small-note'>Decision rule: pick the highest multiple that meets speed + profit constraints.</span>",
        unsafe_allow_html=True
    )

st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

# =========================
# CHARTS (CLEAN, DECISION-ORIENTED)
# =========================
c1, c2 = st.columns(2)

with c1:
    fig = px.line(pred, x="multiple", y="P_sell_60d", title="Probability of selling ≤60d vs markup multiple")
    fig.update_yaxes(range=[0, 1])
    st.plotly_chart(fig, use_container_width=True)

with c2:
    fig = px.line(pred, x="multiple", y="P_sell_30d", title="Probability of selling ≤30d vs markup multiple")
    fig.update_yaxes(range=[0, 1])
    st.plotly_chart(fig, use_container_width=True)

c3, c4 = st.columns(2)

with c3:
    fig = px.line(pred, x="multiple", y="net_profit", title="Estimated net profit vs markup multiple (after commissions)")
    st.plotly_chart(fig, use_container_width=True)

with c4:
    fig = px.line(pred, x="multiple", y="est_cycles_year", title="Estimated compounding cycles/year vs markup multiple (proxy)")
    st.plotly_chart(fig, use_container_width=True)

# =========================
# DECISION TABLE
# =========================
st.markdown("## Pricing options (decision table)")

table = pred.copy()
table["P_sell_30d"] = (table["P_sell_30d"] * 100).round(1)
table["P_sell_60d"] = (table["P_sell_60d"] * 100).round(1)
table["sale_price"] = table["sale_price"].round(0)
table["net_profit"] = table["net_profit"].round(0)
table["net_roi"] = (table["net_roi"] * 100).round(1)
table["est_days"] = table["est_days"].round(0)
table["est_cycles_year"] = table["est_cycles_year"].round(1)

show_cols = ["multiple", "sale_price", "decision", "P_sell_30d", "P_sell_60d", "net_profit", "net_roi", "est_days", "est_cycles_year"]
st.dataframe(table[show_cols], use_container_width=True, height=420)

st.download_button(
    "Download pricing scan CSV",
    data=table[show_cols].to_csv(index=False).encode("utf-8"),
    file_name="pricing_decision_scan.csv",
    mime="text/csv",
)

with st.expander("How to read this (client-friendly)"):
    st.markdown("""
- **Multiple**: sale_price / total_cost  
- **P_sell_30d / P_sell_60d**: model-estimated probability your cash buyer sells within the target window  
- **Net profit / ROI**: estimated after commissions (10% of sale + 8% of profit when positive)  
- **GO** means it meets your target speed and profit constraints  
- **Recommended max multiple** is the **highest GO** — it maximizes price while maintaining speed targets  
""")

# NOTE for you (not client) — keep this short, or remove entirely if client-facing
st.caption("Tip: If probabilities look flat across multiples, it usually means location/other factors drive speed more than markup alone in your historical data.")
