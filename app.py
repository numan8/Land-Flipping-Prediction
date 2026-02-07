import os
import json
import requests
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="Cash Sales Velocity — Prediction", layout="wide")
st.title("Cash Sales Velocity — Prediction Dashboard")
st.caption("Loads trained models from Google Drive, then predicts P(sell ≤30d) and P(sell ≤60d).")

# =========================
# 1) Put your Google Drive file IDs here
# =========================
CAL30_FILE_ID = "https://drive.google.com/file/d/1LmJ1Av5CEvMsJ4I85DrpxCeXO0tc4dN4/view?usp=sharing"
CAL60_FILE_ID = "https://drive.google.com/file/d/1U6IfGFEAtOQXI0tHtAcWDHGM3p29eqjf/view?usp=sharing"
FEATURES_FILE_ID = "https://drive.google.com/file/d/1H6hkyzbfe4aGkpWH4y8FJgcLp_kYLSo2/view?usp=sharing"
REPORT_FILE_ID = "https://drive.google.com/file/d/1qBhw9MOxImlkEIl7opsMn9xPf-17_n6s/view?usp=sharing"

ART_DIR = "artifacts"
CAL30_PATH = os.path.join(ART_DIR, "cal30.pkl")
CAL60_PATH = os.path.join(ART_DIR, "cal60.pkl")
FEATURES_PATH = os.path.join(ART_DIR, "features.json")
REPORT_PATH = os.path.join(ART_DIR, "model_report.json")

def gdrive_direct_url(file_id: str) -> str:
    return f"https://drive.google.com/uc?export=download&id={file_id}"

def download_from_gdrive(file_id: str, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        return

    url = gdrive_direct_url(file_id)
    with requests.Session() as s:
        r = s.get(url, stream=True)
        r.raise_for_status()

        # Handle Google Drive confirmation token for large files
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
def load_models():
    with st.spinner("Downloading model artifacts from Google Drive..."):
        download_from_gdrive(CAL30_FILE_ID, CAL30_PATH)
        download_from_gdrive(CAL60_FILE_ID, CAL60_PATH)
        download_from_gdrive(FEATURES_FILE_ID, FEATURES_PATH)
        download_from_gdrive(REPORT_FILE_ID, REPORT_PATH)

    cal30 = joblib.load(CAL30_PATH)
    cal60 = joblib.load(CAL60_PATH)

    with open(FEATURES_PATH, "r") as f:
        meta = json.load(f)
    features = meta["features"] if isinstance(meta, dict) and "features" in meta else meta

    report = {}
    try:
        with open(REPORT_PATH, "r") as f:
            report = json.load(f)
    except Exception:
        pass

    return cal30, cal60, features, report

cal30, cal60, FEATURES, report = load_models()

# =========================
# 2) UI inputs
# =========================
st.sidebar.header("Inputs")
county_state = st.sidebar.text_input("County, State", value="Los Angeles, CA")
city = st.sidebar.text_input("City", value="Lancaster")
acres = st.sidebar.number_input("Acres", min_value=0.0, value=2.5, step=0.1)
total_cost = st.sidebar.number_input("Total Purchase Price", min_value=1.0, value=6000.0, step=100.0)

st.sidebar.subheader("Scan price multiples")
m_min = st.sidebar.number_input("Min multiple", min_value=0.5, value=1.5, step=0.1)
m_max = st.sidebar.number_input("Max multiple", min_value=0.6, value=6.0, step=0.1)
m_step = st.sidebar.number_input("Step", min_value=0.05, value=0.1, step=0.05)

target_p60 = st.sidebar.slider("Target P(sell ≤60d)", 0.0, 1.0, 0.70, 0.01)
target_p30 = st.sidebar.slider("Target P(sell ≤30d)", 0.0, 1.0, 0.60, 0.01)
enforce_30 = st.sidebar.checkbox("Also enforce ≤30d target", value=False)

# =========================
# 3) Build features EXACTLY as training expects
#    IMPORTANT: This assumes your training used these feature names.
#    If your FEATURES include other columns, we must create them here too.
# =========================
def make_feature_row(multiple: float):
    # Minimal safe defaults
    row = {
        "multiple": float(multiple),
        "total_cost": float(total_cost),
        "acres": float(acres),
        "county_state": str(county_state),
        "city": str(city),
    }
    return row

multiples = np.arange(m_min, m_max + 1e-9, m_step)
X_new = pd.DataFrame([make_feature_row(m) for m in multiples])

# Ensure ALL expected features exist
for col in FEATURES:
    if col not in X_new.columns:
        X_new[col] = np.nan

X_new = X_new[FEATURES].copy()

# If your model expects numeric-only, coerce where possible
for c in X_new.columns:
    if X_new[c].dtype == "object":
        X_new[c] = X_new[c].astype("string")

# =========================
# 4) Predict
# =========================
p30 = cal30.predict_proba(X_new)[:, 1]
p60 = cal60.predict_proba(X_new)[:, 1]

pred = pd.DataFrame({
    "multiple": multiples,
    "sale_price_est": multiples * float(total_cost),
    "P_sell_30d": p30,
    "P_sell_60d": p60,
})

# Recommendation
mask = pred["P_sell_60d"] >= target_p60
if enforce_30:
    mask &= pred["P_sell_30d"] >= target_p30

feasible = pred[mask]
rec = feasible.sort_values("multiple").iloc[-1] if len(feasible) else None

# =========================
# 5) Output
# =========================
st.subheader("Model report (from training)")
if report:
    st.json(report)
else:
    st.info("No report loaded (optional).")

if rec is None:
    st.warning("No multiple in your scan meets the selected probability target(s). Try lowering targets or scanning lower multiples.")
else:
    st.success(
        f"Recommended MAX multiple: **{rec['multiple']:.2f}x** "
        f"(sale ≈ **${rec['sale_price_est']:,.0f}**) "
        f"| P30=**{rec['P_sell_30d']:.2f}** P60=**{rec['P_sell_60d']:.2f}**"
    )

c1, c2 = st.columns(2)
with c1:
    fig = px.line(pred, x="multiple", y="P_sell_60d", title="P(sell ≤60d) vs multiple")
    fig.update_yaxes(range=[0, 1])
    st.plotly_chart(fig, use_container_width=True)

with c2:
    fig = px.line(pred, x="multiple", y="P_sell_30d", title="P(sell ≤30d) vs multiple")
    fig.update_yaxes(range=[0, 1])
    st.plotly_chart(fig, use_container_width=True)

st.markdown("### Prediction table")
st.dataframe(pred.round(3), use_container_width=True)

st.download_button(
    "Download prediction scan CSV",
    data=pred.to_csv(index=False).encode("utf-8"),
    file_name="prediction_scan.csv",
    mime="text/csv",
)
