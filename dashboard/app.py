"""
Streamlit dashboard for the ERCOT grid stress forecaster.

Displays:
- A price chart with actual vs predicted (enhanced model vs DAM)
- A composite grid stress index over time
- An alerts panel showing spike probabilities
- Feature importance rankings

Run with: streamlit run dashboard/app.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import xgboost as xgb

from utils.config import FEATURES_DIR
from models.utils import select_enhanced_features

st.set_page_config(page_title="ERCOT Grid Stress Forecaster", layout="wide")
st.title("ERCOT Grid Stress & Price Forecaster")
st.caption("Sentiment-enhanced ML forecaster for Texas electricity prices")


@st.cache_data
def load_data():
    df = pd.read_parquet(FEATURES_DIR / "feature_matrix_engineered.parquet")
    return df


@st.cache_resource
def load_models():
    enhanced_reg_path = Path("models/artifacts/enhanced_regressor.json")
    enhanced_clf_path = Path("models/artifacts/enhanced_classifier.json")
    reg = xgb.XGBRegressor()
    clf = xgb.XGBClassifier()
    if enhanced_reg_path.exists():
        reg.load_model(str(enhanced_reg_path))
    if enhanced_clf_path.exists():
        clf.load_model(str(enhanced_clf_path))
    return reg, clf


df = load_data()
reg, clf = load_models()

# Sidebar controls
st.sidebar.header("View Settings")
date_min = df.index.min().date()
date_max = df.index.max().date()
default_start = max(date_min, (df.index.max() - pd.Timedelta(days=30)).date())

date_range = st.sidebar.date_input(
    "Date range",
    value=(default_start, date_max),
    min_value=date_min,
    max_value=date_max,
)
if isinstance(date_range, tuple) and len(date_range) == 2:
    start, end = date_range
    mask = (df.index.date >= start) & (df.index.date <= end)
    view = df[mask].copy()
else:
    view = df.copy()

# Compute predictions on view
features = select_enhanced_features(df)
X = view[features].fillna(method="ffill").fillna(0)
try:
    view["model_forecast"] = reg.predict(X)
except Exception:
    view["model_forecast"] = None
try:
    view["spike_probability"] = clf.predict_proba(X)[:, 1]
except Exception:
    view["spike_probability"] = 0

# --- Top-level KPIs ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("Current Price", f"${view['HB_HUBAVG'].iloc[-1]:.2f}/MWh" if len(view) else "n/a")
col2.metric("Peak in Window", f"${view['HB_HUBAVG'].max():.2f}/MWh" if len(view) else "n/a")
col3.metric("Spike Events", int(view['price_spike_flag'].sum()) if len(view) else 0)
col4.metric(
    "Max Spike Probability",
    f"{view['spike_probability'].max():.1%}" if len(view) else "n/a"
)

# --- Main price chart ---
st.subheader("Price Forecast vs Actual")
fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_trace(go.Scatter(
    x=view.index, y=view["HB_HUBAVG"],
    name="Actual HB_HUBAVG", line=dict(color="#1f77b4", width=1.5)
))
if "da_HB_HUBAVG" in view.columns:
    fig.add_trace(go.Scatter(
        x=view.index, y=view["da_HB_HUBAVG"],
        name="Day-Ahead Market", line=dict(color="#888", width=1, dash="dot")
    ))
if view["model_forecast"].notna().any():
    fig.add_trace(go.Scatter(
        x=view.index, y=view["model_forecast"],
        name="Model Forecast", line=dict(color="#d62728", width=1.5)
    ))
fig.add_hline(y=200, line_dash="dash", line_color="orange",
              annotation_text="Spike threshold ($200)")
fig.add_trace(go.Scatter(
    x=view.index, y=view["spike_probability"],
    name="Spike Prob (24h)", line=dict(color="#2ca02c", width=1),
    yaxis="y2",
), secondary_y=True)
fig.update_yaxes(title_text="Price ($/MWh)", secondary_y=False)
fig.update_yaxes(title_text="Spike Probability", secondary_y=True, range=[0, 1])
fig.update_layout(height=500, hovermode="x unified")
st.plotly_chart(fig, use_container_width=True)

# --- Grid stress index ---
st.subheader("Composite Grid Stress Index")
if "stress_score" in view.columns:
    stress_fig = go.Figure()
    stress_fig.add_trace(go.Scatter(
        x=view.index, y=view["stress_score"],
        fill="tozeroy", line=dict(color="#ff7f0e"),
        name="Stress Score"
    ))
    stress_fig.update_layout(
        height=250, yaxis_title="Concurrent Stress Conditions",
        xaxis_title=""
    )
    st.plotly_chart(stress_fig, use_container_width=True)

# --- Alerts panel ---
st.subheader("Active Alerts")
if len(view) and view["spike_probability"].max() > 0.5:
    alerts = view[view["spike_probability"] > 0.5].copy()
    alerts = alerts[["HB_HUBAVG", "spike_probability", "stress_score"]].tail(20)
    st.dataframe(alerts.style.format({
        "HB_HUBAVG": "${:.2f}",
        "spike_probability": "{:.1%}",
    }))
else:
    st.info("No active spike alerts in the selected window.")

# --- Feature importance ---
st.subheader("Model Feature Importance")
fi_path = Path("models/artifacts/enhanced_feature_importance.csv")
if fi_path.exists():
    fi = pd.read_csv(fi_path, index_col=0, header=None, names=["feature", "importance"])
    fi = fi.sort_values("importance", ascending=True).tail(20)
    bar = go.Figure(go.Bar(
        x=fi["importance"], y=fi["feature"], orientation="h",
    ))
    bar.update_layout(height=500, xaxis_title="Importance", yaxis_title="")
    st.plotly_chart(bar, use_container_width=True)
else:
    st.info("Train the enhanced model to see feature importance.")
