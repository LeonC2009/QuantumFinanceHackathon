import json
import os

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Quantum ESG Hedging", page_icon="⚛️", layout="wide")
st.title("Dynamic Antipodal Hedging: Real-Time Carbon Neutrality")
st.caption(
    "Quantum & AI Sustainability Hackathon · Hanken School of Economics — "
    "QUBO-selected, fixed-cardinality energy futures hedged via Borsuk-Ulam antipodal mapping."
)

LOG_PATH = "simulation_log.json"


# ---------------------------------------------------------------------------
# Load simulation data
# ---------------------------------------------------------------------------
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    with open(path, "r") as f:
        data = json.load(f)
    return pd.DataFrame(data)


if not os.path.exists(LOG_PATH):
    st.error(
        f"Couldn't find `{LOG_PATH}` next to this app. Run `cardinality_qubo.py` "
        "first so it can write the simulation log, then reload this page."
    )
    st.stop()

try:
    df = load_data(LOG_PATH)
except (json.JSONDecodeError, ValueError) as e:
    st.error(f"`{LOG_PATH}` exists but couldn't be parsed as a simulation log: {e}")
    st.stop()

required_cols = {
    "step",
    "portfolio_value",
    "carbon_drift",
    "hedged_carbon_exposure",
    "long_positions",
    "short_positions",
}
missing = required_cols - set(df.columns)
if missing:
    st.error(f"`{LOG_PATH}` is missing expected column(s): {', '.join(sorted(missing))}")
    st.stop()

if df.empty:
    st.warning(f"`{LOG_PATH}` loaded but contains no rows yet.")
    st.stop()

# ---------------------------------------------------------------------------
# Executive metrics
# ---------------------------------------------------------------------------
st.markdown("### Portfolio Performance Metrics")

mean_abs_drift = df["carbon_drift"].abs().mean()
mean_abs_hedged = df["hedged_carbon_exposure"].abs().mean()
neutralized_pct = (
    max(0.0, (1 - mean_abs_hedged / mean_abs_drift) * 100) if mean_abs_drift else 0.0
)

col1, col2, col3 = st.columns(3)
col1.metric("Final Portfolio Value", f"${df['portfolio_value'].iloc[-1]:,.2f}")
col2.metric("Total Carbon Drift (Unhedged)", f"{df['carbon_drift'].sum():.4f}")
col3.metric(
    "Net Portfolio Carbon Exposure",
    f"{df['hedged_carbon_exposure'].mean():.4f}",
    f"-{neutralized_pct:.0f}% neutralized vs. unhedged drift",
)

st.divider()

# ---------------------------------------------------------------------------
# Dual visualization: returns vs. risk
# ---------------------------------------------------------------------------
colA, colB = st.columns(2)

with colA:
    st.markdown("#### Cumulative Portfolio Value")
    fig_val = px.line(
        df,
        x="step",
        y="portfolio_value",
        labels={"step": "Time Step", "portfolio_value": "Value (USD)"},
        color_discrete_sequence=["#00FF00"],
    )
    fig_val.update_layout(plot_bgcolor="black", paper_bgcolor="black", font_color="white")
    st.plotly_chart(fig_val, use_container_width=True)

with colB:
    st.markdown("#### Carbon Exposure: Unhedged vs. Hedged")
    fig_carbon = go.Figure()
    # Volatile unhedged market drift
    fig_carbon.add_trace(
        go.Scatter(
            x=df["step"],
            y=df["carbon_drift"],
            mode="lines",
            name="Market Carbon Drift",
            line=dict(color="#FF4444", width=2),
        )
    )
    # Flat hedged exposure via Borsuk-Ulam antipodal mapping
    fig_carbon.add_trace(
        go.Scatter(
            x=df["step"],
            y=df["hedged_carbon_exposure"],
            mode="lines",
            name="Quantum Hedged Exposure",
            line=dict(color="#00AAFF", width=3, dash="dot"),
        )
    )
    fig_carbon.update_layout(
        plot_bgcolor="black",
        paper_bgcolor="black",
        font_color="white",
        xaxis_title="Time Step",
        yaxis_title="Carbon Impact Factor",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    st.plotly_chart(fig_carbon, use_container_width=True)

# ---------------------------------------------------------------------------
# Live trade execution log
# ---------------------------------------------------------------------------
st.markdown("### Current Execution State (Fixed Cardinality)")
st.dataframe(
    df[["step", "long_positions", "short_positions"]].tail(5),
    use_container_width=True,
)
