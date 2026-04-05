"""Live WFA progress dashboard — run with: streamlit run wfa_dashboard.py"""

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

PROGRESS_FILE = Path(__file__).resolve().parent.parent.parent / "docs" / "wfa-progress.json"

st.set_page_config(page_title="WFA Live Monitor", layout="wide")
st.title("Hybrid Backtest — Live Monitor")


@st.fragment(run_every=3)
def live_charts():
    if not PROGRESS_FILE.exists():
        st.info("Waiting for WFA to start... (no progress file yet)")
        return

    raw = json.loads(PROGRESS_FILE.read_text())
    dates = pd.to_datetime(raw["dates"])
    n = len(dates)

    if n == 0:
        st.info("WFA started but no data points yet...")
        return

    col1, col2, col3 = st.columns(3)
    col1.metric("Progress", f"{n} days")
    col2.metric("Latest Date", dates[-1].strftime("%Y-%m-%d"))
    hybrid_ret = raw["hybrid_equity"][-1] / raw["hybrid_equity"][0] - 1 if raw["hybrid_equity"][0] > 0 else 0
    col3.metric("Hybrid Return", f"{hybrid_ret:.1%}")

    eq_fig = go.Figure()
    eq_fig.add_scatter(x=dates, y=raw["sp500_equity"], name="S&P 500",
                       line=dict(color="orange", width=1), opacity=0.7)
    eq_fig.add_scatter(x=dates, y=raw["hybrid_equity"], name="Hybrid",
                       line=dict(color="green", width=2))
    eq_fig.update_layout(
        title=f"Equity Curves ({raw.get('pct', '')})",
        yaxis_title="Equity ($)", yaxis_type="log",
        height=450, template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    st.plotly_chart(eq_fig, width="stretch")

    dd_fig = go.Figure()
    dd_fig.add_scatter(x=dates, y=raw["sp500_dd"], name="S&P 500 DD",
                       fill="tozeroy", line=dict(color="orange", width=0.8),
                       fillcolor="rgba(255,165,0,0.2)")
    dd_fig.add_scatter(x=dates, y=raw["hybrid_dd"], name="Hybrid DD",
                       line=dict(color="green", width=1.2))
    dd_fig.add_hline(y=-0.10, line_dash="dash", line_color="red",
                     annotation_text="Bear entry (-10%)")
    dd_fig.add_hline(y=-0.05, line_dash="dash", line_color="green",
                     annotation_text="Bear exit (-5%)")
    dd_fig.update_layout(
        title="Drawdown",
        yaxis_title="Drawdown", yaxis_tickformat=".0%",
        height=300, template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    st.plotly_chart(dd_fig, width="stretch")

    st.caption(f"Auto-refreshes every 3 seconds • Reading from `{PROGRESS_FILE}`")


live_charts()
