"""Live Full Pairs Trading progress dashboard.

Usage:
    streamlit run reference/python_pairstrading/pairs_dashboard.py --server.port 8502
"""

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

PROGRESS_FILE = Path(__file__).resolve().parent.parent.parent / "docs" / "pairs-progress.json"

st.set_page_config(page_title="Full Pairs WFA Monitor", layout="wide")
st.title("Full Pairs Trading — Live Monitor")


@st.fragment(run_every=3)
def live_charts():
    if not PROGRESS_FILE.exists():
        st.info("Waiting for pairs WFA to start... (no progress file yet)")
        st.code("bash reference/python_pairstrading/run_pairs_only.sh")
        return

    raw = json.loads(PROGRESS_FILE.read_text())
    dates = pd.to_datetime(raw["dates"])
    n = len(dates)

    if n == 0:
        st.info("Pairs WFA started but no data points yet...")
        return

    equity = raw["pairs_equity"]
    dd = raw["pairs_dd"]

    cum_ret = equity[-1] / equity[0] - 1 if equity[0] > 0 else 0
    max_dd = min(dd) if dd else 0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Progress", raw.get("pct", f"{n} days"))
    col2.metric("Latest Date", dates[-1].strftime("%Y-%m-%d"))
    col3.metric("Cumulative Return", f"{cum_ret:.1%}")
    col4.metric("Max Drawdown", f"{max_dd:.1%}")

    # ── Equity Curve ──
    eq_fig = go.Figure()
    eq_fig.add_scatter(
        x=dates, y=equity, name="Full Pairs",
        line=dict(color="#8B5CF6", width=2),
    )
    eq_fig.add_hline(
        y=equity[0], line_dash="dot", line_color="gray",
        annotation_text=f"Start ${equity[0]:,.0f}",
    )
    eq_fig.update_layout(
        title=f"Full Pairs Trading Equity ({raw.get('pct', '')})",
        yaxis_title="Equity ($)", yaxis_type="log",
        height=500, template="plotly_white",
    )
    st.plotly_chart(eq_fig, use_container_width=True)

    # ── Drawdown Chart ──
    dd_fig = go.Figure()
    dd_fig.add_scatter(
        x=dates, y=dd, name="Drawdown",
        fill="tozeroy",
        line=dict(color="#8B5CF6", width=1),
        fillcolor="rgba(139, 92, 246, 0.2)",
    )
    dd_fig.update_layout(
        title="Drawdown",
        yaxis_title="Drawdown", yaxis_tickformat=".0%",
        height=300, template="plotly_white",
    )
    st.plotly_chart(dd_fig, use_container_width=True)

    st.caption(f"Auto-refreshes every 3 seconds • Reading from `{PROGRESS_FILE}`")


live_charts()
