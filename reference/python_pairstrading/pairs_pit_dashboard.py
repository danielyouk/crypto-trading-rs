"""PIT (survivorship-bias-free) Pairs Trading dashboard with optional biased comparison.

Shows the honest backtest results using point-in-time S&P 500 membership.
If the biased run (run_pairs_only.py) results are available, overlays them
for a direct comparison — the key teaching visual.

Usage:
    streamlit run reference/python_pairstrading/pairs_pit_dashboard.py --server.port 8503
"""

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

PIT_PROGRESS = Path(__file__).resolve().parent.parent.parent / "docs" / "pairs-pit-progress.json"
BIASED_PROGRESS = Path(__file__).resolve().parent.parent.parent / "docs" / "pairs-progress.json"

st.set_page_config(page_title="PIT Pairs WFA — Survivorship Bias Fix", layout="wide")
st.title("Pairs Trading — Survivorship Bias Comparison")
st.caption(
    "**Left (purple):** Biased backtest (2026 S&P 500 list applied to 1996+) · "
    "**Right (green):** Honest backtest (point-in-time membership per rebalance date)"
)


def _load(path: Path) -> dict | None:
    if not path.exists():
        return None
    raw = json.loads(path.read_text())
    if not raw.get("dates"):
        return None
    return raw


@st.fragment(run_every=3)
def live_charts():
    pit = _load(PIT_PROGRESS)
    biased = _load(BIASED_PROGRESS)

    if pit is None:
        st.info("Waiting for PIT WFA to start... (no progress file yet)")
        st.code("bash reference/python_pairstrading/run_pairs_pit.sh")
        return

    pit_dates = pd.to_datetime(pit["dates"])
    pit_equity = pit["pit_equity"]
    pit_dd = pit["pit_dd"]

    pit_ret = pit_equity[-1] / pit_equity[0] - 1 if pit_equity[0] > 0 else 0
    pit_max_dd = min(pit_dd) if pit_dd else 0

    # ── Metrics ──
    cols = st.columns(5)
    cols[0].metric("PIT Progress", pit.get("pct", f"{len(pit_dates)} days"))
    cols[1].metric("PIT Cumulative Return", f"{pit_ret:.1%}")
    cols[2].metric("PIT Max Drawdown", f"{pit_max_dd:.1%}")

    if biased:
        biased_equity = biased["pairs_equity"]
        biased_ret = biased_equity[-1] / biased_equity[0] - 1 if biased_equity[0] > 0 else 0
        biased_dd = biased["pairs_dd"]
        biased_max_dd = min(biased_dd) if biased_dd else 0
        cols[3].metric("Biased Return", f"{biased_ret:.1%}")
        cols[4].metric("Bias Inflation", f"{biased_ret / pit_ret:.1f}x" if pit_ret > 0 else "N/A")
    else:
        cols[3].metric("Biased Return", "N/A (run run_pairs_only.py)")
        cols[4].metric("Bias Inflation", "N/A")

    # ── Equity Curve ──
    eq_fig = go.Figure()

    if biased:
        biased_dates = pd.to_datetime(biased["dates"])
        eq_fig.add_scatter(
            x=biased_dates, y=biased["pairs_equity"],
            name="Biased (2026 list)",
            line=dict(color="#8B5CF6", width=2, dash="dot"),
            opacity=0.7,
        )

    eq_fig.add_scatter(
        x=pit_dates, y=pit_equity,
        name="PIT (honest)",
        line=dict(color="#10B981", width=2),
    )

    eq_fig.add_hline(
        y=pit_equity[0], line_dash="dot", line_color="gray",
        annotation_text=f"Start ${pit_equity[0]:,.0f}",
    )
    eq_fig.update_layout(
        title="Equity Curves — Biased vs. Point-in-Time (log scale)",
        yaxis_title="Equity ($)", yaxis_type="log",
        height=550, template="plotly_white",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    )
    st.plotly_chart(eq_fig, use_container_width=True)

    # ── Drawdown ──
    dd_fig = go.Figure()

    if biased:
        dd_fig.add_scatter(
            x=biased_dates, y=biased["pairs_dd"],
            name="Biased DD",
            line=dict(color="#8B5CF6", width=1, dash="dot"),
            opacity=0.5,
        )

    dd_fig.add_scatter(
        x=pit_dates, y=pit_dd,
        name="PIT DD",
        fill="tozeroy",
        line=dict(color="#10B981", width=1),
        fillcolor="rgba(16, 185, 129, 0.15)",
    )
    dd_fig.update_layout(
        title="Drawdown Comparison",
        yaxis_title="Drawdown", yaxis_tickformat=".0%",
        height=300, template="plotly_white",
    )
    st.plotly_chart(dd_fig, use_container_width=True)

    # ── Summary table ──
    if biased and pit_ret > 0:
        st.subheader("Survivorship Bias Impact")
        comparison = pd.DataFrame({
            "Metric": ["Cumulative Return", "Max Drawdown", "Bias Inflation Factor"],
            "Biased (2026 list)": [f"{biased_ret:.1%}", f"{biased_max_dd:.1%}", "—"],
            "PIT (honest)": [f"{pit_ret:.1%}", f"{pit_max_dd:.1%}", "—"],
            "Difference": [
                f"{biased_ret - pit_ret:+.1%}",
                f"{biased_max_dd - pit_max_dd:+.1%}",
                f"{biased_ret / pit_ret:.1f}x",
            ],
        })
        st.dataframe(comparison, hide_index=True, use_container_width=True)

        st.info(
            f"**Survivorship bias inflated returns by {biased_ret / pit_ret:.1f}x.** "
            f"The biased backtest shows {biased_ret:.0%} vs. the honest {pit_ret:.0%}. "
            "The early period (1996–2005) accounts for most of the inflation."
        )

    st.caption(f"Auto-refreshes every 3 seconds • PIT: `{PIT_PROGRESS}` • Biased: `{BIASED_PROGRESS}`")


live_charts()
