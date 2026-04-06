"""Live WFA progress dashboard — run with: streamlit run wfa_dashboard.py"""

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

PROGRESS_FILE = Path(__file__).resolve().parent.parent.parent / "docs" / "wfa-progress.json"

st.set_page_config(page_title="WFA Live Monitor", layout="wide")
st.title("Hybrid Backtest — Live Monitor")

COLORS = {"sp500": "orange", "hybrid": "green", "pairs": "#8B5CF6"}


def _add_bear_bands(fig, events, show_labels=True):
    """Add red shaded bands for bear (pairs trading) episodes."""
    entries = [e for e in events if e["type"] == "bear_entry"]
    exits = [e for e in events if e["type"] == "bear_exit"]

    exit_iter = iter(exits)
    for i, entry in enumerate(entries):
        x0 = pd.Timestamp(entry["date"])
        matching_exit = None
        for ex in exit_iter:
            if ex["date"] >= entry["date"]:
                matching_exit = ex
                break

        x1 = pd.Timestamp(matching_exit["date"]) if matching_exit else x0

        fig.add_vrect(
            x0=x0, x1=x1,
            fillcolor="rgba(220, 50, 50, 0.10)",
            layer="below", line_width=0,
        )

        if show_labels:
            bear_days = matching_exit.get("bear_days", "?") if matching_exit else "?"
            trades = entry.get("wfa_trades", "?")
            fig.add_annotation(
                x=x0 + (x1 - x0) / 2,
                y=1.0, yref="paper",
                text=f"<b>BEAR #{i+1}</b><br>{bear_days}d, {trades} trades",
                showarrow=False,
                font=dict(size=8, color="white"),
                bgcolor="rgba(180, 40, 40, 0.8)",
                borderpad=3,
            )


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

    regime_events = raw.get("regime_events", [])
    has_pairs = "pairs_equity" in raw and len(raw["pairs_equity"]) > 0

    # ── Metrics ──
    cols = st.columns(5 if has_pairs else 4)
    cols[0].metric("Progress", f"{n} days")
    cols[1].metric("Latest Date", dates[-1].strftime("%Y-%m-%d"))

    hybrid_ret = raw["hybrid_equity"][-1] / raw["hybrid_equity"][0] - 1 if raw["hybrid_equity"][0] > 0 else 0
    cols[2].metric("Hybrid Return", f"{hybrid_ret:.1%}")

    sp500_ret = raw["sp500_equity"][-1] / raw["sp500_equity"][0] - 1 if raw["sp500_equity"][0] > 0 else 0
    cols[3].metric("S&P 500 Return", f"{sp500_ret:.1%}")

    if has_pairs:
        valid_pairs = [v for v in raw["pairs_equity"] if v == v]  # filter NaN
        if valid_pairs:
            pairs_ret = valid_pairs[-1] / valid_pairs[0] - 1 if valid_pairs[0] > 0 else 0
            cols[4].metric("Full Pairs Return", f"{pairs_ret:.1%}")

    n_bear = len([e for e in regime_events if e["type"] == "bear_entry"])

    # ── Equity Curve ──
    eq_fig = go.Figure()
    eq_fig.add_scatter(x=dates, y=raw["sp500_equity"], name="S&P 500",
                       line=dict(color=COLORS["sp500"], width=1), opacity=0.7)
    eq_fig.add_scatter(x=dates, y=raw["hybrid_equity"], name="Hybrid",
                       line=dict(color=COLORS["hybrid"], width=2.5))
    if has_pairs:
        eq_fig.add_scatter(x=dates, y=raw["pairs_equity"], name="Full Pairs",
                           line=dict(color=COLORS["pairs"], width=1.5, dash="dot"),
                           opacity=0.8)

    if regime_events:
        _add_bear_bands(eq_fig, regime_events, show_labels=True)

    legend_note = f"  —  <span style='color:rgba(180,40,40,0.6)'>■</span> Bear Episodes ({n_bear})"
    eq_fig.update_layout(
        title=f"Equity Curves ({raw.get('pct', '')}){legend_note}",
        yaxis_title="Equity ($)", yaxis_type="log",
        height=500, template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    st.plotly_chart(eq_fig, use_container_width=True)

    # ── Drawdown Chart ──
    dd_fig = go.Figure()
    dd_fig.add_scatter(x=dates, y=raw["sp500_dd"], name="S&P 500 DD",
                       fill="tozeroy", line=dict(color=COLORS["sp500"], width=0.8),
                       fillcolor="rgba(255,165,0,0.15)")
    dd_fig.add_scatter(x=dates, y=raw["hybrid_dd"], name="Hybrid DD",
                       line=dict(color=COLORS["hybrid"], width=1.2))
    if has_pairs and "pairs_dd" in raw:
        dd_fig.add_scatter(x=dates, y=raw["pairs_dd"], name="Full Pairs DD",
                           line=dict(color=COLORS["pairs"], width=1, dash="dot"),
                           opacity=0.8)
    dd_fig.add_hline(y=-0.15, line_dash="dash", line_color="red",
                     annotation_text="Bear entry (-15%)")

    if regime_events:
        _add_bear_bands(dd_fig, regime_events, show_labels=False)

    dd_fig.update_layout(
        title="Drawdown",
        yaxis_title="Drawdown", yaxis_tickformat=".0%",
        height=350, template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    st.plotly_chart(dd_fig, use_container_width=True)

    # ── Regime Events Table ──
    if regime_events:
        st.subheader("Regime Transitions")
        events_df = pd.DataFrame(regime_events)
        display_cols = ["date", "type", "reason"]
        extra = [c for c in ["wfa_trades", "active_days", "bear_days", "equity"] if c in events_df.columns]
        st.dataframe(events_df[display_cols + extra], use_container_width=True, hide_index=True)

    st.caption(f"Auto-refreshes every 3 seconds • Reading from `{PROGRESS_FILE}`")


live_charts()
