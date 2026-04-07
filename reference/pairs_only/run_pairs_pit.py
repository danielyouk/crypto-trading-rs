"""Survivorship-bias-free Pairs Trading WFA using point-in-time S&P 500 membership.

Uses the hanshof/sp500_constituents dataset to ensure each rebalance window
only considers tickers that were actually in the S&P 500 on that date.

Usage:
    source .venv/bin/activate
    python reference/python_pairstrading/run_pairs_pit.py

Writes progress to docs/pairs-pit-progress.json for Streamlit dashboard.
"""

import datetime
import json
import logging
import sys
import traceback
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

LOG_FILE = Path(__file__).resolve().parent.parent / "docs" / "pairs-pit-run.log"
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="w"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("pairs-pit")

from dotenv import load_dotenv
load_dotenv()

from pairs_eda.sp500_history import Sp500History, download_all_historical_prices
from pairs_eda.rolling_phase2 import (
    RollingPhase2Config,
    RollingPhase2Input,
    run_phase2_rolling,
)

PROGRESS_FILE = Path(__file__).resolve().parent.parent / "docs" / "pairs-pit-progress.json"
PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)

_dates: list[str] = []
_equity_y: list[float] = []
_dd_y: list[float] = []
_peak = 0.0


def save_progress(pct_label: str = ""):
    data = {
        "dates": _dates,
        "pit_equity": _equity_y,
        "pit_dd": _dd_y,
        "pct": pct_label,
        "progress": len(_dates),
    }
    PROGRESS_FILE.write_text(json.dumps(data))


def on_step(day, equity, step_idx, total):
    global _peak

    day_str = day.strftime("%Y-%m-%d")
    _dates.append(day_str)
    _equity_y.append(equity)

    _peak = max(_peak, equity)
    dd = equity / _peak - 1.0 if _peak > 0 else 0.0
    _dd_y.append(dd)

    pct = (step_idx + 1) / total * 100
    label = f"{pct:.0f}% ({day.strftime('%Y-%m')})"

    if step_idx % 10 == 0 or step_idx == total - 1:
        save_progress(label)
        print(f"\r  Progress: {label}", end="", flush=True)


def main():
    log.info("=" * 60)
    log.info("Point-in-Time (PIT) Pairs Trading WFA Runner")
    log.info("  Survivorship-bias-free backtest")
    log.info("=" * 60)

    log.info("[1/3] Loading historical S&P 500 membership...")
    history = Sp500History.from_csv()
    summary = history.summary()
    for k, v in summary.items():
        log.info(f"  {k}: {v}")

    log.info("[2/3] Downloading/loading price data for ALL historical tickers...")
    all_prices = download_all_historical_prices(history, verbose=True)
    log.info(f"  {all_prices.shape[1]} tickers, {all_prices.shape[0]} trading days")

    log.info("[3/3] Configuring & running PIT pairs WFA...")
    wfa_config = RollingPhase2Config(
        training_months=36,
        expanding_window=False,
        validation_days=180,
        rebalance_frequency="MS",
        coint_significance=0.10,
        coint_retest_margin=0.02,
        min_correlation=0.40,
        max_correlation=0.85,
        min_overlap_pct=0.80,
        top_n_candidates=200,
        windows=tuple(range(10, 32, 2)),
        zscore_thresholds=tuple(round(1.0 + i * 0.1, 1) for i in range(16)),
        stress_test_window_step=2,
        stress_test_zscore_step=0.1,
        watchlist_size=200,
        max_slots=10,
        max_new_entries_per_day=3,
        leverage=3.0,
        max_drop_quantile=0.90,
        entry_zscore_default=2.0,
        exit_zscore=0.0,
        stop_loss_pct=0.08,
        min_holding_days=3,
        circuit_breaker_pct=0.12,
        min_entry_score=0.3,
        max_sector_slots=3,
        min_spread_range_pct=0.03,
        commission_per_leg_bps=0.5,
        slippage_per_leg_bps=0.5,
    )

    initial_capital = 10_000.0

    universe_fn = history.universe_fn()
    log.info(f"  Universe function: PIT membership lookup ({summary['unique_tickers_ever']} unique tickers)")

    wfa_input = RollingPhase2Input(
        prices=all_prices,
        initial_capital=initial_capital,
        config=wfa_config,
        universe_fn=universe_fn,
    )

    grid_size = len(wfa_config.windows) * len(wfa_config.zscore_thresholds)
    log.info(f"  Grid: {len(wfa_config.windows)} windows x {len(wfa_config.zscore_thresholds)} z-thresholds = {grid_size} combos")
    log.info(f"  Capital: ${initial_capital:,.0f}, Leverage: {wfa_config.leverage:.0f}x")
    log.info(f"  Progress file: {PROGRESS_FILE}")

    result = run_phase2_rolling(wfa_input, on_step=on_step, step_interval=1)

    save_progress("Complete")
    print()

    log.info("")
    log.info("=" * 60)
    log.info("RESULTS (Point-in-Time, survivorship-bias-free)")
    log.info("=" * 60)
    log.info(f"Total trades : {len(result.trades)}")
    final_eq = result.daily_equity.iloc[-1]
    start_eq = result.daily_equity.iloc[0]
    cum_ret = final_eq / start_eq - 1
    log.info(f"Final equity : ${final_eq:,.0f}")
    log.info(f"Cumulative   : {cum_ret:.1%}")

    result_file = PROGRESS_FILE.parent / "pairs-pit-result.json"
    result_summary = {
        "total_trades": len(result.trades),
        "final_equity": float(final_eq),
        "cumulative_return": float(cum_ret),
        "max_drawdown": float(result.summary.get("max_drawdown", 0)),
    }
    result_file.write_text(json.dumps(result_summary, indent=2))
    log.info(f"Summary saved to {result_file}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log.error(traceback.format_exc())
        raise
