"""Standalone Full Pairs Trading WFA runner (no hybrid, no regime switching).

Usage:
    source .venv/bin/activate
    python reference/python_pairstrading/run_pairs_only.py

Writes progress to docs/pairs-progress.json for Streamlit dashboard.
"""

import datetime
import json
import logging
import sys
import traceback
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

LOG_FILE = Path(__file__).resolve().parent.parent / "docs" / "pairs-run.log"
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="w"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("pairs")

from dotenv import load_dotenv
load_dotenv()

from pairs_eda import (
    ExaRunMode,
    default_gemini_backend,
    download_with_retry,
    fetch_sp500_constituents_table,
    fetch_sp500_sector_map,
)
from pairs_eda.rolling_phase2 import (
    RollingPhase2Config,
    RollingPhase2Input,
    run_phase2_rolling,
)

PROGRESS_FILE = Path(__file__).resolve().parent.parent / "docs" / "pairs-progress.json"
PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)

_dates: list[str] = []
_equity_y: list[float] = []
_dd_y: list[float] = []
_peak = 0.0


def save_progress(pct_label: str = ""):
    data = {
        "dates": _dates,
        "pairs_equity": _equity_y,
        "pairs_dd": _dd_y,
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
    log.info("Full Pairs Trading WFA Runner")
    log.info("=" * 60)

    log.info("[1/3] Fetching S&P 500 constituents...")
    exa_backend = default_gemini_backend()
    sp500 = fetch_sp500_constituents_table(
        on_failure="exa", exa_backend=exa_backend,
        exa_mode=ExaRunMode.LIVE, verbose=True,
    )
    sp500_list = sp500["Symbol"].tolist()
    sp500_sector_map = fetch_sp500_sector_map(verbose=True)
    log.info(f"  {len(sp500_list)} symbols, {len(sp500_sector_map)} with sector data")

    log.info("[2/3] Downloading price data...")
    DOWNLOAD_START = "1990-01-01"
    sp500_daily_prices = download_with_retry(
        sp500_list, start=DOWNLOAD_START,
        end=datetime.datetime.today(), interval="1d",
        progress=True, threads=True, auto_adjust=False, max_retries=2,
    )
    log.info(f"  {sp500_daily_prices.shape[1]} tickers, {sp500_daily_prices.shape[0]} trading days")

    log.info("[3/3] Configuring & running full pairs WFA...")
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
    wfa_input = RollingPhase2Input(
        prices=sp500_daily_prices,
        initial_capital=initial_capital,
        config=wfa_config,
        sector_map=sp500_sector_map,
    )

    grid_size = len(wfa_config.windows) * len(wfa_config.zscore_thresholds)
    log.info(f"  Grid: {len(wfa_config.windows)} windows × {len(wfa_config.zscore_thresholds)} z-thresholds = {grid_size} combos")
    log.info(f"  Capital: ${initial_capital:,.0f}, Leverage: {wfa_config.leverage:.0f}x")
    log.info(f"  Progress file: {PROGRESS_FILE}")

    result = run_phase2_rolling(wfa_input, on_step=on_step, step_interval=1)

    save_progress("Complete")
    print()

    log.info("")
    log.info("=" * 60)
    log.info("RESULTS")
    log.info("=" * 60)
    log.info(f"Total trades : {len(result.trades)}")
    log.info(f"Final equity : ${result.equity.iloc[-1]:,.0f}")
    cum_ret = result.equity.iloc[-1] / result.equity.iloc[0] - 1
    log.info(f"Cumulative   : {cum_ret:.1%}")

    result_file = PROGRESS_FILE.parent / "pairs-result.json"
    summary = {
        "total_trades": len(result.trades),
        "final_equity": float(result.equity.iloc[-1]),
        "cumulative_return": float(cum_ret),
    }
    result_file.write_text(json.dumps(summary, indent=2))
    log.info(f"Summary saved to {result_file}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log.error(traceback.format_exc())
        raise
