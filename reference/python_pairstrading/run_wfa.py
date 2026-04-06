"""Standalone WFA + Hybrid Backtest runner.

Usage:
    source .venv/bin/activate
    python reference/python_pairstrading/run_wfa.py

Writes progress to docs/wfa-progress.json for Streamlit dashboard.
"""

import datetime
import json
import logging
import sys
import traceback
from pathlib import Path

import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "python"))

LOG_FILE = Path(__file__).resolve().parent.parent.parent / "docs" / "wfa-run.log"
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="w"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("wfa")

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
    run_hybrid_backtest,
)

# ── Progress file ──
PROGRESS_FILE = Path(__file__).resolve().parent.parent.parent / "docs" / "wfa-progress.json"
PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)

_sp500_x: list[str] = []
_sp500_y: list[float] = []
_hybrid_x: list[str] = []
_hybrid_y: list[float] = []
_sp500_dd_y: list[float] = []
_hybrid_dd_y: list[float] = []
_hybrid_peak = 0.0


def save_progress(pct_label: str = ""):
    data = {
        "dates": _sp500_x,
        "sp500_equity": _sp500_y,
        "hybrid_equity": _hybrid_y,
        "sp500_dd": _sp500_dd_y,
        "hybrid_dd": _hybrid_dd_y,
        "pct": pct_label,
        "progress": len(_sp500_x),
    }
    PROGRESS_FILE.write_text(json.dumps(data))


def on_step(day, equity, sp500_eq, sp500_dd, step_idx, total):
    global _hybrid_peak

    day_str = day.strftime("%Y-%m-%d")
    _sp500_x.append(day_str)
    _sp500_y.append(sp500_eq)
    _hybrid_x.append(day_str)
    _hybrid_y.append(equity)

    _hybrid_peak = max(_hybrid_peak, equity)
    hybrid_dd = equity / _hybrid_peak - 1.0 if _hybrid_peak > 0 else 0.0
    _sp500_dd_y.append(sp500_dd)
    _hybrid_dd_y.append(hybrid_dd)

    pct = (step_idx + 1) / total * 100
    label = f"{pct:.0f}% ({day.strftime('%Y-%m')})"

    if step_idx % 10 == 0 or step_idx == total - 1:
        save_progress(label)
        print(f"\r  Progress: {label}", end="", flush=True)


def main():
    log.info("=" * 60)
    log.info("WFA Hybrid Backtest Runner")
    log.info("=" * 60)

    log.info("[1/4] Fetching S&P 500 constituents...")
    exa_backend = default_gemini_backend()
    sp500 = fetch_sp500_constituents_table(
        on_failure="exa", exa_backend=exa_backend,
        exa_mode=ExaRunMode.LIVE, verbose=True,
    )
    sp500_list = sp500["Symbol"].tolist()
    sp500_sector_map = fetch_sp500_sector_map(verbose=True)
    log.info(f"  {len(sp500_list)} symbols, {len(sp500_sector_map)} with sector data")

    log.info("[2/4] Downloading price data...")
    DOWNLOAD_START = "1990-01-01"
    sp500_daily_prices = download_with_retry(
        sp500_list, start=DOWNLOAD_START,
        end=datetime.datetime.today(), interval="1d",
        progress=True, threads=True, auto_adjust=False, max_retries=2,
    )
    log.info(f"  {sp500_daily_prices.shape[1]} tickers, {sp500_daily_prices.shape[0]} trading days")

    log.info("[3/4] Configuring WFA...")
    wfa_config = RollingPhase2Config(
        training_months=36,
        expanding_window=False,
        validation_days=180,
        rebalance_frequency="MS",
        coint_significance=0.05,
        coint_retest_margin=0.02,
        min_correlation=0.40,
        max_correlation=0.85,
        min_overlap_pct=0.90,
        top_n_candidates=200,
        windows=tuple(range(10, 32, 2)),
        zscore_thresholds=tuple(round(1.5 + i * 0.1, 1) for i in range(16)),
        watchlist_size=20,
        max_slots=7,
        max_new_entries_per_day=2,
        leverage=3.0,
        max_drop_quantile=0.90,
        entry_zscore_default=2.0,
        exit_zscore=0.0,
        stop_loss_pct=0.05,
        min_holding_days=3,
        circuit_breaker_pct=0.12,
        min_entry_score=0.5,
        max_sector_slots=2,
        min_spread_range_pct=0.05,
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

    log.info("[4/4] Running hybrid backtest...")
    log.info("  Downloading SPY benchmark...")
    spy_raw = yf.download(
        "SPY", start=sp500_daily_prices.index[0],
        end=sp500_daily_prices.index[-1], progress=False,
    )
    sp500_benchmark = spy_raw["Close"].squeeze()
    sp500_benchmark.index = sp500_benchmark.index.tz_localize(None)

    ENTRY_DD = -0.10
    EXIT_MA_WINDOW = 100
    EXIT_SLOPE_WINDOW = 20
    EXIT_SLOPE_CONFIRM = 15
    MIN_BEAR_DAYS = 60
    COOLDOWN_DAYS = 40
    PAIRS_CARRY_BPS = 200.0     # margin rate - short rebate ≈ 2%/yr
    FX_HEDGE_CARRY_BPS = 350.0  # realistic IBKR margin spread ≈ 3.5%/yr (always a cost)

    log.info(f"  Bear entry: drawdown ≤ {ENTRY_DD:.0%}")
    log.info(f"  Bear exit:  {EXIT_MA_WINDOW}d MA slope (avg over {EXIT_SLOPE_CONFIRM}d) > 0, "
             f"min {MIN_BEAR_DAYS}d in bear, {COOLDOWN_DAYS}d cooldown")
    log.info(f"  Carry costs: pairs={PAIRS_CARRY_BPS:.0f}bps/yr, FX hedge={FX_HEDGE_CARRY_BPS:.0f}bps/yr")
    log.info(f"  Progress file: {PROGRESS_FILE}")

    hybrid_result = run_hybrid_backtest(
        wfa_input, sp500_benchmark,
        entry_dd=ENTRY_DD,
        exit_ma_window=EXIT_MA_WINDOW,
        exit_slope_window=EXIT_SLOPE_WINDOW,
        exit_slope_confirm_days=EXIT_SLOPE_CONFIRM,
        min_bear_days=MIN_BEAR_DAYS,
        cooldown_days=COOLDOWN_DAYS,
        pairs_carry_bps=PAIRS_CARRY_BPS,
        fx_hedge_carry_bps=FX_HEDGE_CARRY_BPS,
        on_step=on_step, step_interval=1,
    )

    save_progress("Complete")

    log.info("")
    log.info("=" * 60)
    log.info("RESULTS")
    log.info("=" * 60)
    log.info(f"Bear episodes   : {int(hybrid_result.summary['bear_episodes'])}")
    log.info(f"Days in pairs   : {int(hybrid_result.summary['days_in_pairs'])}/{int(hybrid_result.summary['days_total'])} ({hybrid_result.summary['pairs_pct']:.1%})")
    log.info(f"{'─' * 50}")
    for k, v in hybrid_result.summary.items():
        if "pct" in k or "return" in k or "drawdown" in k:
            log.info(f"  {k:30s}  {v:>10.2%}")
        else:
            log.info(f"  {k:30s}  {v:>10.2f}")

    result_file = PROGRESS_FILE.parent / "wfa-result.json"
    result_file.write_text(json.dumps(hybrid_result.summary, indent=2))
    log.info(f"Final summary saved to {result_file}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log.error("FATAL: unhandled exception\n%s", traceback.format_exc())
        sys.exit(1)
