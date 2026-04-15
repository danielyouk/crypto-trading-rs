"""PIT-aware Hybrid Backtest: S&P 500 in bull, PIT Pairs Trading in bear.

Same hybrid logic as run_wfa.py but with point-in-time S&P 500 membership
and EODHD-enriched price data (1,108 tickers including delisted/bankrupt).

Usage:
    source .venv/bin/activate
    python runners/run_wfa_pit.py

Writes progress to docs/wfa-pit-progress.json for dashboard overlay.
"""

import json
import logging
import sys
import traceback
from pathlib import Path

import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

LOG_FILE = Path(__file__).resolve().parent.parent / "docs" / "wfa-pit-run.log"
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="w"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("wfa-pit")

from dotenv import load_dotenv
load_dotenv()

from pairs_eda.sp500_history import Sp500History, download_all_historical_prices
from pairs_eda.rolling_phase2 import (
    RollingPhase2Config,
    RollingPhase2Input,
    run_hybrid_backtest,
)

PROGRESS_FILE = Path(__file__).resolve().parent.parent / "docs" / "wfa-pit-progress.json"
PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)

_dates: list[str] = []
_sp500_y: list[float] = []
_hybrid_y: list[float] = []
_sp500_dd_y: list[float] = []
_hybrid_dd_y: list[float] = []
_hybrid_peak = 0.0
_regime_events: list[dict] = []


def save_progress(pct_label: str = ""):
    data = {
        "dates": _dates,
        "sp500_equity": _sp500_y,
        "hybrid_equity": _hybrid_y,
        "sp500_dd": _sp500_dd_y,
        "hybrid_dd": _hybrid_dd_y,
        "regime_events": _regime_events,
        "pct": pct_label,
        "progress": len(_dates),
    }
    PROGRESS_FILE.write_text(json.dumps(data))


def on_step(day, equity, sp500_eq, sp500_dd, step_idx, total):
    global _hybrid_peak

    day_str = day.strftime("%Y-%m-%d")
    _dates.append(day_str)
    _sp500_y.append(sp500_eq)
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


def on_regime_change(event: dict):
    _regime_events.append(event)
    save_progress("")


def main():
    log.info("=" * 60)
    log.info("PIT Hybrid Backtest Runner")
    log.info("  S&P 500 (bull) + PIT Pairs Trading (bear)")
    log.info("  Survivorship-bias-free using EODHD data")
    log.info("=" * 60)

    log.info("[1/3] Loading PIT S&P 500 membership + enriched price data...")
    history = Sp500History.from_csv()
    summary = history.summary()
    for k, v in summary.items():
        log.info(f"  {k}: {v}")

    all_prices = download_all_historical_prices(history, verbose=True)
    log.info(f"  Price panel: {all_prices.shape[1]} tickers, {all_prices.shape[0]} days")

    log.info("[2/3] Configuring WFA...")
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
        max_ticker_exposure=1,
        min_spread_range_pct=0.03,
        commission_per_leg_bps=0.5,
        slippage_per_leg_bps=0.5,
    )

    initial_capital = 10_000.0
    universe_fn = history.universe_fn()
    log.info(f"  Universe: PIT membership ({summary['unique_tickers_ever']} unique tickers)")

    wfa_input = RollingPhase2Input(
        prices=all_prices,
        initial_capital=initial_capital,
        config=wfa_config,
        universe_fn=universe_fn,
    )

    grid_size = len(wfa_config.windows) * len(wfa_config.zscore_thresholds)
    log.info(f"  Grid: {len(wfa_config.windows)} windows x {len(wfa_config.zscore_thresholds)} z-thresholds = {grid_size} combos")

    log.info("[3/3] Running PIT hybrid backtest...")
    log.info("  Downloading SPY benchmark...")
    spy_raw = yf.download(
        "SPY", start=all_prices.index[0],
        end=all_prices.index[-1], progress=False,
    )
    sp500_benchmark = spy_raw["Close"].squeeze()
    sp500_benchmark.index = sp500_benchmark.index.tz_localize(None)

    ENTRY_DD = -0.15
    ENTRY_SLOPE_CONFIRM = 15
    EXIT_MA_WINDOW = 100
    EXIT_SLOPE_WINDOW = 20
    EXIT_SLOPE_CONFIRM = 15
    MIN_BEAR_DAYS = 60
    COOLDOWN_DAYS = 40
    PAIRS_CARRY_BPS = 0.0
    FX_HEDGE_CARRY_BPS = 0.0

    log.info(f"  Bear entry: DD ≤ {ENTRY_DD:.0%} AND slope < 0 (avg {ENTRY_SLOPE_CONFIRM}d)")
    log.info(f"  Bear exit:  slope > 0 (avg {EXIT_SLOPE_CONFIRM}d), min {MIN_BEAR_DAYS}d, cooldown {COOLDOWN_DAYS}d")
    log.info(f"  Progress file: {PROGRESS_FILE}")

    hybrid_result = run_hybrid_backtest(
        wfa_input, sp500_benchmark,
        entry_dd=ENTRY_DD,
        entry_slope_confirm_days=ENTRY_SLOPE_CONFIRM,
        exit_ma_window=EXIT_MA_WINDOW,
        exit_slope_window=EXIT_SLOPE_WINDOW,
        exit_slope_confirm_days=EXIT_SLOPE_CONFIRM,
        min_bear_days=MIN_BEAR_DAYS,
        cooldown_days=COOLDOWN_DAYS,
        pairs_carry_bps=PAIRS_CARRY_BPS,
        fx_hedge_carry_bps=FX_HEDGE_CARRY_BPS,
        on_step=on_step, on_regime_change=on_regime_change,
        step_interval=1,
    )

    save_progress("Complete")
    print()

    log.info("")
    log.info("=" * 60)
    log.info("RESULTS (PIT Hybrid)")
    log.info("=" * 60)
    log.info(f"Bear episodes   : {int(hybrid_result.summary['bear_episodes'])}")
    log.info(f"Days in pairs   : {int(hybrid_result.summary['days_in_pairs'])}/{int(hybrid_result.summary['days_total'])} ({hybrid_result.summary['pairs_pct']:.1%})")
    for k, v in hybrid_result.summary.items():
        if "pct" in k or "return" in k or "drawdown" in k:
            log.info(f"  {k:30s}  {v:>10.2%}")
        else:
            log.info(f"  {k:30s}  {v:>10.2f}")

    result_file = PROGRESS_FILE.parent / "wfa-pit-result.json"
    result_file.write_text(json.dumps(hybrid_result.summary, indent=2))
    log.info(f"Summary saved to {result_file}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log.error("FATAL:\n%s", traceback.format_exc())
        sys.exit(1)
