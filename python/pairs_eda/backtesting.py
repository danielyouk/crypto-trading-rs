"""Pairs-trading backtesting functions (functional style).

Pipeline:
    compute_zscore  ->  compute_signals  ->  summarize_signals  ->  calculate_margin
                                                                         |
    run_pair_pipeline (chains all four, returns PairPipelineState with all intermediates)
    grid_search_pair  (runs run_pair_pipeline across a parameter grid)

Intraday variant:
    compute_zscore_intraday  ->  (same signal / summary / margin functions)
    backtest_pair_intraday   (chains the intraday variant)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
import math
from typing import Iterable, Optional, cast

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


def _column_as_series(frame: pd.DataFrame, column: str) -> pd.Series:
    """Return a typed Series for a DataFrame column access."""
    return cast(pd.Series, frame[column])


def _unrealized_pnl(
    sig: int,
    s1_units: float,
    s2_units: float,
    s1_start: float,
    s2_start: float,
    s1_current: float,
    s2_current: float,
) -> float:
    """Mark-to-market unrealized P&L for an open pairs position.

    Used only for the stop-loss check (no slippage applied).
    Slippage is applied at actual exit when computing the final P&L.

    sig == 1  : long stock1, short stock2
    sig == -1 : long stock2, short stock1
    """
    if sig == 1:
        return (s1_current - s1_start) * s1_units - (s2_current - s2_start) * s2_units
    return (s2_current - s2_start) * s2_units - (s1_current - s1_start) * s1_units


@dataclass(frozen=True)
class PairPipelineState:
    """All intermediate artifacts produced by run_pair_pipeline for one pair.

    Every stage of the pipeline is accessible as a direct attribute so you can
    inspect each step in a notebook without re-running anything:

        prices_a, prices_b
             │
             ▼  compute_zscore(window)
        pair_stats        — DataFrame, date-indexed
             │               stock1 | stock2 | ratio | ma | msd | zscore | signal
             │
             ▼  summarize_signals() + calculate_margin(margin_init, margin_ratio)
        signal_summary    — DataFrame, one row per contiguous signal period
             │               signal | time_start | time_end
             │               stock1_start_price | stock1_final_price
             │               stock2_start_price | stock2_final_price
             │               stock1_units* | stock2_units*
             │               pnl* | commission* | margin_after*
             │               stopped_out* | stop_date*
             │               (* NaN for neutral periods and post-liquidation rows)
             │
             ▼
        margin_final      — float  (final margin after all trades)
        liquidation_date  — str | None  (date simulation stopped; None if solvent)
        n_stops           — int   (number of trades cut short by stop-loss)

    Input parameters are stored verbatim for reference:
        pair, window, zscore_threshold, margin_init, margin_ratio
    """

    pair: tuple[str | None, str | None]
    window: int
    zscore_threshold: float
    margin_init: float
    margin_ratio: float
    pair_stats: pd.DataFrame
    signal_summary: pd.DataFrame
    margin_final: float
    liquidation_date: str | None
    n_stops: int  # number of trades cut short by stop-loss


# ---------------------------------------------------------------------------
# Z-Score
# ---------------------------------------------------------------------------

class KalmanParams(BaseModel):
    """Hyperparameters for 1-D local-level Kalman filter on log price ratio.

    The Kalman filter replaces Simple Moving Average (SMA) for Z-score
    computation.  Unlike SMA, which treats a 20 % structural jump as
    "contaminated" data for a full ``window`` days, the Kalman filter
    recognises regime changes within **1–3 bars** and resets its
    internal state to the new level.

    Model:
    ┌───────────────────────────────────────────────┐
    │  State equation:   x[t] = x[t-1] + w,  w ~ N(0, Q)      │
    │  Observation eq:   y[t] = x[t]   + v,  v ~ N(0, R)      │
    │                                                           │
    │  Q = process variance   (how fast the "true mean" drifts) │
    │  R = measurement noise  (how noisy each observation is)   │
    │  Q/R ratio = adaptiveness: higher → faster mean tracking  │
    └───────────────────────────────────────────────┘

    Auto-tuning (when Q and/or R are None):
        R = sample variance of the first ``window`` bars of log-ratio
        Q = R / window

        This yields a Kalman gain that roughly mirrors a ``window``-bar
        EMA in steady state, but adapts much faster after large
        innovations (jumps).

    Args:
        process_variance:        Q — state noise. None = auto (R / window).
        measurement_variance:    R — observation noise. None = auto (burn-in var).
        innovation_variance_floor: Minimum denominator S to prevent division
                                   by zero.  Default 1e-18.

    Example:
        >>> params = KalmanParams(process_variance=1e-5, measurement_variance=1e-3)
        >>> df = compute_zscore(prices_a, prices_b, window=60, kalman_params=params)
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    process_variance: Optional[float] = Field(
        default=None,
        description="State noise Q. If None, auto-calculated as R / window.",
    )
    measurement_variance: Optional[float] = Field(
        default=None,
        description="Observation noise R. If None, auto-calculated from "
        "variance of first `window` bars.",
    )
    innovation_variance_floor: float = Field(
        default=1e-18,
        ge=0.0,
        description="Minimum S to avoid division by zero.",
    )


def _kalman_filter_loop(
    y: np.ndarray,
    window: int,
    q: Optional[float],
    r: Optional[float],
    floor_s: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """1-D local-level Kalman filter producing predictive mean, std, and Z-score.

    ┌─────────────────────────────────────────────────────────────────┐
    │  For each bar t:                                               │
    │                                                                │
    │  PREDICT  x_pred = x_filt[t-1]           (state propagation)  │
    │           p_pred = p_filt[t-1] + Q       (uncertainty grows)   │
    │           S      = p_pred + R            (innovation variance) │
    │                                                                │
    │  Z-SCORE  z[t]   = (y[t] - x_pred) / √S  (if t >= window)    │
    │                                                                │
    │  UPDATE   K      = p_pred / S            (Kalman gain)         │
    │           x_filt = x_pred + K·(y[t] - x_pred)                 │
    │           p_filt = (1 - K) · p_pred                            │
    └─────────────────────────────────────────────────────────────────┘

    Why Kalman gain matters for drift recovery:
        After a structural jump, the innovation (y[t] - x_pred) is large.
        Because K = p_pred / (p_pred + R), the filter pulls x_filt towards
        the new observed level in proportion to the surprise.  Typical
        steady-state K ≈ 0.10–0.25 for our Q/R ratios, meaning the filter
        absorbs 60–90 % of a jump within 5–10 bars — far faster than the
        ``window``-bar delay inherent in SMA.

    Auto-tuning (when Q/R are None):
        R = Var(y[:window])  — estimated from burn-in only (no lookahead)
        Q = R / window       — yields steady-state gain ≈ 1/√window

    Args:
        y:       Log price ratio array, may contain NaN.
        window:  Burn-in size; Z-scores are NaN for t < window.
        q:       Process variance Q (None = auto).
        r:       Measurement variance R (None = auto).
        floor_s: Minimum innovation variance S to prevent div-by-zero.

    Returns:
        (pred_mean, pred_std, z_out) — each np.ndarray of length n.
        pred_mean[t] = E[ratio[t] | data up to t-1]   (predictive mean)
        pred_std[t]  = √Var(ratio[t] | data up to t-1) (innovation std)
        z_out[t]     = normalised innovation (NaN for t < window)

    Performance notes:
        PERF-001: Python for-loop over n bars. Cannot vectorise because each
                  step depends on the previous step's filtered state (x_filt,
                  p_filt). A Cython/Numba JIT would give ~50× speedup.
    """
    n = y.size
    pred_mean = np.full(n, np.nan, dtype=np.float64)
    pred_std = np.full(n, np.nan, dtype=np.float64)
    z_out = np.full(n, np.nan, dtype=np.float64)

    # Auto-tune R and Q using ONLY the burn-in window (no lookahead)
    if r is None or q is None:
        w = max(int(window), 1)
        m = min(w, n)
        seg = y[:m]
        valid_seg = seg[np.isfinite(seg)]
        v = float(np.var(valid_seg, ddof=1)) if valid_seg.size >= 2 else 0.0
        auto_r = v if v > 1e-18 else 1e-12
        auto_q = auto_r / float(w)

        r = r if r is not None else auto_r
        q = q if q is not None else auto_q

    x_filt = 0.0
    p_filt = 1e4 * r  # diffuse prior — large initial uncertainty
    initialized = False

    for t in range(n):  # PERF-001: sequential dependency, not vectorisable
        yt = y[t]

        if not initialized:
            if np.isfinite(yt):
                x_filt = yt
                p_filt = r
                initialized = True
            continue

        # 1. Predict (Time Update)
        x_pred = x_filt
        p_pred = p_filt + q
        s = p_pred + r  # innovation variance S = P_pred + R

        pred_mean[t] = x_pred
        pred_std[t] = math.sqrt(s)

        # 2. Z-score: normalised innovation (only after burn-in)
        if t >= window and np.isfinite(yt) and s >= floor_s:
            z_out[t] = (yt - x_pred) / math.sqrt(s)

        # 3. Update (Measurement Update)
        if np.isfinite(yt) and s >= floor_s:
            k = p_pred / s  # Kalman gain K ∈ [0, 1]
            x_filt = x_pred + k * (yt - x_pred)
            p_filt = (1.0 - k) * p_pred
        else:
            # NaN observation: state unchanged, uncertainty grows
            x_filt = x_pred
            p_filt = p_pred

    return pred_mean, pred_std, z_out


def compute_zscore(
    prices_a: pd.Series,
    prices_b: pd.Series,
    window: int,
    kalman_params: Optional[KalmanParams] = None,
) -> pd.DataFrame:
    """Z-score from log price ratio via 1-D Kalman filter (adaptive mean tracker).

    ┌──────────────────────────────────────────────────────────────────┐
    │  ratio[t]  = log(price_a[t] / price_b[t])                      │
    │  ma[t]     = E[ratio[t] | y₁…y_{t-1}]       (predictive mean)  │
    │  msd[t]    = √Var(ratio[t] | y₁…y_{t-1})    (innovation std)   │
    │  zscore[t] = (ratio[t] − ma[t]) / msd[t]    (normalised innov) │
    └──────────────────────────────────────────────────────────────────┘

    Why Kalman instead of SMA:
        SMA treats every observation equally within the window.  After a
        structural jump (e.g. earnings surprise +20 %), the rolling mean
        takes a full ``window`` days to "catch up", producing a persistent
        ghost Z-score signal the entire time.

        The Kalman filter recognises that the large innovation is a *regime
        change*, not noise.  It pulls its internal state towards the new
        level within 1–3 bars, so the Z-score returns to the [-1, +1]
        neutral band within **5–10 trading days** instead of ``window`` days.

    Flow:
        prices_a, prices_b
             │  log(a / b)
             ▼
        ratio  (n × 1)
             │  _kalman_filter_loop(ratio, window, Q, R)
             ▼
        ma, msd, zscore   ← all NaN for t < window (burn-in)

    Args:
        prices_a:      Adj Close series for stock A (DatetimeIndex).
        prices_b:      Adj Close series for stock B (same index as prices_a).
        window:        Burn-in window size.  First ``window`` rows yield NaN
                       Z-scores.  Also used for auto-tuning Q and R.
        kalman_params: Optional KalmanParams.  If None, Q and R are
                       auto-tuned from the first ``window`` bars.

    Returns:
        DataFrame (same index as input) with columns:
            stock1, stock2, ratio, ma, msd, zscore.

    Raises:
        ValueError: If indexes have duplicates or do not match.

    Example:
        >>> df = compute_zscore(aapl, msft, window=60)
        >>> entries = df[df["zscore"].abs() > 2.0]
    """
    pa = prices_a.sort_index()
    pb = prices_b.sort_index()
    
    if pa.index.duplicated().any() or pb.index.duplicated().any():
        raise ValueError("prices_a and prices_b must have unique DatetimeIndex.")
    if not pa.index.equals(pb.index):
        raise ValueError("prices_a.index must equal prices_b.index.")

    params = kalman_params or KalmanParams()

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio_raw = np.log(pa.to_numpy(dtype=np.float64) / pb.to_numpy(dtype=np.float64))
        
    ratio = pd.Series(ratio_raw, index=pa.index)

    ma_arr, msd_arr, zscore_arr = _kalman_filter_loop(
        y=ratio_raw,
        window=window,
        q=params.process_variance,
        r=params.measurement_variance,
        floor_s=params.innovation_variance_floor,
    )

    return pd.DataFrame(
        {
            "stock1": pa,
            "stock2": pb,
            "ratio": ratio,
            "ma": ma_arr,
            "msd": msd_arr,
            "zscore": zscore_arr,
        },
        index=pa.index,
    )


def compute_zscore_intraday(
    prices_a_intraday: pd.Series,
    prices_b_intraday: pd.Series,
    prices_a_daily: pd.Series,
    prices_b_daily: pd.Series,
    window: int,
) -> pd.DataFrame:
    """Z-score using daily rolling stats applied to an intraday price ratio.

    Rolling mean and std are computed on daily data (look-ahead free),
    then merged onto each intraday bar via the calendar date.  The z-score
    uses the intraday log-ratio against the daily rolling stats.

    Args:
        prices_a_intraday: Intraday Adj Close for stock A.
        prices_b_intraday: Intraday Adj Close for stock B.
        prices_a_daily:    Daily Adj Close for stock A.
        prices_b_daily:    Daily Adj Close for stock B.
        window:            Rolling window size in trading days.

    Returns:
        DataFrame (intraday index) with columns:
            stock1, stock2, ratio, ma, msd, zscore.
    """
    intraday = pd.DataFrame(
        {"stock1": prices_a_intraday, "stock2": prices_b_intraday}
    )
    intraday["Day"] = cast(pd.DatetimeIndex, intraday.index).date  # type: ignore[attr-defined]
    intraday["ratio"] = np.log(intraday["stock1"] / intraday["stock2"])

    daily_ratio = np.log(prices_a_daily / prices_b_daily)
    daily = pd.DataFrame({"Day": cast(pd.DatetimeIndex, prices_a_daily.index).date})  # type: ignore[attr-defined]
    daily["ma"] = daily_ratio.rolling(window=window).mean().shift(1).values
    daily["msd"] = daily_ratio.rolling(window=window).std().shift(1).values

    merged = pd.merge(intraday, daily, on="Day", how="left")
    merged.index = intraday.index
    merged["zscore"] = (merged["ratio"] - merged["ma"]) / merged["msd"]

    return cast(pd.DataFrame, merged[["stock1", "stock2", "ratio", "ma", "msd", "zscore"]])


# ---------------------------------------------------------------------------
# Signals
# ---------------------------------------------------------------------------

def compute_signals(zscore: pd.Series, threshold: float) -> pd.Series:
    """Trading signals from z-score and threshold.

    Signal rules:
        zscore >  threshold  AND  zscore <  5   ->  -1  (A overvalued: short A, long B)
        zscore < -threshold  AND  zscore > -5   ->   1  (A undervalued: long A, short B)
        -1 < zscore < 1                         ->   0  (neutral / close positions)
        |zscore| >= 5                            ->  hold previous signal (extreme)
        deadband (between 1..threshold)          ->  hold previous signal

    NaN values are forward-filled, then remaining NaN filled with 0.

    Args:
        zscore:    Z-score series (output of compute_zscore).
        threshold: Entry threshold (e.g. 2.0).

    Returns:
        Series of integer signals: -1, 0, or 1.
    """
    signal = pd.Series(np.nan, index=zscore.index)
    signal = np.where(
        (zscore > threshold) & (zscore < 5), -1, signal
    )
    signal = np.where(
        (zscore < -threshold) & (zscore > -5), 1, signal
    )
    signal = np.where(
        (zscore > -1) & (zscore < 1), 0, signal
    )
    signal = pd.Series(signal, index=zscore.index)
    signal = signal.ffill().fillna(0)
    return signal


# ---------------------------------------------------------------------------
# Signal Summary
# ---------------------------------------------------------------------------

def summarize_signals(
    prices_a: pd.Series,
    prices_b: pd.Series,
    signals: pd.Series,
) -> list[dict]:
    """Group consecutive signal periods with start/end dates and prices.

    Each dict represents one contiguous period where the signal stayed constant:
        signal, time_start, time_end,
        stock1_start_price, stock1_final_price,
        stock2_start_price, stock2_final_price

    "Final price" of a period = start price of the NEXT period (the price at
    the moment the signal changes).  For the last period, final price = the
    last price in the data.

    Args:
        prices_a: Stock A prices (same index as signals).
        prices_b: Stock B prices (same index as signals).
        signals:  Signal series from compute_signals.

    Returns:
        List of dicts, one per signal period.
    """
    df = pd.DataFrame(
        {
            "stock1": prices_a.values,
            "stock2": prices_b.values,
            "signal": signals.values,
            "time": prices_a.index,
        },
        index=prices_a.index,
    )

    groups = df["signal"].diff().ne(0).cumsum()
    summary: pd.DataFrame = cast(
        pd.DataFrame,
        df.groupby(groups)
        .agg(
            signal=("signal", "first"),
            time_start=("time", "first"),
            stock1_start_price=("stock1", "first"),
            stock2_start_price=("stock2", "first"),
        )
        .reset_index(drop=True),
    )

    summary["time_end"] = _column_as_series(summary, "time_start").shift(-1)
    summary["stock1_final_price"] = _column_as_series(summary, "stock1_start_price").shift(-1)
    summary["stock2_final_price"] = _column_as_series(summary, "stock2_start_price").shift(-1)

    last_idx = summary.index[-1]
    summary.loc[last_idx, "time_end"] = df.index[-1]
    summary.loc[last_idx, "stock1_final_price"] = df["stock1"].iloc[-1]
    summary.loc[last_idx, "stock2_final_price"] = df["stock2"].iloc[-1]

    cols = [
        "signal",
        "time_start",
        "time_end",
        "stock1_start_price",
        "stock1_final_price",
        "stock2_start_price",
        "stock2_final_price",
    ]
    return cast(pd.DataFrame, summary[cols]).to_dict("records")


# ---------------------------------------------------------------------------
# Margin / P&L Calculation
# ---------------------------------------------------------------------------

_BUY_SLIPPAGE = 1.0003   # 3 pips adverse fill
_SELL_SLIPPAGE = 0.9997
_COMMISSION_PER_SHARE = 0.005
_MIN_COMMISSION = 1.0
_MAX_COMMISSION_PCT = 0.01
_SEC_FEE_RATE = 0.000008
_FINRA_TAF_PER_SHARE = 0.000166
# Margin below this value means both legs' minimum commissions ($1 each) cannot
# be covered, so the account is effectively bankrupt and simulation should stop.
_MIN_VIABLE_MARGIN = 2.0


def calculate_margin(
    signal_summary: list[dict],
    margin_init: float,
    margin_ratio: float,
    *,
    fractional: bool = True,
    stop_loss_pct: float = 0.0,
    pair_stats: pd.DataFrame | None = None,
    cooldown_days: int = 0,
) -> dict:
    """Simulate margin evolution over active signal periods.

    Commission model (Interactive Brokers US equities):
        Buy leg:  USD 0.005/share, min USD 1, max 1% of leg value.
        Sell leg: same + SEC Transaction Fee + FINRA TAF.

    Price slippage: 3 pips (buy at x1.0003, sell at x0.9997).

    Each active period (signal = +/-1) allocates 50% of buying power
    to each leg.  Buying power = current margin / margin_ratio.

    Stop-loss & cooldown:
        When stop_loss_pct > 0 and pair_stats is provided, each active trade
        is monitored day-by-day.  If unrealized P&L drops below
        ``-stop_loss_pct × buying_power``, the trade exits immediately at
        that day's close.

        After a stop-loss, the next ``cooldown_days`` trading bars are skipped
        — those bars would carry contaminated z-score history from the abnormal
        move that triggered the stop, so any signal during that window is
        ignored.

        Stop-loss check uses raw (no-slippage) prices; slippage is still
        applied at the actual exit when computing the final P&L.

    Args:
        signal_summary:  Output of summarize_signals.
        margin_init:     Starting collateral (e.g. 3000).
        margin_ratio:    Margin requirement (e.g. 0.25 → 4× leverage).
        fractional:      True = dollar-value sizing; False = integer shares.
        stop_loss_pct:   Fraction of buying_power; exit if unrealized loss
                         exceeds this.  0.0 = disabled (default).
        pair_stats:      Daily price DataFrame (stock1 / stock2 columns,
                         DatetimeIndex).  Required for stop-loss checks.
        cooldown_days:   Trading bars to skip after a stop-loss event.
                         Typically set to the z-score rolling window so that
                         contaminated history cycles out before re-entry.

    Returns:
        Dict with:
            margin           - final margin after all trades.
            trade_count      - number of active trades executed.
            trade_log        - list of dicts per trade.
            liquidation_date - date simulation stopped due to bankruptcy; None if solvent.
            n_stops          - number of trades exited early by stop-loss.
    """
    margin = margin_init
    buying_power = margin / margin_ratio
    trade_log: list[dict] = []
    liquidation_date: str | None = None
    cooldown_until: pd.Timestamp | None = None
    n_stops = 0

    for trade in signal_summary:
        sig = trade["signal"]
        if sig not in (1, -1):
            continue

        # Bankruptcy guard
        if margin < _MIN_VIABLE_MARGIN:
            liquidation_date = trade["time_start"]
            break

        # Cooldown guard: skip trades whose entry falls within the post-stop-loss
        # window.  The z-score window worth of bars needs to cycle out first.
        if cooldown_until is not None:
            if pd.Timestamp(trade["time_start"]) <= cooldown_until:
                continue

        s1_start = trade["stock1_start_price"]
        s1_final = trade["stock1_final_price"]
        s2_start = trade["stock2_start_price"]
        s2_final = trade["stock2_final_price"]

        half_bp = 0.5 * buying_power

        if fractional:
            s1_units = half_bp / s1_start
            s2_units = half_bp / s2_start
        else:
            s1_units = math.floor(half_bp / s1_start)
            s2_units = math.floor(half_bp / s2_start)

        # --- Stop-loss check (daily mark-to-market within the signal period) ---
        actual_s1_final = s1_final
        actual_s2_final = s2_final
        stopped_out = False
        stop_date: str | None = None

        if stop_loss_pct > 0.0 and pair_stats is not None:
            trade_start_ts = pd.Timestamp(trade["time_start"])
            trade_end_ts = pd.Timestamp(trade["time_end"])
            daily_window = pair_stats.loc[
                (pair_stats.index >= trade_start_ts) & (pair_stats.index < trade_end_ts),
                ["stock1", "stock2"],
            ]
            for ts, row in daily_window.iterrows():
                unr = _unrealized_pnl(
                    sig,
                    s1_units, s2_units,
                    s1_start, s2_start,
                    float(row["stock1"]), float(row["stock2"]),
                )
                if unr < -stop_loss_pct * buying_power:
                    actual_s1_final = float(row["stock1"])
                    actual_s2_final = float(row["stock2"])
                    stopped_out = True
                    stop_date = str(cast(pd.Timestamp, ts).date())
                    n_stops += 1
                    if cooldown_days > 0:
                        stop_pos = int(pair_stats.index.searchsorted(ts))
                        end_pos = min(stop_pos + cooldown_days, len(pair_stats) - 1)
                        cooldown_until = cast(pd.Timestamp, pair_stats.index[end_pos])
                    break

        # --- Commission (IB rates) ---
        if sig == 1:  # buy stock1, sell stock2
            comm_buy = min(
                max(s1_units * _COMMISSION_PER_SHARE, _MIN_COMMISSION),
                half_bp * _MAX_COMMISSION_PCT,
            )
            comm_sell = (
                min(
                    max(s2_units * _COMMISSION_PER_SHARE, _MIN_COMMISSION),
                    half_bp * _MAX_COMMISSION_PCT,
                )
                + _SEC_FEE_RATE * half_bp
                + _FINRA_TAF_PER_SHARE * s2_units
            )
        else:  # buy stock2, sell stock1
            comm_buy = min(
                max(s2_units * _COMMISSION_PER_SHARE, _MIN_COMMISSION),
                half_bp * _MAX_COMMISSION_PCT,
            )
            comm_sell = (
                min(
                    max(s1_units * _COMMISSION_PER_SHARE, _MIN_COMMISSION),
                    half_bp * _MAX_COMMISSION_PCT,
                )
                + _SEC_FEE_RATE * half_bp
                + _FINRA_TAF_PER_SHARE * s1_units
            )
        total_commission = comm_buy + comm_sell

        # --- P&L with 3-pip price adjustment (uses actual exit prices) ---
        if sig == 1:  # long stock1, short stock2
            pnl = (
                (actual_s1_final * _SELL_SLIPPAGE - s1_start * _BUY_SLIPPAGE) * s1_units
                - (actual_s2_final * _BUY_SLIPPAGE - s2_start * _SELL_SLIPPAGE) * s2_units
            )
        else:  # long stock2, short stock1
            pnl = (
                (actual_s2_final * _SELL_SLIPPAGE - s2_start * _BUY_SLIPPAGE) * s2_units
                - (actual_s1_final * _BUY_SLIPPAGE - s1_start * _SELL_SLIPPAGE) * s1_units
            )

        margin += pnl - total_commission
        buying_power = margin / margin_ratio

        trade_log.append(
            {
                **trade,
                "stock1_final_price": actual_s1_final,
                "stock2_final_price": actual_s2_final,
                "stock1_units": s1_units,
                "stock2_units": s2_units,
                "pnl": pnl,
                "commission": total_commission,
                "margin_after": margin,
                "stopped_out": stopped_out,
                "stop_date": stop_date,
            }
        )

    return {
        "margin": margin,
        "trade_count": len(trade_log),
        "trade_log": trade_log,
        "liquidation_date": liquidation_date,
        "n_stops": n_stops,
    }


# ---------------------------------------------------------------------------
# High-Level Pipeline
# ---------------------------------------------------------------------------

def run_pair_pipeline(
    prices_a: pd.Series,
    prices_b: pd.Series,
    window: int,
    zscore_threshold: float,
    margin_init: float,
    margin_ratio: float,
    *,
    pair: tuple[str, str] | None = None,
    fractional: bool = True,
    stop_loss_pct: float = 0.0,
) -> PairPipelineState:
    """Run one full daily backtest and return all intermediate state.

    Pipeline:
        compute_zscore -> compute_signals -> summarize_signals -> calculate_margin

    Use this in notebooks when you want to inspect each intermediate artifact
    (z-score DataFrame, signals, grouped summary, and margin trade log) while
    keeping a single state object for the pair.

    Args:
        prices_a: Adj Close series for stock A.
        prices_b: Adj Close series for stock B.
        window: Rolling z-score window in trading bars (daily here).
        zscore_threshold: Entry threshold for signal generation.
        margin_init: Starting margin/collateral amount.
        margin_ratio: Margin requirement ratio (e.g. 0.25 means 4x buying power).
        pair: Optional label `(ticker_a, ticker_b)` for reporting.
        fractional: True = dollar-value sizing per leg; False = integer shares via math.floor.
        stop_loss_pct: Exit a trade early if unrealized loss exceeds this fraction
            of buying_power (e.g. 0.05 = 5%).  After a stop-loss, the next
            ``window`` trading bars are skipped (cooldown) so contaminated
            z-score history cycles out before re-entry.  0.0 = disabled.

    Returns:
        PairPipelineState containing all intermediate artifacts plus final
        `commission` values are included in `signal_summary` trade columns.
    """
    pair_label: tuple[str | None, str | None] = pair or (
        cast(str | None, prices_a.name),
        cast(str | None, prices_b.name),
    )

    spread_stats = compute_zscore(prices_a, prices_b, window)
    zscore = _column_as_series(spread_stats, "zscore")
    signals = compute_signals(zscore, zscore_threshold)

    # Combine zscore stats + signal into one inspectable DataFrame.
    pair_stats = spread_stats.assign(signal=signals)

    signal_summary_records = summarize_signals(
        _column_as_series(spread_stats, "stock1"),
        _column_as_series(spread_stats, "stock2"),
        signals,
    )
    margin_result = calculate_margin(
        signal_summary_records,
        margin_init,
        margin_ratio,
        fractional=fractional,
        stop_loss_pct=stop_loss_pct,
        pair_stats=pair_stats if stop_loss_pct > 0.0 else None,
        cooldown_days=window if stop_loss_pct > 0.0 else 0,
    )

    # Merge trade-level P&L columns into signal_summary.
    # Active trades (signal != 0) get pnl/commission/units/margin_after filled.
    # Neutral periods (signal == 0) get NaN in those columns.
    signal_summary = pd.DataFrame(signal_summary_records)
    trade_cols = [
        "time_start",
        "stock1_units", "stock2_units",
        "pnl", "commission", "margin_after",
        "stopped_out", "stop_date",
    ]
    if margin_result["trade_log"]:
        trade_df = pd.DataFrame(margin_result["trade_log"])[trade_cols]
        signal_summary = signal_summary.merge(trade_df, on="time_start", how="left")
    else:
        for col in trade_cols[1:]:
            signal_summary[col] = np.nan

    return PairPipelineState(
        pair=pair_label,
        window=window,
        zscore_threshold=zscore_threshold,
        margin_init=margin_init,
        margin_ratio=margin_ratio,
        pair_stats=pair_stats,
        signal_summary=signal_summary,
        margin_final=margin_result["margin"],
        liquidation_date=margin_result["liquidation_date"],
        n_stops=margin_result["n_stops"],
    )



def backtest_pair_intraday(
    prices_a_intraday: pd.Series,
    prices_b_intraday: pd.Series,
    prices_a_daily: pd.Series,
    prices_b_daily: pd.Series,
    window: int,
    zscore_threshold: float,
    margin_init: float,
    margin_ratio: float,
    *,
    pair: tuple[str, str] | None = None,
    fractional: bool = True,
) -> dict:
    """Full intraday backtest: daily rolling stats + intraday z-score.

    Same pipeline as backtest_pair but uses compute_zscore_intraday
    for the z-score step.
    """
    pair_label = pair or (prices_a_intraday.name, prices_b_intraday.name)

    zdf = compute_zscore_intraday(
        prices_a_intraday,
        prices_b_intraday,
        prices_a_daily,
        prices_b_daily,
        window,
    )
    signals = compute_signals(_column_as_series(zdf, "zscore"), zscore_threshold)
    signal_summary = summarize_signals(
        _column_as_series(zdf, "stock1"),
        _column_as_series(zdf, "stock2"),
        signals,
    )
    result = calculate_margin(
        signal_summary, margin_init, margin_ratio, fractional=fractional
    )

    return {
        "pair": pair_label,
        "window": window,
        "zscore_threshold": zscore_threshold,
        "margin": result["margin"],
    }


def grid_search_pair(
    prices_a: pd.Series,
    prices_b: pd.Series,
    windows: Iterable[int],
    zscore_thresholds: Iterable[float],
    margin_init: float,
    margin_ratio: float,
    *,
    pair: tuple[str, str] | None = None,
    fractional: bool = True,
    stop_loss_pct: float = 0.0,
) -> list[dict]:
    """Run run_pair_pipeline across all (window, zscore_threshold) combinations.

    Internal structure:

        windows = [3, 4, ..., 29]        zscore_thresholds = [2.0, 2.1, ..., 4.0]
              │                                    │
              └──────────┬───────────────────────┘
                         │  nested list comprehension
                         ▼
              for w in windows:
                for z in zscore_thresholds:
                    ┌─────────────────────────────────────────────┐
                    │  run_pair_pipeline(prices_a, prices_b,      │
                    │      window=w, zscore_threshold=z, ...)     │
                    │          │                                  │
                    │          ▼                                  │
                    │  PairPipelineState                          │
                    │      .pair / .window / .zscore_threshold    │
                    │      .margin_final                          │
                    │          │                                  │
                    │          ▼                                  │
                    │  { pair, window, zscore_threshold, margin } │
                    └─────────────────────────────────────────────┘
                         │
                         ▼  collect all N×M dicts into a list
              results = [ {...}, {...}, ..., {...} ]   (N × M entries)
                         │
                         ▼  sort by margin descending
              return [ best_result, ..., worst_result ]

    Args:
        windows:           Iterable of window sizes (e.g. range(3, 30)).
        zscore_thresholds: Iterable of thresholds (e.g. np.linspace(2.0, 4.0, 21)).
        stop_loss_pct:     Forwarded to run_pair_pipeline.  0.0 = disabled.
        (remaining args forwarded to run_pair_pipeline)

    Returns:
        List of compact result dicts {pair, window, zscore_threshold, margin, n_stops},
        sorted by margin descending.
    """
    results = [
        {
            "pair": s.pair,
            "window": s.window,
            "zscore_threshold": s.zscore_threshold,
            "margin": s.margin_final,
            "n_stops": s.n_stops,
        }
        for s in (
            run_pair_pipeline(
                prices_a,
                prices_b,
                window=w,
                zscore_threshold=z,
                margin_init=margin_init,
                margin_ratio=margin_ratio,
                pair=pair,
                fractional=fractional,
                stop_loss_pct=stop_loss_pct,
            )
            for w in windows
            for z in zscore_thresholds
        )
    ]
    results.sort(key=lambda r: r["margin"], reverse=True)
    return results
