"""Rolling Phase 2 portfolio simulation with sticky watchlist controls.

This module implements a historically consistent daily simulation framework:

1) Monthly rebalance windows are built with trailing Phase 1 training slices.
2) Each rebalance scores candidate pairs by robustness over train/validation.
3) Sticky watchlist rules reduce churn (retention, persistence, turnover control).
4) Daily portfolio state machine executes slot-constrained trigger entries/exits.
5) Portfolio-level metrics are computed from an equity curve (TWR, Sharpe, MDD, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, cast

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, field_validator, model_validator

from pairs_eda.correlation import filter_volatile_tickers

from pairs_eda.backtesting import compute_zscore
from pairs_eda.correlation import find_candidate_pairs

from statsmodels.tsa.stattools import coint


class RollingPhase2Config(BaseModel):
    """Configuration for rolling Phase 2 simulation.

    Complexity:
        Scoring cost is O(R * P * W * Z * T), where:
            R = rebalance count,
            P = candidate pairs per rebalance,
            W = tested windows,
            Z = tested z-thresholds,
            T = bars in train/validation slice.

    Failure modes:
        - Too short history for `training_months` + `validation_days` yields no schedule.
        - Very large candidate universe can be computationally expensive.
    """

    training_months: int = Field(default=24, ge=12)
    expanding_window: bool = Field(
        default=False,
        description="If True, Phase 1 starts from the earliest available data "
        "instead of a fixed trailing window. The training set grows over time, "
        "accumulating all past regime knowledge. training_months is still used "
        "to determine the first rebalance date.",
    )
    validation_days: int = Field(default=63, ge=21)
    rebalance_frequency: str = Field(default="MS", description="Pandas offset alias, e.g. MS")
    coint_significance: float = Field(
        default=0.05,
        ge=0.01,
        le=0.20,
        description="p-value threshold for Engle-Granger cointegration test. "
        "Pairs with p-value < this threshold are considered cointegrated.",
    )
    coint_retest_margin: float = Field(
        default=0.02,
        ge=0.0,
        le=0.10,
        description="Margin of safety for cointegration cache. "
        "If |cached_p_value - coint_significance| <= margin, the pair is retested. "
        "Set to 1.0 to force retesting all pairs every rebalance (disable cache).",
    )
    coint_cache_max_streak: int = Field(
        default=6,
        ge=1,
        description="Maximum number of consecutive rebalances a cached p-value can be used "
        "before a forced retest. Prevents stale cache entries from living forever.",
    )
    min_correlation: float = Field(default=0.40, ge=-1.0, le=1.0)
    max_correlation: float = Field(default=0.85, ge=-1.0, le=1.0)
    min_overlap_pct: float = Field(default=0.90, ge=0.0, le=1.0)
    top_n_candidates: Optional[int] = Field(default=200, ge=1)
    windows: tuple[int, ...] = Field(default=(20, 30, 40, 60))
    zscore_thresholds: tuple[float, ...] = Field(default=(1.5, 2.0, 2.5, 3.0))
    stress_test_window_step: int = Field(
        default=10, ge=1,
        description="Window interval used to find neighbors during the zero-cost stress test. "
        "Should match the typical spacing in the `windows` tuple."
    )
    stress_test_zscore_step: float = Field(
        default=0.5, gt=0.0,
        description="Z-score interval used to find neighbors during the zero-cost stress test. "
        "Should match the typical spacing in the `zscore_thresholds` tuple."
    )
    stress_test_max_drop_pct: float = Field(
        default=0.5, ge=0.0, le=1.0,
        description="Maximum allowed profit drop compared to the best parameter. "
        "If any neighboring parameter's profit drops by more than this fraction, "
        "the pair is rejected as an overfit peak. E.g., 0.5 means reject if profit halves."
    )
    top_k_for_robustness: int = Field(default=6, ge=2)
    watchlist_size: int = Field(default=20, ge=1)
    max_slots: int = Field(default=7, ge=1)
    max_new_entries_per_day: int = Field(
        default=0, ge=0,
        description="Daily cap for new position entries. "
        "0 disables this cap (unlimited entries per day).",
    )
    leverage: float = Field(default=1.0, ge=1.0)
    max_drop_quantile: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="If > 0, exclude tickers whose worst single-day drop "
        "exceeds this quantile within the training window. "
        "E.g. 0.90 = drop the worst 10%. Applied per-rebalance (no look-ahead).",
    )
    entry_zscore_default: float = Field(default=2.0, gt=0.0)
    exit_zscore: float = Field(default=1.0)
    max_zscore: float = Field(default=5.0, gt=0.0)
    stop_loss_pct: float = Field(default=0.05, ge=0.0)
    min_holding_days: int = Field(
        default=0, ge=0,
        description="Minimum holding days before mean-reversion exit is allowed. "
        "Stop-loss exits are always allowed. 0 disables this.",
    )
    stop_loss_max_slip_pct: float = Field(
        default=0.02, ge=0.0,
        description="Max additional slippage beyond stop_loss_pct when stop fires. "
        "Simulates real-time stop-loss order execution. "
        "Actual loss per trade is capped at (stop_loss_pct + stop_loss_max_slip_pct) "
        "× slot_notional. Set to 0 for ideal stop execution.",
    )
    circuit_breaker_pct: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Portfolio-level circuit breaker. If equity drops more than "
        "this fraction below the peak equity within the current rebalance period, "
        "ALL open positions are closed and no new entries are allowed until "
        "the next rebalance. E.g. 0.15 = halt at -15% drawdown from peak. "
        "Set to 0 to disable.",
    )
    commission_per_leg_bps: float = Field(default=0.5, ge=0.0)
    slippage_per_leg_bps: float = Field(default=1.5, ge=0.0)
    min_entry_score: float = Field(
        default=0.0, ge=0.0,
        description="Minimum final_score required to open a new position. "
        "Pairs below this threshold are skipped even when slots are available. "
        "Set to 0 to disable (accept any watchlist pair). "
        "Typical range: 0.3-0.6 depending on scoring calibration.",
    )
    max_sector_slots: int = Field(
        default=0, ge=0,
        description="Max open positions sharing the same GICS sector. "
        "Requires sector_map in RollingPhase2Input. "
        "Set to 0 to disable (no sector constraint).",
    )
    min_spread_range_pct: float = Field(
        default=0.0, ge=0.0,
        description="Minimum recent spread range as a fraction of mid-price. "
        "Computed as (max - min) of (price_a / price_b) over 60 days, divided "
        "by its mean. If the price ratio barely moves, the pair is too flat to "
        "trade profitably. E.g. 0.05 = require at least 5% ratio swing. "
        "Set to 0 to disable.",
    )
    annual_risk_free_rate: float = Field(default=0.0, ge=0.0)
    start_date: Optional[pd.Timestamp] = None
    end_date: Optional[pd.Timestamp] = None

    model_config = {"arbitrary_types_allowed": True}

    @field_validator("windows")
    @classmethod
    def _validate_windows(cls, values: tuple[int, ...]) -> tuple[int, ...]:
        if len(values) == 0:
            raise ValueError("windows cannot be empty")
        if min(values) <= 1:
            raise ValueError("all windows must be > 1")
        return tuple(sorted(set(values)))

    @field_validator("zscore_thresholds")
    @classmethod
    def _validate_zscores(cls, values: tuple[float, ...]) -> tuple[float, ...]:
        if len(values) == 0:
            raise ValueError("zscore_thresholds cannot be empty")
        if min(values) <= 0.0:
            raise ValueError("all zscore_thresholds must be > 0")
        return tuple(sorted(set(values)))


class RollingPhase2Input(BaseModel):
    """Validated input for `run_phase2_rolling`.

    Args:
        prices: Daily adjusted-close panel shaped (dates x tickers).
        initial_capital: Starting portfolio equity.
        config: Rolling configuration.
        pair_universe: Optional pre-filtered pairs to avoid full pair search each rebalance.
    """

    prices: pd.DataFrame
    initial_capital: float = Field(gt=0.0)
    config: RollingPhase2Config
    pair_universe: Optional[list[tuple[str, str]]] = None
    sector_map: Optional[dict[str, str]] = None

    model_config = {"arbitrary_types_allowed": True}

    @field_validator("prices")
    @classmethod
    def _validate_prices(cls, value: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(value, pd.DataFrame):
            raise TypeError("prices must be a pandas DataFrame")
        if value.empty:
            raise ValueError("prices cannot be empty")
        if value.shape[1] < 2:
            raise ValueError("prices must contain at least two tickers")
        if not isinstance(value.index, pd.DatetimeIndex):
            raise TypeError("prices index must be DatetimeIndex")
        return value

    def model_post_init(self, __context: Any) -> None:
        sorted_prices = cast(pd.DataFrame, self.prices.sort_index())
        if sorted_prices.index.duplicated().any():
            raise ValueError("prices index contains duplicates")
        object.__setattr__(self, "prices", sorted_prices)


class RebalanceWindow(BaseModel):
    """One monthly rebalance window linking Phase 1 and Phase 2 sub-periods."""

    rebalance_date: pd.Timestamp
    phase1_start: pd.Timestamp
    phase1_end: pd.Timestamp
    phase2_start: pd.Timestamp
    phase2_end: pd.Timestamp

    model_config = {"arbitrary_types_allowed": True}


class RollingPhase2Output(BaseModel):
    """Structured output from rolling Phase 2 portfolio simulation."""

    schedule: list[RebalanceWindow]
    rebalance_table: pd.DataFrame
    trades: pd.DataFrame
    daily_equity: pd.Series
    daily_return: pd.Series
    twr_path: pd.Series
    summary: dict[str, float]

    model_config = {"arbitrary_types_allowed": True}


@dataclass
class _OpenPosition:
    """A live pairs trade carried across rebalance boundaries.

    Created when the WFA executor opens a new position and persists in the
    ``open_positions`` list until the Z-score crosses the exit threshold
    (or the position is force-closed at rebalance).

    Fields:
        pair:            (ticker_a, ticker_b) tuple identifying the spread.
        direction:       +1 = long spread (long A, short B),
                         -1 = short spread (short A, long B).
        window:          Look-back window (in trading days) used to compute
                         the Z-score that triggered this entry.
        entry_zscore:    Z-score at which the position was opened.
        exit_zscore:     Z-score threshold for closing (typically near zero,
                         i.e. mean-reversion target).
        entry_date:      Timestamp when the trade was opened.
        score_on_entry:  final_score from ``compute_robust_pair_scores`` at
                         entry time.  Used for position sizing and audit.
        slot_notional:   Capital allocated to this slot at entry
                         (= current equity / max_slots).
        qty_a:           Number of shares (or units) of ticker A.
        qty_b:           Number of shares (or units) of ticker B.
        entry_buy_a:     Fill price of the buy leg for ticker A.
        entry_sell_a:    Fill price of the sell leg for ticker A.
        entry_buy_b:     Fill price of the buy leg for ticker B.
        entry_sell_b:    Fill price of the sell leg for ticker B.

    Note:
        For a long-spread trade (direction=+1), entry_buy_a is the actual
        buy price and entry_sell_b is the actual sell price.  The "opposite"
        fields (entry_sell_a, entry_buy_b) are populated at exit.
    """

    pair: tuple[str, str]
    direction: int
    window: int
    entry_zscore: float
    exit_zscore: float
    entry_date: pd.Timestamp
    score_on_entry: float
    slot_notional: float
    qty_a: float
    qty_b: float
    entry_buy_a: float
    entry_sell_a: float
    entry_buy_b: float
    entry_sell_b: float


def _pair_to_key(pair: tuple[str, str]) -> str:
    return f"{pair[0]}|{pair[1]}"


def _key_to_pair(key: str) -> tuple[str, str]:
    a, b = key.split("|", 1)
    return (a, b)


# ---------------------------------------------------------------------------
# Cointegration Cache — avoid retesting pairs whose result won't change
# ---------------------------------------------------------------------------

@dataclass
class _CointCacheEntry:
    """Cached Engle-Granger p-value for one pair.

    Example:
        {"AAPL|MSFT": _CointCacheEntry(p_value=0.001, passed=True, streak=7)}
    """

    p_value: float
    passed: bool
    streak: int  # Number of consecutive rebalances this cached result has been used
    tested_at: str  # rebalance date string for audit trail


def _test_cointegration(
    pair: tuple[str, str],
    prices: pd.DataFrame,
    significance: float,
) -> _CointCacheEntry | None:
    """Run Engle-Granger cointegration test for one pair.

    Returns None if either ticker is missing or overlap is too short (< 252 bars).
    """
    a, b = pair
    if a not in prices.columns or b not in prices.columns:
        return None
    common = cast(pd.DataFrame, prices[[a, b]].dropna())
    if len(common) < 252:
        return None
    try:
        _, p_value, _ = coint(common[a], common[b])
    except Exception:
        return None
    return _CointCacheEntry(
        p_value=float(p_value),
        passed=float(p_value) < significance,
        streak=1,
        tested_at="",
    )


def filter_cointegrated_cached(
    pairs: list[tuple[str, str]],
    prices: pd.DataFrame,
    cache: dict[str, _CointCacheEntry],
    *,
    significance: float = 0.05,
    margin: float = 0.02,
    max_streak: int = 6,
    rebalance_label: str = "",
) -> tuple[list[tuple[str, str]], dict[str, _CointCacheEntry], dict[str, int]]:
    """Filter pairs by cointegration, reusing cached results when safe.

    Logic per pair:
        1. If NOT in cache → full test → store result (streak=1).
        2. If in cache AND |p - significance| > margin AND streak < max_streak → reuse (skip test, streak+1).
        3. If in cache AND borderline OR streak >= max_streak → retest → store result (streak=1).
        
    Note: Pairs not in the `pairs` input list are implicitly evicted from the returned cache.

    Args:
        pairs:            Candidate pairs (post-correlation filter).
        prices:           Phase 1 training price panel.
        cache:            Mutable dict keyed by "A|B" → _CointCacheEntry.
        significance:     p-value cut-off for Engle-Granger (default 0.05).
        margin:           Margin around significance that forces a retest.
        max_streak:       Max consecutive uses before forced retest.
        rebalance_label:  Tag for audit trail (e.g. "2005-Q1").

    Returns:
        (passed_pairs, updated_cache, stats)
        stats keys: "tested", "cache_hit", "passed"
    """
    passed: list[tuple[str, str]] = []
    new_cache: dict[str, _CointCacheEntry] = {}
    tested = 0
    cache_hit = 0

    for pair in pairs:
        key = _pair_to_key(pair)
        entry = cache.get(key)

        if entry is not None:
            needs_retest = (entry.streak >= max_streak) or (abs(entry.p_value - significance) <= margin)
            if not needs_retest:
                cache_hit += 1
                new_entry = _CointCacheEntry(
                    p_value=entry.p_value,
                    passed=entry.passed,
                    streak=entry.streak + 1,
                    tested_at=entry.tested_at,
                )
                new_cache[key] = new_entry
                if new_entry.passed:
                    passed.append(pair)
                continue

        new_entry = _test_cointegration(pair, prices, significance)
        if new_entry is None:
            continue
        tested += 1
        new_entry.tested_at = rebalance_label
        new_cache[key] = new_entry
        if new_entry.passed:
            passed.append(pair)

    stats = {"tested": tested, "cache_hit": cache_hit, "passed": len(passed)}
    return passed, new_cache, stats


def build_rolling_timeline(inp: RollingPhase2Input) -> list[RebalanceWindow]:
    """Build monthly rolling timeline with trailing Phase 1 windows.

    Flow:
        first_rebalance = start + training_months
            -> each rebalance month:
                 phase1 = trailing [training_months]
                 phase2 = [rebalance_date, next_rebalance - 1 business day]
    """

    cfg = inp.config
    prices = inp.prices
    data_start = prices.index.min()
    data_end = prices.index.max()

    sim_start = pd.Timestamp(cfg.start_date) if cfg.start_date is not None else data_start
    sim_end = pd.Timestamp(cfg.end_date) if cfg.end_date is not None else data_end
    sim_start = max(sim_start, data_start)
    sim_end = min(sim_end, data_end)
    if sim_start >= sim_end:
        return []

    first_rebalance = pd.Timestamp(sim_start) + pd.DateOffset(months=cfg.training_months)
    first_rebalance = pd.Timestamp(first_rebalance).replace(day=1)
    if first_rebalance > sim_end:
        return []

    rebalance_dates = pd.date_range(first_rebalance, sim_end, freq=cfg.rebalance_frequency)
    if len(rebalance_dates) == 0:
        return []

    windows: list[RebalanceWindow] = []
    for i, rebalance_date in enumerate(rebalance_dates):
        p1_end = cast(pd.Timestamp, rebalance_date - pd.Timedelta(days=1))
        if cfg.expanding_window:
            p1_start = cast(pd.Timestamp, sim_start)
        else:
            p1_start = cast(pd.Timestamp, rebalance_date - pd.DateOffset(months=cfg.training_months))
        p2_start = cast(pd.Timestamp, rebalance_date)
        next_rebalance = (
            cast(pd.Timestamp, rebalance_dates[i + 1])
            if i + 1 < len(rebalance_dates)
            else cast(pd.Timestamp, sim_end + pd.Timedelta(days=1))
        )
        p2_end = cast(pd.Timestamp, min(next_rebalance - pd.Timedelta(days=1), sim_end))

        if p1_start < data_start:
            continue
        if p2_start > p2_end:
            continue
        if len(prices.loc[p1_start:p1_end]) < max(cfg.windows) + cfg.validation_days:
            continue
        windows.append(
            RebalanceWindow(
                rebalance_date=rebalance_date,
                phase1_start=p1_start,
                phase1_end=p1_end,
                phase2_start=p2_start,
                phase2_end=p2_end,
            )
        )
    return windows


def _evaluate_pair_surface(
    pair: tuple[str, str],
    prices_train: pd.DataFrame,
    cfg: RollingPhase2Config,
) -> Optional[dict[str, float | tuple[str, str]]]:
    """Compute robustness metrics for one pair across (window, zscore) grid."""

    a, b = pair
    if a not in prices_train.columns or b not in prices_train.columns:
        return None

    pair_df = cast(pd.DataFrame, prices_train[[a, b]].dropna())
    if len(pair_df) < max(cfg.windows) + cfg.validation_days + 5:
        return None

    split_idx = len(pair_df) - cfg.validation_days
    train_df = pair_df.iloc[:split_idx]
    val_df = pair_df.iloc[split_idx:]
    if len(train_df) < max(cfg.windows) + 3:
        return None
    if len(val_df) < max(cfg.windows) + 3:
        return None

    n_train_days = float(len(train_df))
    n_val_days = float(len(val_df))

    rows: list[dict[str, float]] = []
    zscore_cache: dict[int, tuple[pd.Series, pd.Series]] = {}
    for window in cfg.windows:
        train_stats = compute_zscore(train_df[a], train_df[b], window=window)
        val_stats = compute_zscore(val_df[a], val_df[b], window=window)

        train_ret = cast(pd.Series, train_stats["ratio"]).diff().fillna(0.0)
        val_ret = cast(pd.Series, val_stats["ratio"]).diff().fillna(0.0)
        train_z = cast(pd.Series, train_stats["zscore"]).replace([np.inf, -np.inf], np.nan)
        val_z = cast(pd.Series, val_stats["zscore"]).replace([np.inf, -np.inf], np.nan)

        zscore_cache[window] = (train_z, val_z)

        for zthr in cfg.zscore_thresholds:
            train_pos = np.where(
                (train_z > zthr) & (train_z < cfg.max_zscore), -1.0, 
                np.where((train_z < -zthr) & (train_z > -cfg.max_zscore), 1.0, 0.0)
            )
            val_pos = np.where(
                (val_z > zthr) & (val_z < cfg.max_zscore), -1.0, 
                np.where((val_z < -zthr) & (val_z > -cfg.max_zscore), 1.0, 0.0)
            )

            train_pos_s = pd.Series(train_pos, index=train_z.index).ffill().fillna(0.0)
            val_pos_s = pd.Series(val_pos, index=val_z.index).ffill().fillna(0.0)

            if cfg.exit_zscore != 0.0:
                train_pos_s = train_pos_s.where(~((train_pos_s == 1.0) & (train_z >= -cfg.exit_zscore)), 0.0)
                train_pos_s = train_pos_s.where(~((train_pos_s == -1.0) & (train_z <= cfg.exit_zscore)), 0.0)
                val_pos_s = val_pos_s.where(~((val_pos_s == 1.0) & (val_z >= -cfg.exit_zscore)), 0.0)
                val_pos_s = val_pos_s.where(~((val_pos_s == -1.0) & (val_z <= cfg.exit_zscore)), 0.0)

            train_strat = cast(pd.Series, train_pos_s.shift(1).fillna(0.0) * train_ret).sum()
            val_strat = cast(pd.Series, val_pos_s.shift(1).fillna(0.0) * val_ret).sum()
            
            # Count trades (transitions from 0 to +/-1)
            train_entries = ((train_pos_s != 0.0) & (train_pos_s.shift(1).fillna(0.0) == 0.0)).sum()
            val_entries = ((val_pos_s != 0.0) & (val_pos_s.shift(1).fillna(0.0) == 0.0)).sum()
            
            if not np.isfinite(train_strat) or not np.isfinite(val_strat):
                continue
                
            # --- New Consistency Gate (Per-Trade) ---
            if train_entries == 0 or train_strat <= 0.0:
                continue  # Must have made money in the past
                
            train_per_trade = float(train_strat / train_entries)
            
            if val_entries > 0:
                if val_strat <= 0.0:
                    continue  # Traded recently but lost money -> broken pattern
                val_per_trade = float(val_strat / val_entries)
                if val_per_trade > train_per_trade * 3.0:
                    continue  # Abnormally high recent returns -> structural break / luck
            else:
                val_per_trade = 0.0  # No trades, but allowed to pass!

            rows.append(
                {
                    "window": float(window),
                    "zscore": float(zthr),
                    "train_margin": float(train_strat),
                    "validation_margin": float(val_strat),
                    "train_margin_daily": float(train_strat) / n_train_days,
                    "val_margin_daily": float(val_strat) / n_val_days,
                    "train_per_trade": train_per_trade,
                    "val_per_trade": val_per_trade,
                }
            )

    if not rows:
        return None
    
    # Sort by historical robustness (train_margin) since validation might have 0 trades
    surface = pd.DataFrame(rows).sort_values("train_margin", ascending=False)
    top_k = surface.head(min(cfg.top_k_for_robustness, len(surface)))
    if len(top_k) == 0:
        return None

    dist_window = float(top_k["window"].std(ddof=0) / max(top_k["window"].mean(), 1e-9))
    dist_zscore = float(top_k["zscore"].std(ddof=0) / max(top_k["zscore"].mean(), 1e-9))
    train_margin = float(top_k["train_margin"].mean())
    validation_margin = float(top_k["validation_margin"].mean())
    diff_margin = abs(train_margin - validation_margin)

    train_daily = float(top_k["train_margin_daily"].mean())
    val_daily = float(top_k["val_margin_daily"].mean())

    # --- Hard consistency gate (Phase 2a must confirm Phase 1) ---
    # (a) Training period must be profitable on a per-day basis.
    if train_daily <= 0.0:
        return None

    best_window = int(round(float(top_k["window"].median())))
    best_window = min(cfg.windows, key=lambda x: abs(x - best_window))
    best_z = float(top_k["zscore"].median())
    best_z = min(cfg.zscore_thresholds, key=lambda x: abs(x - best_z))

    # --- Zero-Cost Stress Test (Neighborhood Cliff Check) ---
    # We check the immediate neighbors of (best_window, best_z) in the computed grid.
    # If any neighbor is unprofitable, the peak is too fragile (overfit) -> reject.
    # If the profit drops by more than 50% in any neighboring parameter -> reject.
    best_row = surface[(surface["window"] == best_window) & (surface["zscore"] == best_z)]
    if best_row.empty:
        return None
        
    best_profit = float(best_row["train_margin"].iloc[0])
    
    # Define neighbors by value (e.g. +/- 10 days for window, +/- 0.5 for z-score)
    # We only check neighbors that actually exist in our configured grid.
    window_step = cfg.stress_test_window_step
    zscore_step = cfg.stress_test_zscore_step
    
    neighbor_windows = [
        w for w in cfg.windows 
        if abs(w - best_window) <= window_step
    ]
    neighbor_zscores = [
        z for z in cfg.zscore_thresholds 
        if abs(z - best_z) <= zscore_step + 1e-9  # small epsilon for float comparison
    ]

    for nw in neighbor_windows:
        for nz in neighbor_zscores:
            if nw == best_window and nz == best_z:
                continue
            
            neighbor_row = surface[(surface["window"] == nw) & (surface["zscore"] == nz)]
            if neighbor_row.empty:
                # Neighbor was rejected by consistency gates (e.g. lost money in validation)
                return None
                
            neighbor_profit = float(neighbor_row["train_margin"].iloc[0])
            
            # Cliff check: Did profit drop by more than allowed pct just by shifting one parameter step?
            if neighbor_profit < best_profit * (1.0 - cfg.stress_test_max_drop_pct):
                return None

    # (c) Z-score volatility consistency for the chosen window.
    #     The std of z-score reflects spread dynamics; it should be
    #     structurally similar across train and validation periods.
    if best_window in zscore_cache:
        tz, vz = zscore_cache[best_window]
        train_z_std = float(tz.dropna().std())
        val_z_std = float(vz.dropna().std())
        if train_z_std > 1e-6 and val_z_std > 1e-6:
            z_vol_ratio = val_z_std / train_z_std
            if z_vol_ratio > 2.5 or z_vol_ratio < 0.4:
                return None

    diff_ratio = abs(val_daily - train_daily) / (val_daily + 1e-9)
    stability = 1.0 / (1.0 + dist_window + dist_zscore + diff_ratio)
    profit = float(np.tanh(val_daily * 252.0 * 30.0))
    base_score = 0.6 * profit + 0.4 * stability

    return {
        "pair": pair,
        "window": float(best_window),
        "entry_zscore": float(best_z),
        "dist_window": dist_window,
        "dist_zscore": dist_zscore,
        "train_margin": train_margin,
        "validation_margin": validation_margin,
        "diff_margin": diff_margin,
        "base_score": float(base_score),
    }


def compute_robust_pair_scores(
    prices_train: pd.DataFrame,
    cfg: RollingPhase2Config,
    pair_universe: Optional[list[tuple[str, str]]] = None,
    coint_cache: Optional[dict[str, _CointCacheEntry]] = None,
    rebalance_label: str = "",
    sector_map: Optional[dict[str, str]] = None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Score candidate pairs by robustness.

    Called once per rebalance date inside the WFA loop. This is the heart of
    Phase 1: it takes a trailing price panel, runs the 4-step pair selection
    funnel, and returns scored candidates for the watchlist.

    ┌─────────────────────────────────────────────────────────────────────┐
    │  prices_train (dates × tickers)                                     │
    │       │                                                             │
    │       ├─ 1. filter_volatile_tickers  → remove extreme movers        │
    │       ├─ 2. find_candidate_pairs     → correlation band [0.40,0.85] │
    │       ├─ 3. filter_cointegrated      → Engle-Granger ADF (cached)   │
    │       └─ 4. _evaluate_pair_surface   → train/val robustness grid    │
    │                                                                     │
    │  Output: scored DataFrame sorted by final_score descending          │
    └─────────────────────────────────────────────────────────────────────┘

    Args:
        prices_train:
            Adj Close panel shaped (dates × tickers). DatetimeIndex required.
            This is the Phase 1 trailing window (e.g. 36 months of daily prices)
            sliced by the caller as ``prices.loc[phase1_start:phase1_end]``.
            Must contain at least 2 ticker columns and enough rows to cover
            ``max(cfg.windows) + cfg.validation_days``.
        cfg:
            Rolling Phase 2 configuration (correlation band, overlap pct,
            cointegration significance, z-score thresholds, etc.).
        pair_universe:
            Optional pre-filtered list of (ticker_a, ticker_b) tuples.
            If provided, skips Step 1-2 (volatile filter + correlation filter)
            and uses these pairs directly for cointegration + scoring.
            Useful for debugging or restricting the universe manually.
        coint_cache:
            Mutable dict tracking cached Engle-Granger p-values per pair.
            Keyed by "TICKER_A|TICKER_B" → _CointCacheEntry. Persists across
            rebalances. If None, cointegration filtering is skipped entirely.
        rebalance_label:
            Human-readable tag for the current rebalance date (e.g. "2020-01-01").
            Stored in _CointCacheEntry.tested_at for audit trail / debugging.
        sector_map:
            Optional ticker → GICS sector mapping (e.g. {"AAPL": "Information Technology"}).
            Passed to filter_volatile_tickers for sector-adjusted shock calculation.
            If None, raw absolute returns are used for volatility filtering.

    Returns:
        (scored_df, coint_stats)

        scored_df:
            DataFrame with columns: pair_key, ticker_a, ticker_b, window,
            entry_zscore, dist_window, dist_zscore, train_margin,
            validation_margin, diff_margin, base_score, final_score.
            Sorted by final_score descending. Empty DataFrame (with correct
            columns) if no pairs survive the funnel.
        coint_stats:
            {"tested": N, "cache_hit": N, "passed": N} — diagnostics for
            the cointegration cache performance at this rebalance.

    Raises:
        ValueError: If prices_train has fewer than 2 tickers or is not
            indexed by DatetimeIndex.

    Example:
        >>> scored, stats = compute_robust_pair_scores(
        ...     prices.loc["2017-01":"2019-12"],
        ...     cfg=wfa_config,
        ...     coint_cache=cache,
        ...     rebalance_label="2020-01-01",
        ...     sector_map={"AAPL": "Information Technology", "XOM": "Energy"},
        ... )
        >>> scored.head(3)[["pair_key", "final_score"]]
    """
    # ── Input validation ──
    if not isinstance(prices_train.index, pd.DatetimeIndex):
        raise ValueError(
            f"prices_train must have a DatetimeIndex, got {type(prices_train.index).__name__}."
        )
    if prices_train.shape[1] < 2:
        raise ValueError(
            f"prices_train must contain at least 2 tickers, got {prices_train.shape[1]}."
        )
    min_rows_needed = max(cfg.windows) + cfg.validation_days
    if len(prices_train) < min_rows_needed:
        raise ValueError(
            f"prices_train has {len(prices_train)} rows, but needs at least "
            f"{min_rows_needed} (max window {max(cfg.windows)} + validation_days "
            f"{cfg.validation_days})."
        )

    coint_stats: dict[str, int] = {"tested": 0, "cache_hit": 0, "passed": 0}

    train_filtered = prices_train
    if cfg.max_drop_quantile > 0.0:
        train_filtered = filter_volatile_tickers(
            prices_train,
            max_move_quantile=cfg.max_drop_quantile,
            sector_map=sector_map,
        )

    if pair_universe is None:
        candidate_map = find_candidate_pairs(
            train_filtered,
            min_correlation=cfg.min_correlation,
            max_correlation=cfg.max_correlation,
            top_n=cfg.top_n_candidates,
            use_returns=True,
            min_overlap_pct=cfg.min_overlap_pct,
        )
        pairs = list(candidate_map.keys())
    else:
        pairs = pair_universe

    if coint_cache is not None:
        pairs, _, coint_stats = filter_cointegrated_cached(
            pairs,
            prices_train,
            coint_cache,
            significance=cfg.coint_significance,
            margin=cfg.coint_retest_margin,
            max_streak=cfg.coint_cache_max_streak,
            rebalance_label=rebalance_label,
        )

    rows: list[dict[str, Any]] = []
    for pair in pairs:
        evaluated = _evaluate_pair_surface(pair, prices_train, cfg)
        if evaluated is None:
            continue
        key = _pair_to_key(cast(tuple[str, str], evaluated["pair"]))
        final_score = float(evaluated["base_score"])

        rows.append(
            {
                "pair_key": key,
                "ticker_a": pair[0],
                "ticker_b": pair[1],
                "window": int(cast(float, evaluated["window"])),
                "entry_zscore": float(evaluated["entry_zscore"]),
                "dist_window": float(evaluated["dist_window"]),
                "dist_zscore": float(evaluated["dist_zscore"]),
                "train_margin": float(evaluated["train_margin"]),
                "validation_margin": float(evaluated["validation_margin"]),
                "diff_margin": float(evaluated["diff_margin"]),
                "base_score": float(evaluated["base_score"]),
                "final_score": final_score,
            }
        )

    empty_df = pd.DataFrame(
        columns=[
            "pair_key",
            "ticker_a",
            "ticker_b",
            "window",
            "entry_zscore",
            "dist_window",
            "dist_zscore",
            "train_margin",
            "validation_margin",
            "diff_margin",
            "base_score",
            "final_score",
        ]
    )
    if not rows:
        return empty_df, coint_stats
    scored = pd.DataFrame(rows).sort_values("final_score", ascending=False).reset_index(drop=True)
    return scored, coint_stats


def _compute_unrealized_pnl(position: _OpenPosition, price_a: float, price_b: float) -> float:
    """Unrealized P&L at mid prices for stop-loss and daily equity mark-to-market."""

    if position.direction == 1:
        pnl_long = (price_a - position.entry_buy_a) * position.qty_a
        pnl_short = (position.entry_sell_b - price_b) * position.qty_b
        return pnl_long + pnl_short
    pnl_long = (price_b - position.entry_buy_b) * position.qty_b
    pnl_short = (position.entry_sell_a - price_a) * position.qty_a
    return pnl_long + pnl_short


def _commission_from_notional(slot_notional: float, bps: float) -> float:
    """Two-leg commission (entry OR exit) using per-leg bps."""
    return 2.0 * (slot_notional * 0.5) * bps / 10000.0


def _create_feature_cache(
    prices: pd.DataFrame,
    pair: tuple[str, str],
    window: int,
) -> pd.DataFrame:
    """Create date-aligned pair feature table with no-lookahead z-score."""

    a, b = pair
    pair_px = cast(pd.DataFrame, prices[[a, b]].dropna())
    stats = compute_zscore(pair_px[a], pair_px[b], window=window)
    out = pd.DataFrame(
        {
            "price_a": cast(pd.Series, stats["stock1"]),
            "price_b": cast(pd.Series, stats["stock2"]),
            "zscore": cast(pd.Series, stats["zscore"]),
        }
    )
    return out


def run_phase2_rolling(inp: RollingPhase2Input) -> RollingPhase2Output:
    """Run full rolling Phase 2 simulation.

    Pipeline:
        timeline → rebalance scoring (with cointegration cache)
        → daily slot simulation → metrics

    Cointegration caching:
        A persistent cache maps each pair key to its last Engle-Granger p-value.
        At each rebalance, only borderline pairs (p near 0.05) or old entries
        (streak >= max_streak) are retested. Deep-pass and deep-fail pairs reuse
        the cache, cutting ~80% of the cointegration computation across turns.
    """

    cfg = inp.config
    prices = inp.prices
    sector_map = inp.sector_map or {}
    schedule = build_rolling_timeline(inp)
    if len(schedule) == 0:
        raise ValueError("No valid rolling schedule. Adjust date range/training settings.")

    watchlist_by_rebalance: dict[pd.Timestamp, pd.DataFrame] = {}
    rebalance_rows: list[dict[str, Any]] = []
    coint_cache: dict[str, _CointCacheEntry] = {}
    coint_stats_all: list[dict[str, Any]] = []

    for window in schedule:
        phase1 = prices.loc[window.phase1_start:window.phase1_end]
        rebalance_label = str(window.rebalance_date.date())
        scored, coint_stats = compute_robust_pair_scores(
            phase1,
            cfg,
            pair_universe=inp.pair_universe,
            coint_cache=coint_cache,
            rebalance_label=rebalance_label,
            sector_map=sector_map,
        )
        coint_stats_all.append({
            "rebalance": rebalance_label,
            **coint_stats,
        })
        
        # Take top N pairs as the active watchlist
        active_watchlist = scored.head(cfg.watchlist_size).copy()
        watchlist_by_rebalance[window.rebalance_date] = active_watchlist

        if not active_watchlist.empty:
            snap = active_watchlist.copy()
            snap["rebalance_date"] = window.rebalance_date
            rebalance_rows.extend(snap.to_dict("records"))

    sim_start = schedule[0].phase2_start
    sim_end = schedule[-1].phase2_end
    sim_dates = cast(pd.DatetimeIndex, prices.loc[sim_start:sim_end].index)
    if len(sim_dates) == 0:
        raise ValueError("No simulation dates inside built schedule.")

    feature_cache: dict[tuple[str, str, int], pd.DataFrame] = {}
    open_positions: dict[str, _OpenPosition] = {}
    trades: list[dict[str, Any]] = []
    equity_series: list[float] = []
    slot_usage: list[float] = []
    realized_equity = float(inp.initial_capital)
    entry_count = 0
    exit_count = 0
    skipped_low_score = 0

    rebalance_idx = 0
    current_watchlist = watchlist_by_rebalance[schedule[0].rebalance_date]
    current_slot_notional = realized_equity * cfg.leverage / cfg.max_slots
    cb_peak_equity = realized_equity
    cb_cooldown_remaining = 0
    circuit_breaker_count = 0

    for day in sim_dates:
        while (
            rebalance_idx + 1 < len(schedule)
            and day >= schedule[rebalance_idx + 1].phase2_start
        ):
            rebalance_idx += 1
            current_watchlist = watchlist_by_rebalance[schedule[rebalance_idx].rebalance_date]
            current_slot_notional = realized_equity * cfg.leverage / cfg.max_slots

        for key in list(open_positions.keys()):
            pos = open_positions[key]
            cache_key = (pos.pair[0], pos.pair[1], pos.window)
            if cache_key not in feature_cache:
                feature_cache[cache_key] = _create_feature_cache(prices, pos.pair, pos.window)
            feat = feature_cache[cache_key]
            if day not in feat.index:
                continue
            row = cast(pd.Series, feat.loc[day])
            z = float(row["zscore"])
            px_a = float(row["price_a"])
            px_b = float(row["price_b"])
            if not np.isfinite(z) or px_a <= 0.0 or px_b <= 0.0:
                continue

            unrealized = _compute_unrealized_pnl(pos, px_a, px_b)
            holding_days = int((day - pos.entry_date).days)
            can_mean_revert_exit = holding_days >= cfg.min_holding_days
            hit_exit = can_mean_revert_exit and (
                (pos.direction == 1 and z >= -pos.exit_zscore) or
                (pos.direction == -1 and z <= pos.exit_zscore)
            )
            hit_stop = cfg.stop_loss_pct > 0.0 and unrealized <= (-cfg.stop_loss_pct * pos.slot_notional)
            if not (hit_exit or hit_stop):
                continue

            slip = cfg.slippage_per_leg_bps / 10000.0
            if pos.direction == 1:
                exit_sell_a = px_a * (1.0 - slip)
                exit_buy_b = px_b * (1.0 + slip)
                pnl = (exit_sell_a - pos.entry_buy_a) * pos.qty_a + (pos.entry_sell_b - exit_buy_b) * pos.qty_b
            else:
                exit_sell_b = px_b * (1.0 - slip)
                exit_buy_a = px_a * (1.0 + slip)
                pnl = (exit_sell_b - pos.entry_buy_b) * pos.qty_b + (pos.entry_sell_a - exit_buy_a) * pos.qty_a

            if hit_stop:
                max_loss = -(cfg.stop_loss_pct + cfg.stop_loss_max_slip_pct) * pos.slot_notional
                pnl = max(pnl, max_loss)

            exit_commission = _commission_from_notional(pos.slot_notional, cfg.commission_per_leg_bps)
            entry_commission = _commission_from_notional(pos.slot_notional, cfg.commission_per_leg_bps)
            realized_equity += pnl - exit_commission
            exit_count += 1
            trades.append(
                {
                    "pair_key": key,
                    "ticker_a": pos.pair[0],
                    "ticker_b": pos.pair[1],
                    "entry_date": pos.entry_date,
                    "exit_date": day,
                    "direction": pos.direction,
                    "window": pos.window,
                    "entry_zscore": pos.entry_zscore,
                    "exit_zscore_value": z,
                    "holding_days": holding_days,
                    "score_on_entry": pos.score_on_entry,
                    "slot_notional": pos.slot_notional,
                    "pnl": pnl,
                    "commission": entry_commission + exit_commission,
                    "exit_reason": "stop_loss" if hit_stop else "mean_reversion",
                }
            )
            open_positions.pop(key, None)

        # --- Portfolio-level circuit breaker (uses total equity = realized + unrealized) ---
        if cfg.circuit_breaker_pct > 0.0 and cb_cooldown_remaining <= 0:
            cb_unrealized = 0.0
            for _cb_k, _cb_p in open_positions.items():
                _cb_ck = (_cb_p.pair[0], _cb_p.pair[1], _cb_p.window)
                if _cb_ck in feature_cache and day in feature_cache[_cb_ck].index:
                    _cb_r = cast(pd.Series, feature_cache[_cb_ck].loc[day])
                    cb_unrealized += _compute_unrealized_pnl(
                        _cb_p, float(_cb_r["price_a"]), float(_cb_r["price_b"])
                    )
            total_equity = realized_equity + cb_unrealized
            cb_peak_equity = max(cb_peak_equity, total_equity)
            dd_from_peak = (total_equity - cb_peak_equity) / cb_peak_equity
            if dd_from_peak <= -cfg.circuit_breaker_pct:
                circuit_breaker_count += 1
                cb_cooldown_remaining = 5
                for cb_key in list(open_positions.keys()):
                    cb_pos = open_positions[cb_key]
                    cb_cache_key = (cb_pos.pair[0], cb_pos.pair[1], cb_pos.window)
                    if cb_cache_key in feature_cache and day in feature_cache[cb_cache_key].index:
                        cb_row = cast(pd.Series, feature_cache[cb_cache_key].loc[day])
                        cb_px_a, cb_px_b = float(cb_row["price_a"]), float(cb_row["price_b"])
                        cb_slip = cfg.slippage_per_leg_bps / 10000.0
                        if cb_pos.direction == 1:
                            cb_pnl = (cb_px_a * (1.0 - cb_slip) - cb_pos.entry_buy_a) * cb_pos.qty_a + \
                                     (cb_pos.entry_sell_b - cb_px_b * (1.0 + cb_slip)) * cb_pos.qty_b
                        else:
                            cb_pnl = (cb_px_b * (1.0 - cb_slip) - cb_pos.entry_buy_b) * cb_pos.qty_b + \
                                     (cb_pos.entry_sell_a - cb_px_a * (1.0 + cb_slip)) * cb_pos.qty_a
                        cb_comm = _commission_from_notional(cb_pos.slot_notional, cfg.commission_per_leg_bps)
                        realized_equity += cb_pnl - cb_comm
                        exit_count += 1
                        trades.append({
                            "pair_key": cb_key, "ticker_a": cb_pos.pair[0], "ticker_b": cb_pos.pair[1],
                            "entry_date": cb_pos.entry_date, "exit_date": day,
                            "direction": cb_pos.direction, "window": cb_pos.window,
                            "entry_zscore": cb_pos.entry_zscore, "exit_zscore_value": 0.0,
                            "holding_days": int((day - cb_pos.entry_date).days),
                            "score_on_entry": cb_pos.score_on_entry,
                            "slot_notional": cb_pos.slot_notional,
                            "pnl": cb_pnl, "commission": cb_comm * 2,
                            "exit_reason": "circuit_breaker",
                        })
                open_positions.clear()
                cb_peak_equity = realized_equity
        elif cb_cooldown_remaining > 0:
            cb_cooldown_remaining -= 1

        available_slots = cfg.max_slots - len(open_positions)
        daily_new_entries = 0
        if available_slots > 0 and not current_watchlist.empty:
            ranked = current_watchlist.sort_values("final_score", ascending=False)
            for candidate in ranked.to_dict("records"):
                if available_slots <= 0:
                    break
                if cfg.max_new_entries_per_day > 0 and daily_new_entries >= cfg.max_new_entries_per_day:
                    break
                pair_key = str(candidate["pair_key"])
                if pair_key in open_positions:
                    continue
                if cfg.min_entry_score > 0.0 and float(candidate["final_score"]) < cfg.min_entry_score:
                    skipped_low_score += 1
                    break
                pair = _key_to_pair(pair_key)

                # --- Sector diversification constraint ---
                if cfg.max_sector_slots > 0 and sector_map:
                    pair_sectors = {sector_map.get(pair[0], ""), sector_map.get(pair[1], "")} - {""}
                    sector_counts: dict[str, int] = {}
                    for pos in open_positions.values():
                        for t in pos.pair:
                            s = sector_map.get(t, "")
                            if s:
                                sector_counts[s] = sector_counts.get(s, 0) + 1
                    if any(sector_counts.get(s, 0) >= cfg.max_sector_slots * 2 for s in pair_sectors):
                        continue

                window = int(candidate["window"])
                cache_key = (pair[0], pair[1], window)
                if cache_key not in feature_cache:
                    feature_cache[cache_key] = _create_feature_cache(prices, pair, window)
                feat = feature_cache[cache_key]
                if day not in feat.index:
                    continue

                # --- Minimum spread range check (price ratio range) ---
                if cfg.min_spread_range_pct > 0.0:
                    day_loc = feat.index.get_loc(day)
                    lookback = min(60, int(day_loc))
                    if lookback >= 10:
                        sl = slice(int(day_loc) - lookback, int(day_loc) + 1)
                        pa = feat["price_a"].iloc[sl]
                        pb = feat["price_b"].iloc[sl]
                        valid = (pb > 0).all()
                        if valid:
                            ratio = pa / pb
                            ratio_range = float((ratio.max() - ratio.min()) / ratio.mean())
                            if ratio_range < cfg.min_spread_range_pct:
                                continue

                row = cast(pd.Series, feat.loc[day])
                z = float(row["zscore"])
                px_a = float(row["price_a"])
                px_b = float(row["price_b"])
                if not np.isfinite(z) or px_a <= 0.0 or px_b <= 0.0:
                    continue

                entry_z = float(candidate["entry_zscore"])
                direction = 0
                if entry_z <= z < cfg.max_zscore:
                    direction = -1
                elif -cfg.max_zscore < z <= -entry_z:
                    direction = 1
                if direction == 0:
                    continue

                half_notional = 0.5 * current_slot_notional
                slip = cfg.slippage_per_leg_bps / 10000.0
                if direction == 1:
                    entry_buy_a = px_a * (1.0 + slip)
                    entry_sell_b = px_b * (1.0 - slip)
                    qty_a = half_notional / entry_buy_a
                    qty_b = half_notional / entry_sell_b
                    entry_sell_a = px_a * (1.0 - slip)
                    entry_buy_b = px_b * (1.0 + slip)
                else:
                    entry_buy_b = px_b * (1.0 + slip)
                    entry_sell_a = px_a * (1.0 - slip)
                    qty_b = half_notional / entry_buy_b
                    qty_a = half_notional / entry_sell_a
                    entry_buy_a = px_a * (1.0 + slip)
                    entry_sell_b = px_b * (1.0 - slip)

                entry_commission = _commission_from_notional(current_slot_notional, cfg.commission_per_leg_bps)
                realized_equity -= entry_commission
                open_positions[pair_key] = _OpenPosition(
                    pair=pair,
                    direction=direction,
                    window=window,
                    entry_zscore=entry_z,
                    exit_zscore=cfg.exit_zscore,
                    entry_date=day,
                    score_on_entry=float(candidate["final_score"]),
                    slot_notional=current_slot_notional,
                    qty_a=qty_a,
                    qty_b=qty_b,
                    entry_buy_a=entry_buy_a,
                    entry_sell_a=entry_sell_a,
                    entry_buy_b=entry_buy_b,
                    entry_sell_b=entry_sell_b,
                )
                entry_count += 1
                available_slots -= 1
                daily_new_entries += 1

        unrealized_total = 0.0
        for pos in open_positions.values():
            cache_key = (pos.pair[0], pos.pair[1], pos.window)
            feat = feature_cache[cache_key]
            if day not in feat.index:
                continue
            row = cast(pd.Series, feat.loc[day])
            unrealized_total += _compute_unrealized_pnl(pos, float(row["price_a"]), float(row["price_b"]))

        equity_series.append(realized_equity + unrealized_total)
        slot_usage.append(len(open_positions) / cfg.max_slots)

    daily_equity = pd.Series(equity_series, index=sim_dates, name="equity")
    daily_return = cast(pd.Series, daily_equity.pct_change().fillna(0.0))
    twr_path = cast(pd.Series, (1.0 + daily_return).cumprod() - 1.0)

    running_max = cast(pd.Series, daily_equity.cummax())
    drawdown = cast(pd.Series, daily_equity / running_max - 1.0)
    max_drawdown = float(drawdown.min()) if len(drawdown) > 0 else 0.0

    rf_daily = cfg.annual_risk_free_rate / 252.0
    excess = cast(pd.Series, daily_return - rf_daily)
    daily_vol = float(daily_return.std(ddof=0)) if len(daily_return) > 1 else 0.0
    annual_vol = daily_vol * np.sqrt(252.0)
    sharpe = 0.0
    if daily_vol > 1e-12:
        sharpe = float(excess.mean() / daily_vol * np.sqrt(252.0))

    trade_df = pd.DataFrame(trades)
    avg_holding = float(trade_df["holding_days"].mean()) if not trade_df.empty else 0.0
    years = max((sim_dates[-1] - sim_dates[0]).days / 365.25, 1e-9)

    total_coint_tested = sum(s["tested"] for s in coint_stats_all)
    total_coint_cached = sum(s["cache_hit"] for s in coint_stats_all)
    cache_hit_rate = total_coint_cached / max(total_coint_tested + total_coint_cached, 1)

    summary = {
        "cumulative_return": float(daily_equity.iloc[-1] / daily_equity.iloc[0] - 1.0),
        "twr_total": float(twr_path.iloc[-1]),
        "max_drawdown": max_drawdown,
        "volatility_annualized": float(annual_vol),
        "sharpe_ratio": float(sharpe),
        "slot_utilization": float(np.mean(slot_usage)) if slot_usage else 0.0,
        "max_open_slots": float(max((u * cfg.max_slots for u in slot_usage), default=0.0)),
        "entry_count": float(entry_count),
        "exit_count": float(exit_count),
        "turnover_per_year": float(exit_count / years),
        "average_holding_days": avg_holding,
        "performance_net_of_costs": float(daily_equity.iloc[-1] - daily_equity.iloc[0]),
        "coint_tests_total": float(total_coint_tested),
        "coint_cache_hits_total": float(total_coint_cached),
        "coint_cache_hit_rate": float(cache_hit_rate),
        "circuit_breaker_triggers": float(circuit_breaker_count),
        "skipped_low_score": float(skipped_low_score),
    }

    return RollingPhase2Output(
        schedule=schedule,
        rebalance_table=pd.DataFrame(rebalance_rows),
        trades=trade_df,
        daily_equity=daily_equity,
        daily_return=daily_return,
        twr_path=twr_path,
        summary=summary,
    )

