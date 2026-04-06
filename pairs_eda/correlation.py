"""Find highly correlated ticker pairs from a price panel."""

from __future__ import annotations

import logging
import warnings
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator

logger = logging.getLogger(__name__)


class FilterVolatileInput(BaseModel):
    """Validated input for filter_volatile_tickers."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    prices: pd.DataFrame = Field(description="Adj Close panel (dates × tickers)")
    max_move_quantile: float = Field(
        default=0.90,
        gt=0.0,
        lt=1.0,
        description="Drop tickers above this quantile of worst single-day move metric",
    )
    sector_map: Optional[dict[str, str]] = Field(
        default=None,
        description="Ticker → sector mapping. If provided, uses sector-adjusted shocks.",
    )

    @field_validator("sector_map")
    @classmethod
    def strip_empty_sector_labels(cls, v: Optional[dict[str, str]]) -> Optional[dict[str, str]]:
        if v is None:
            return None
        out: dict[str, str] = {}
        for ticker, sector in v.items():
            if not isinstance(ticker, str) or not ticker:
                continue
            if not isinstance(sector, str) or not sector.strip():
                continue
            out[ticker] = sector.strip()
        return out if out else None


class FilterVolatileOutput(BaseModel):
    """Structured output from filter_volatile_tickers."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    filtered_prices: pd.DataFrame
    threshold: float
    max_move_per_ticker: pd.Series
    n_before: int
    n_kept: int
    n_volatile_dropped: int
    n_no_data: int


def _sectors_with_at_least_two_tickers(
    tickers: list[str],
    sector_map: dict[str, str],
) -> set[str]:
    """Sectors that have >= 2 panel tickers mapped (required for meaningful de-meaning)."""
    from collections import Counter

    counts: Counter[str] = Counter()
    for t in tickers:
        s = sector_map.get(t)
        if s is not None:
            counts[s] += 1
    return {s for s, c in counts.items() if c >= 2}


def _per_ticker_max_abs_move(
    daily_returns: pd.DataFrame,
    sector_map: Optional[dict[str, str]],
) -> pd.Series:
    """Worst single-day magnitude per ticker: sector-adjusted where valid, else raw |r|.
    
    Performance notes:
        PERF-001: Uses Numpy broadcasting instead of Pandas stack/pivot.
        This avoids creating O(D·T) long-form DataFrames and is significantly faster.
        NaN values are safely ignored using np.nanmean and np.nanmax.
    """
    tickers = list(daily_returns.columns)
    raw_max = daily_returns.abs().max()

    if not sector_map:
        return raw_max

    multi = _sectors_with_at_least_two_tickers(tickers, sector_map)
    if not multi:
        return raw_max

    # 1. Extract raw numpy array (Dates × Tickers)
    returns_arr = daily_returns.to_numpy()
    
    # 2. Map tickers to sector indices for fast grouping
    # We only care about sectors in `multi`. Others get index -1.
    sector_list = list(multi)
    sector_to_idx = {s: i for i, s in enumerate(sector_list)}
    
    ticker_sector_indices = np.array([
        sector_to_idx.get(sector_map.get(t, ""), -1) 
        for t in tickers
    ])
    
    # 3. Calculate sector means per day (Dates × Sectors)
    n_dates = returns_arr.shape[0]
    n_sectors = len(sector_list)
    sector_means = np.full((n_dates, n_sectors), np.nan)
    
    for sec_idx in range(n_sectors):
        # Boolean mask for columns belonging to this sector
        col_mask = (ticker_sector_indices == sec_idx)
        # Extract only those columns
        sec_data = returns_arr[:, col_mask]
        # Calculate row-wise mean, ignoring NaNs. 
        # Suppress RuntimeWarning for all-NaN slices (e.g., holiday for whole sector)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            sector_means[:, sec_idx] = np.nanmean(sec_data, axis=1)
            
    # 4. Calculate shocks (Broadcasting)
    # Create an array of the same shape as returns_arr to hold the means
    expanded_means = np.full_like(returns_arr, np.nan)
    
    # Only fill in means for tickers that belong to a valid multi-sector
    valid_mask = (ticker_sector_indices != -1)
    valid_indices = ticker_sector_indices[valid_mask]
    
    # Broadcast the sector means to the corresponding ticker columns
    expanded_means[:, valid_mask] = sector_means[:, valid_indices]
    
    # Calculate idiosyncratic shock: |return - sector_mean|
    shocks = np.abs(returns_arr - expanded_means)
    
    # 5. Find max shock per ticker
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        max_shocks = np.nanmax(shocks, axis=0)
        
    # 6. Combine results
    # If a ticker isn't in a valid sector, or if its max_shock is NaN, fall back to raw_max
    combined = raw_max.copy()
    
    for i, t in enumerate(tickers):
        if valid_mask[i] and not np.isnan(max_shocks[i]):
            combined[t] = max_shocks[i]

    return combined


def filter_volatile_tickers_validated(inp: FilterVolatileInput) -> FilterVolatileOutput:
    """Pydantic-wrapped call for filtering volatile tickers."""
    if inp.prices.shape[1] == 0:
        return FilterVolatileOutput(
            filtered_prices=inp.prices.copy(),
            threshold=float("nan"),
            max_move_per_ticker=pd.Series(dtype=float),
            n_before=0,
            n_kept=0,
            n_volatile_dropped=0,
            n_no_data=0,
        )

    panel = inp.prices.sort_index()
    daily_returns = panel.pct_change().dropna(how="all")

    max_move = _per_ticker_max_abs_move(daily_returns, inp.sector_map)

    no_data = max_move.isna()
    n_no_data = int(no_data.sum())
    has_data = ~no_data

    if not has_data.any():
        return FilterVolatileOutput(
            filtered_prices=panel.iloc[:, :0],
            threshold=float("nan"),
            max_move_per_ticker=max_move,
            n_before=panel.shape[1],
            n_kept=0,
            n_volatile_dropped=0,
            n_no_data=n_no_data,
        )

    threshold = float(max_move[has_data].quantile(inp.max_move_quantile))
    too_volatile = has_data & (max_move > threshold)
    keep = has_data & (max_move <= threshold)

    filtered = panel.loc[:, keep]

    return FilterVolatileOutput(
        filtered_prices=filtered,
        threshold=threshold,
        max_move_per_ticker=max_move,
        n_before=panel.shape[1],
        n_kept=int(keep.sum()),
        n_volatile_dropped=int(too_volatile.sum()),
        n_no_data=n_no_data,
    )


def filter_volatile_tickers(
    prices: pd.DataFrame,
    *,
    max_move_quantile: float = 0.90,
    sector_map: Optional[dict[str, str]] = None,
) -> pd.DataFrame:
    """Remove tickers with extreme single-day moves before pair selection.

    ┌─────────────────────────────────────────────────────────────┐
    │  Raw mode:    shock = max(|daily_returns|)                  │
    │  Sector mode: shock = max(|daily_returns - sector_mean|)    │
    └─────────────────────────────────────────────────────────────┘

    Flow:
        prices ──→ pct_change() ──→ shock calculation ──→ quantile threshold ──→ filter

    Both crashes AND surges break pair equilibrium:
    - A crash (fraud, delisting) permanently destroys the spread relationship.
    - A surge (M&A, earnings shock) can equally break the spread by moving
      one leg far from its historical ratio, causing the z-score to escape
      its normal band.

    Filter logic (per ticker):
    - Without ``sector_map``: Uses raw absolute returns.
    - With ``sector_map`` (Sector De-meaning): For tickers in a sector with >= 2
      members, subtracts the sector's daily mean return to isolate idiosyncratic
      shocks. Others fall back to raw absolute returns.

    Args:
        prices:            Adj Close panel (dates × tickers).
        max_move_quantile: Tickers above this quantile of worst single-day
                           absolute move are excluded.  0.90 = drop the worst 10%.
        sector_map:        Optional ticker → sector name mapping.

    Returns:
        Filtered DataFrame with the same date index but fewer ticker columns.

    Performance note:
        PERF-001: Uses Numpy broadcasting instead of Pandas stack/pivot.
        This avoids creating O(D·T) long-form DataFrames and is significantly faster.
        NaN values are safely ignored using np.nanmean and np.nanmax.

    Example:
        >>> filtered = filter_volatile_tickers(prices, max_move_quantile=0.90)
        >>> sec_filtered = filter_volatile_tickers(prices, sector_map={"AAPL": "Tech"})
    """
    inp = FilterVolatileInput(
        prices=prices,
        max_move_quantile=max_move_quantile,
        sector_map=sector_map,
    )
    out = filter_volatile_tickers_validated(inp)
    return out.filtered_prices


def compute_pairwise_return_correlations(
    prices: pd.DataFrame,
    *,
    end: Optional[datetime | pd.Timestamp] = None,
) -> np.ndarray:
    """All unique pairwise returns correlations from a price panel.

    Flow:
        prices[:end]        (dates x N tickers)
         -> pct_change()    (dates x N, first row = NaN, dropped)
         -> .corr()         (N x N symmetric matrix)
         -> upper triangle  (N*(N-1)/2 unique pairs, flat 1-D array)

    What r means (daily returns correlation):
        r ~ 0.10:  Nearly independent daily moves (e.g. tech vs. utilities).
        r ~ 0.30:  Median for S&P 500 pairs over 26 years. Weak co-movement.
        r ~ 0.50:  Top ~5% of S&P 500 pairs. Noticeable daily co-movement.
        r ~ 0.70:  Very strong — stocks move together most days.
        r ~ 0.90+: Near-identical instruments (share classes, tracking ETFs).

    This project uses r in [0.40, 0.85] as a coarse pre-filter:
        0.40: captures meaningful co-movement above the S&P 500 median.
        0.85: excludes structurally identical pairs (e.g. BRK.A/BRK.B).
        The cointegration test (ADF) narrows candidates further.

    Note on methodology:
        Returns correlation is NOT the only (or classic) pair selection
        method. Gatev et al. (2006) used normalized price distance, not
        correlation. Correlation is a fast pre-filter; cointegration is the
        theoretically grounded selection criterion for mean-reversion.

    References:
        Gatev, Goetzmann, Rouwenhorst (2006) "Pairs Trading: Performance
        of a Relative-Value Arbitrage Rule", Review of Financial Studies.
        https://doi.org/10.1093/rfs/hhj020
        (Uses price distance, not correlation. ~11% annualized excess return.)

        Vidyamurthy (2004) "Pairs Trading: Quantitative Methods and
        Analysis", Wiley Finance. (Cointegration-based approach.)

    Args:
        prices: DataFrame shaped (dates x tickers) with Adj Close prices.
        end: Slice data to [:end] before computing. None = use full range.

    Returns:
        1-D numpy array of all unique pairwise return correlations.
        Length = N*(N-1)/2 where N = number of tickers.

    Performance notes:
        PERF-001: .dropna() creates one extra DataFrame copy.
                  Negligible for S&P 500 sizes (~500 tickers).

    Example:
        >>> all_corr = compute_pairwise_return_correlations(data_1d, end=p1_end)
        >>> np.nanmedian(all_corr)   # ~0.31 for S&P 500 over 26 years
        >>> np.nanpercentile(all_corr, 95)  # ~0.50
    """
    # PERF-001: .dropna() creates a copy; avoids NaN row 0 inflating corr() output
    returns = prices.loc[:end].pct_change().dropna()
    correlation_matrix = returns.corr().values

    # Upper triangle only: (A,B) without (B,A) or diagonal (self-corr = 1.0)
    upper_triangle_indices = np.triu_indices_from(correlation_matrix, k=1)
    return correlation_matrix[upper_triangle_indices]


def find_candidate_pairs(
    data: pd.DataFrame,
    *,
    start: Optional[datetime | pd.Timestamp] = None,
    end: Optional[datetime | pd.Timestamp] = None,
    top_n: Optional[int] = None,
    min_correlation: float = 0.40,
    max_correlation: float = 0.85,
    use_returns: bool = True,
    min_overlap_pct: float = 0.90,
) -> dict[tuple[str, str], float]:
    """
    Return correlated ticker pairs filtered by correlation range.

    Parameters
    ----------
    data
        DataFrame shaped (dates × tickers) with adjusted close prices.
    start, end
        Slice the data to ``[start:end]`` before computing correlations.
        If None, uses the full range.
    top_n
        Cap on the number of pairs returned (sorted by correlation descending).
        None means no cap — return all pairs within the correlation range.
    min_correlation
        Lower bound (inclusive). Pairs below this are excluded.
    max_correlation
        Upper bound (inclusive). Pairs above this are excluded.
        Useful for filtering out structurally identical pairs (e.g. GOOG/GOOGL).
    use_returns
        If True (default), compute correlation on daily returns (pct_change)
        instead of raw prices. This removes shared market trends that inflate
        price-level correlations over long periods.
    min_overlap_pct
        Minimum percentage of overlapping data required in the provided window.
        E.g., 0.90 means the pair must have valid data for at least 90% of the days.

    Returns
    -------
    ``{(ticker_a, ticker_b): correlation_value, ...}`` ordered by
    full-period correlation descending. The dict preserves insertion order.
    """
    # ── Input validation (parameters) ──
    if min_correlation > max_correlation:
        raise ValueError(
            f"min_correlation ({min_correlation}) > max_correlation ({max_correlation}): "
            f"the correlation band would be empty."
        )
    if not (-1.0 <= min_correlation <= 1.0):
        raise ValueError(
            f"min_correlation ({min_correlation}) outside [-1, 1] (invalid Pearson bound)."
        )
    if not (-1.0 <= max_correlation <= 1.0):
        raise ValueError(
            f"max_correlation ({max_correlation}) outside [-1, 1] (invalid Pearson bound)."
        )
    if not (0.0 <= min_overlap_pct <= 1.0):
        raise ValueError(
            f"min_overlap_pct ({min_overlap_pct}) must be between 0 and 1."
        )
    if top_n is not None and top_n < 0:
        raise ValueError(
            f"top_n ({top_n}) must be >= 0 or None."
        )

    sliced = data.loc[start:end] if (start is not None or end is not None) else data
    sliced = sliced.sort_index()

    if sliced.shape[1] < 2:
        raise ValueError(
            f"Need at least 2 ticker columns after slicing; got {sliced.shape[1]}."
        )

    n_trading_days = int(len(sliced))
    min_periods = int(n_trading_days * min_overlap_pct)

    # pct_change() on pandas >=3.0 never forward-fills gaps (fill_method removed).
    # NaN gaps stay NaN in returns; .corr() handles them via pairwise deletion.
    series = sliced.pct_change() if use_returns else sliced
    full_corr = series.corr(min_periods=min_periods)

    n = full_corr.shape[0]
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    rows, cols = np.where(mask)
    tickers = full_corr.columns.tolist()

    pairs_with_corr: list[tuple[float, tuple[str, str]]] = []
    for r, c in zip(rows, cols):
        fc = full_corr.iloc[r, c]
        if np.isnan(fc):
            continue
        if not (min_correlation <= fc <= max_correlation):
            continue

        pairs_with_corr.append((float(fc), (tickers[r], tickers[c])))

    pairs_with_corr.sort(key=lambda x: -x[0])

    result = [(pair, corr_val) for corr_val, pair in pairs_with_corr]

    if top_n is not None:
        result = result[:top_n]

    return dict(result)


# ---------------------------------------------------------------------------
# Cointegration filter
# ---------------------------------------------------------------------------

def _test_one_pair(
    a: str,
    b: str,
    prices: pd.DataFrame,
    significance: float,
) -> tuple[str, str] | None:
    """Test one pair for cointegration.  Returns the pair tuple if it passes, else None.

    Designed to be called by joblib workers — must be a module-level function
    (not a lambda/closure) so joblib can pickle it.
    """
    from statsmodels.tsa.stattools import coint

    if a not in prices.columns or b not in prices.columns:
        return None

    pa = prices[a].dropna()
    pb = prices[b].dropna()
    common = pa.index.intersection(pb.index)
    if len(common) < 252:
        return None

    try:
        _, p_value, _ = coint(pa.loc[common], pb.loc[common])
        if p_value < significance:
            return (a, b)
    except Exception:
        pass
    return None


def find_cointegrated_pairs(
    pairs: list[tuple[str, str]],
    prices: pd.DataFrame,
    *,
    significance: float = 0.05,
    n_jobs: int = -1,
) -> list[tuple[str, str]]:
    """Filter candidate pairs by the Engle-Granger cointegration test.

    Pipeline position:

        60K+ pairs
           │  find_candidate_pairs (correlation band)
        ~4K–14K pairs  ← high_correlated_pairs
           │
           ▼  find_cointegrated_pairs  ← YOU ARE HERE
        ~200–500 pairs ← stationary_pairs
           │
           ▼  grid_search_pair / run_grid_search_optimization
        final optimized pairs

    Method (Engle-Granger two-step):
        1. OLS regression:  log(price_A) = α + β·log(price_B) + ε
        2. ADF test on the residual ε
        3. If p-value < significance → the spread is stationary → pair is cointegrated

    Why cointegration, not just correlation?
        Correlation measures co-movement direction.  Cointegration measures whether
        the spread is mean-reverting — which is what pairs trading actually bets on.
        Two stocks can have 0.80 correlation but a non-stationary spread (both trending
        up at different rates).  Cointegration catches this.

    Performance:
        Each pair's coint() test is independent → embarrassingly parallel.
        Uses joblib to distribute across CPU cores.  With 8 cores and 4K pairs,
        ~8× speedup vs. sequential loop.

    Args:
        pairs:        List of (ticker_a, ticker_b) tuples from find_candidate_pairs.
        prices:       Adj Close panel (dates × tickers) — same as used for correlation.
        significance: p-value threshold for the Engle-Granger test (default 0.05).
        n_jobs:       Number of CPU cores for joblib (-1 = all cores, 1 = sequential).

    Returns:
        List of (ticker_a, ticker_b) tuples that pass the cointegration test,
        preserving the input ordering.

    Example:
        >>> stationary_pairs = find_cointegrated_pairs(
        ...     high_correlated_pairs, sp500_daily_prices, significance=0.05
        ... )
        find_cointegrated_pairs: 387/4021 pairs passed (p < 0.05)  [8 cores, 45s]
    """
    if not pairs:
        return []

    from joblib import Parallel, delayed
    from tqdm import tqdm

    results = Parallel(n_jobs=n_jobs)(
        delayed(_test_one_pair)(a, b, prices, significance)
        for a, b in tqdm(pairs, desc="Cointegration test")
    )

    passed = [r for r in results if r is not None]

    import os
    cores_used = (os.cpu_count() or 1) if n_jobs == -1 else n_jobs
    print(
        f"find_cointegrated_pairs: {len(passed)}/{len(pairs)} pairs passed "
        f"(p < {significance})  [{cores_used} cores]"
    )

    return passed
