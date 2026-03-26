"""Find highly correlated ticker pairs from a price panel."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd


def filter_volatile_tickers(
    prices: pd.DataFrame,
    *,
    max_move_quantile: float = 0.90,
) -> pd.DataFrame:
    """Remove tickers with extreme single-day crash risk before pair selection.

    What breaks a pair relationship permanently is a sudden *crash* — fraud,
    near-bankruptcy, forced delisting, or a structural business collapse.
    A sudden *surge* (great earnings, M&A target, product launch) is a good
    sign and does NOT disqualify a ticker.

    Therefore this filter measures only the **worst single-day drop** per
    ticker, not the absolute move.

    Filter logic (per ticker):

        daily_returns = Adj Close.pct_change()   (dates × tickers)
                │
                ▼  clip(upper=0).abs()  →  drop-only returns (gains → 0)
                │
                ▼  .max()              →  max_drop  (1 value per ticker,
                │                          = worst single-day loss magnitude)
                ▼  .quantile(max_move_quantile)  →  threshold
                │
                ▼  keep ticker if  max_drop <= threshold
                                   ^^^^^^^^^^^^^^^^^^^^
                                   adapts to the actual universe distribution
                                   (2008 crisis vs. 2010s bull market treated
                                    consistently — always worst X% excluded)

    Example: AAPL had big positive days (iPhone launch, earnings beats) but
    its worst single-day drop is modest relative to stocks like Enron or
    Lehman — so AAPL survives the filter while genuine crash-prone stocks
    are excluded.

    Args:
        prices:            Adj Close panel (dates × tickers).
        max_move_quantile: Tickers above this quantile of worst single-day
                           drop are excluded.  0.90 = drop the worst 10%.

    Returns:
        Filtered DataFrame with the same date index but fewer ticker columns.

    Example:
        >>> sp500_daily_prices = filter_volatile_tickers(
        ...     sp500_daily_prices, max_move_quantile=0.90
        ... )
        filter_volatile_tickers: dropped 48/481 tickers
          max single-day drop cut-off: 28.1%  (p90)
          Dropped: ['AIG', 'MER', ...]
    """
    daily_returns = prices.pct_change().dropna(how="all")
    # clip(upper=0): turn gains into 0 so only losses contribute.
    # abs(): convert negative losses to positive magnitudes for easy comparison.
    max_drop = daily_returns.clip(upper=0).abs().max()

    threshold = float(max_drop.quantile(max_move_quantile))
    keep = max_drop <= threshold

    n_before = prices.shape[1]
    filtered = prices.loc[:, keep]
    n_dropped = n_before - filtered.shape[1]

    if n_dropped > 0:
        dropped = prices.columns[~keep].tolist()
        print(f"filter_volatile_tickers: dropped {n_dropped}/{n_before} tickers")
        print(f"  max single-day drop cut-off: {threshold:.1%}  (p{max_move_quantile:.0%})")
        print(f"  Dropped: {dropped[:10]}{'...' if len(dropped) > 10 else ''}")
    else:
        print("filter_volatile_tickers: no tickers dropped")

    return filtered


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
    min_overlap_years: float = 5.0,
    recent_years: float = 3.0,
) -> dict[tuple[str, str], float]:
    """
    Return correlated ticker pairs filtered by correlation range.

    Applies a dual condition: each pair must pass the correlation band
    over BOTH the full period AND the most recent ``recent_years``.
    This catches pairs whose long-term relationship has degraded.

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
    min_overlap_years
        Minimum years of overlapping data required for a pair.
        Pairs with fewer overlapping trading days are excluded (NaN correlation).
        Converted to trading days as ``int(years * 252)``.
    recent_years
        Window (in years) for the recency check. A pair must also pass
        ``[min_correlation, max_correlation]`` within this recent window.
        Set to 0 to disable the recency check.

    Returns
    -------
    ``{(ticker_a, ticker_b): correlation_value, ...}`` ordered by
    full-period correlation descending. The dict preserves insertion order.
    """
    # ── Input validation ──
    if min_correlation > max_correlation:
        raise ValueError(
            f"min_correlation ({min_correlation}) > max_correlation ({max_correlation})"
        )
    if not (-1.0 <= min_correlation <= 1.0):
        raise ValueError(f"min_correlation ({min_correlation}) outside [-1, 1]")
    if not (-1.0 <= max_correlation <= 1.0):
        raise ValueError(f"max_correlation ({max_correlation}) outside [-1, 1]")
    if min_overlap_years < 0:
        raise ValueError(f"min_overlap_years ({min_overlap_years}) must be >= 0")
    if recent_years < 0:
        raise ValueError(f"recent_years ({recent_years}) must be >= 0")
    if top_n is not None and top_n < 0:
        raise ValueError(f"top_n ({top_n}) must be >= 0 or None")

    sliced = data.loc[start:end] if (start is not None or end is not None) else data
    sliced = sliced.sort_index()

    if sliced.shape[1] < 2:
        raise ValueError(f"Need at least 2 tickers, got {sliced.shape[1]}")

    min_periods = int(min_overlap_years * 252)

    # pct_change() on pandas >=3.0 never forward-fills gaps (fill_method removed).
    # NaN gaps stay NaN in returns; .corr() handles them via pairwise deletion.
    series = sliced.pct_change() if use_returns else sliced
    full_corr = series.corr(min_periods=min_periods)

    if recent_years > 0:
        recent_td = int(recent_years * 252)
        recent_series = series.iloc[-recent_td:]
        recent_min_periods = min(min_periods, len(recent_series))
        recent_corr = recent_series.corr(min_periods=recent_min_periods)
    else:
        recent_corr = None

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

        if recent_corr is not None:
            rc = recent_corr.iloc[r, c]
            if np.isnan(rc) or not (min_correlation <= rc <= max_correlation):
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

    results = Parallel(n_jobs=n_jobs)(
        delayed(_test_one_pair)(a, b, prices, significance)
        for a, b in pairs
    )

    passed = [r for r in results if r is not None]

    import os
    cores_used = (os.cpu_count() or 1) if n_jobs == -1 else n_jobs
    print(
        f"find_cointegrated_pairs: {len(passed)}/{len(pairs)} pairs passed "
        f"(p < {significance})  [{cores_used} cores]"
    )

    return passed
