"""Find highly correlated ticker pairs from a price panel."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd


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
