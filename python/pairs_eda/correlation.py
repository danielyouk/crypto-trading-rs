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

    Returns
    -------
    ``{(ticker_a, ticker_b): correlation_value, ...}`` ordered by
    correlation descending. The dict preserves insertion order (Python 3.7+).
    """
    sliced = data.loc[start:end] if (start is not None or end is not None) else data

    if sliced.shape[1] < 2:
        raise ValueError(f"Need at least 2 tickers, got {sliced.shape[1]}")

    # pct_change() makes row 0 all-NaN; .corr() handles NaN via pairwise
    # deletion automatically — no need for dropna() which would discard
    # entire rows when ANY ticker has a gap.
    series = sliced.pct_change() if use_returns else sliced
    corr = series.corr()
    n = corr.shape[0]

    # Upper triangle mask (excludes diagonal) → avoids duplicate pairs and self-correlation.
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    rows, cols = np.where(mask)
    tickers = corr.columns.tolist()

    pairs_with_corr = [
        (float(corr.iloc[r, c]), (tickers[r], tickers[c]))
        for r, c in zip(rows, cols)
    ]

    pairs_with_corr.sort(key=lambda x: -x[0])

    filtered = [
        (pair, corr_val)
        for corr_val, pair in pairs_with_corr
        if min_correlation <= corr_val <= max_correlation
    ]

    if top_n is not None:
        filtered = filtered[:top_n]

    return dict(filtered)
