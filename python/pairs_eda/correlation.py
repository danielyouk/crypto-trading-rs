"""Find highly correlated ticker pairs from a price panel."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd


def find_top_pairs(
    data: pd.DataFrame,
    *,
    start: Optional[datetime | pd.Timestamp] = None,
    end: Optional[datetime | pd.Timestamp] = None,
    top_n: int = 3000,
    min_correlation: float = 0.0,
) -> list[tuple[str, str]]:
    """
    Return the top-N most correlated ticker pairs from a price panel.

    Parameters
    ----------
    data
        DataFrame shaped (dates × tickers) with adjusted close prices.
    start, end
        Slice the data to ``[start:end]`` before computing correlations.
        If None, uses the full range.
    top_n
        Number of pairs to return (sorted by correlation descending).
    min_correlation
        Only include pairs with correlation >= this threshold.

    Returns
    -------
    List of ``(ticker_a, ticker_b)`` tuples, sorted by correlation descending.
    """
    sliced = data.loc[start:end] if (start is not None or end is not None) else data

    if sliced.shape[1] < 2:
        raise ValueError(f"Need at least 2 tickers, got {sliced.shape[1]}")

    corr = sliced.corr()
    n = corr.shape[0]

    # Upper triangle mask (excludes diagonal) → avoids duplicate pairs and self-correlation.
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    rows, cols = np.where(mask)
    tickers = corr.columns.tolist()

    pairs_with_corr = [
        (corr.iloc[r, c], (tickers[r], tickers[c]))
        for r, c in zip(rows, cols)
    ]

    pairs_with_corr.sort(key=lambda x: -x[0])

    result = [
        pair for corr_val, pair in pairs_with_corr
        if corr_val >= min_correlation
    ]

    return result[:top_n]
