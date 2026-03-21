"""Helpers for yfinance DataFrame shapes (MultiIndex columns, Adj Close vs Close)."""

from __future__ import annotations

import pandas as pd


def adj_close_or_close_panel(dl: pd.DataFrame) -> pd.DataFrame:
    """
    Return a DataFrame of (dates × tickers) using adjusted prices when available.

    yfinance ``download`` may omit ``Adj Close`` when ``auto_adjust=True`` (only
    ``Close``, already adjusted). For multi-ticker panels, columns are often a
    MultiIndex with price type on level 0 and ticker on level 1.
    """
    if dl is None:
        raise TypeError("download result is None")
    if dl.empty:
        raise ValueError("yfinance.download returned an empty DataFrame")

    if isinstance(dl.columns, pd.MultiIndex):
        level0 = dl.columns.get_level_values(0)
        if "Adj Close" in level0:
            return dl["Adj Close"].copy()
        if "Close" in level0:
            return dl["Close"].copy()
        raise KeyError(
            "MultiIndex panel has neither 'Adj Close' nor 'Close' in level 0; "
            f"got {level0.unique().tolist()!r}"
        )

    if "Adj Close" in dl.columns:
        return dl[["Adj Close"]].copy()
    if "Close" in dl.columns:
        return dl[["Close"]].copy()
    raise KeyError(
        f"Expected 'Adj Close' or 'Close' in columns; got {list(dl.columns)[:20]!r}"
    )
