"""Helpers for yfinance DataFrame shapes (MultiIndex columns, Adj Close vs Close).

Pairs trading on daily data should use Adjusted Close to avoid spurious signals
from stock splits and dividends.  Intraday data (5m, 15m, …) typically only has
Close — that's fine because corporate actions don't happen mid-day.
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd


def _ensure_dataframe(x: pd.DataFrame | pd.Series) -> pd.DataFrame:
    if isinstance(x, pd.DataFrame):
        return x.copy()
    return x.to_frame()


def _adj_close_is_meaningful(
    adj: pd.DataFrame | pd.Series,
    close: pd.DataFrame | pd.Series,
    tolerance: float = 1e-6,
) -> bool:
    """True when Adj Close and Close actually differ (adjustments are present)."""
    a = adj.values.astype(float)
    c = close.values.astype(float)
    mask = np.isfinite(a) & np.isfinite(c) & (c != 0)
    if mask.sum() == 0:
        return False
    return bool(np.any(np.abs(a[mask] / c[mask] - 1.0) > tolerance))


def adj_close_or_close_panel(
    dl: pd.DataFrame,
    *,
    prefer: Optional[str] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Extract a (dates × tickers) price panel from a yfinance download result.

    Selection logic
    ---------------
    1. If ``prefer`` is given, use that column directly.
    2. If both Adj Close and Close exist:
       - Check whether adjustments are actually present (split/dividend effects).
       - Use Adj Close if adjustments exist; warn if they don't.
    3. If only Close exists, use it (typical for intraday data).

    Parameters
    ----------
    dl
        Raw ``yf.download()`` result (possibly MultiIndex columns).
    prefer
        Explicit column name to extract (``"Adj Close"``, ``"Close"``, etc.).
        Skips auto-detection when set.
    verbose
        Print the price-column decision to stdout (useful in notebooks).
    """
    if dl is None:
        raise TypeError("download result is None")
    if dl.empty:
        raise ValueError("yfinance.download returned an empty DataFrame")

    is_multi = isinstance(dl.columns, pd.MultiIndex)

    def _has(col: str) -> bool:
        if is_multi:
            return col in dl.columns.get_level_values(0)
        return col in dl.columns

    def _get(col: str) -> pd.DataFrame:
        return _ensure_dataframe(dl[col])

    def _msg(text: str) -> None:
        if verbose:
            print(f"[pairs_eda] {text}", flush=True)

    if prefer:
        if not _has(prefer):
            available = (
                dl.columns.get_level_values(0).unique().tolist()
                if is_multi
                else list(dl.columns)
            )
            raise KeyError(f"Requested '{prefer}' not found; available: {available!r}")
        _msg(f"Using '{prefer}' (explicitly requested).")
        return _get(prefer)

    has_adj = _has("Adj Close")
    has_close = _has("Close")

    if has_adj and has_close:
        meaningful = _adj_close_is_meaningful(dl["Adj Close"], dl["Close"])
        if meaningful:
            _msg("Using 'Adj Close' — split/dividend adjustments detected.")
            return _get("Adj Close")
        else:
            warnings.warn(
                "[pairs_eda] 'Adj Close' and 'Close' are identical — "
                "no split/dividend adjustments found. "
                "This can happen with auto_adjust=True or short intraday windows. "
                "Using 'Close'.",
                UserWarning,
                stacklevel=2,
            )
            _msg("Using 'Close' — Adj Close identical to Close (no adjustments).")
            return _get("Close")

    if has_adj:
        _msg("Using 'Adj Close' (only adjusted prices available).")
        return _get("Adj Close")

    if has_close:
        _msg("Using 'Close' (no Adj Close column — typical for intraday data).")
        return _get("Close")

    available = (
        dl.columns.get_level_values(0).unique().tolist()
        if is_multi
        else list(dl.columns)
    )
    raise KeyError(f"Neither 'Adj Close' nor 'Close' found; columns: {available!r}")
