"""Helpers for yfinance: download with retry, price column selection.

Pairs trading on daily data should use Adjusted Close to avoid spurious signals
from stock splits and dividends.  Intraday data (5m, 15m, …) typically only has
Close — that's fine because corporate actions don't happen mid-day.
"""

from __future__ import annotations

import time
import warnings
from typing import Any, Optional

import os
import tempfile

import numpy as np
import pandas as pd
import yfinance as yf

# yfinance uses a SQLite cache for timezone lookups. In restricted
# environments the default location may be read-only → OperationalError.
_yf_cache = os.path.join(tempfile.gettempdir(), "yf_cache")
os.makedirs(_yf_cache, exist_ok=True)
yf.set_tz_cache_location(_yf_cache)


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


def download_with_retry(
    tickers: list[str],
    *,
    max_retries: int = 2,
    retry_delay: float = 5.0,
    verbose: bool = True,
    **yf_kwargs: Any,
) -> pd.DataFrame:
    """
    Download data via ``yf.download``, retrying only failed tickers.

    After the initial download, identifies tickers whose columns are all-NaN
    (download failures). Re-downloads only those tickers up to ``max_retries``
    times, merging successful results back into the main DataFrame.

    Parameters
    ----------
    tickers
        List of ticker symbols.
    max_retries
        How many times to retry failed tickers (default 2).
    retry_delay
        Seconds to wait between retries (backs off from Yahoo rate limits).
    verbose
        Print retry progress.
    **yf_kwargs
        Passed directly to ``yf.download()`` (start, end, interval, etc.).
    """
    def _msg(text: str) -> None:
        if verbose:
            print(f"[pairs_eda] {text}", flush=True)

    _msg(f"Downloading {len(tickers)} tickers …")
    dl = yf.download(tickers, **yf_kwargs)
    if dl is None or dl.empty:
        raise RuntimeError("yf.download returned None or empty DataFrame")

    panel = adj_close_or_close_panel(dl, verbose=verbose)

    def _all_nan(s: pd.Series | pd.DataFrame) -> bool:
        return bool(s.isna().all())

    for attempt in range(1, max_retries + 1):
        failed = [c for c in panel.columns if _all_nan(panel[c])]
        if not failed:
            break

        _msg(f"Retry {attempt}/{max_retries}: {len(failed)} failed tickers → {failed}")
        time.sleep(retry_delay)

        retry_dl = yf.download(failed, **yf_kwargs)
        if retry_dl is None or retry_dl.empty:
            _msg(f"Retry {attempt}: still no data.")
            continue

        retry_panel = adj_close_or_close_panel(retry_dl, verbose=False)
        for col in retry_panel.columns:
            if col in panel.columns and not _all_nan(retry_panel[col]):
                panel[col] = retry_panel[col]

        recovered = [c for c in failed if not _all_nan(panel[c])]
        still_failed = [c for c in failed if _all_nan(panel[c])]
        if recovered:
            _msg(f"Retry {attempt}: recovered {recovered}")
        if still_failed:
            _msg(f"Retry {attempt}: still failing {still_failed}")

    final_failed = [c for c in panel.columns if _all_nan(panel[c])]
    if final_failed:
        _msg(f"Dropping {len(final_failed)} tickers after all retries: {final_failed}")
        panel = panel.drop(columns=final_failed)

    return panel


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
