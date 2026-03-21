"""
Fetch S&P 500 constituents (Symbol + Date added) from Wikipedia, with optional Exa fallback.

Wikipedia is the default primary source. If it fails and an ``Sp500ExaBackend`` is
provided, we call ``list_sp500_symbols`` with the requested ``ExaRunMode``.

Exa fallback rows have unknown ``Date added``; we set them to ``1900-01-01`` so
downstream filters that exclude very new listings still behave sensibly.
"""

from __future__ import annotations

import io
import logging
from typing import TYPE_CHECKING, Optional

import pandas as pd
import requests

if TYPE_CHECKING:
    from pairs_eda.exa_fallback import ExaRunMode as _ExaRunMode
    from pairs_eda.exa_fallback import Sp500ExaBackend

logger = logging.getLogger(__name__)

DEFAULT_WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

DEFAULT_HEADERS = {
    "User-Agent": "crypto-trading-rs/1.0 (pairs-trading EDA; https://github.com/danielyouk/crypto-trading-rs)",
    "Accept-Language": "en-US,en;q=0.9",
}


class WikipediaSp500Error(RuntimeError):
    """Raised when the Wikipedia table cannot be fetched or parsed."""


def _fetch_wikipedia_html(url: str, *, timeout: float) -> str:
    resp = requests.get(url, headers=DEFAULT_HEADERS, timeout=timeout)
    resp.raise_for_status()
    return resp.text


def _parse_wikipedia_sp500_table(html: str) -> pd.DataFrame:
    tables = pd.read_html(io.StringIO(html))
    if not tables:
        raise WikipediaSp500Error("No HTML tables found on Wikipedia page.")
    first = tables[0]
    need = {"Symbol", "Date added"}
    missing = need - set(first.columns)
    if missing:
        raise WikipediaSp500Error(
            f"Unexpected Wikipedia columns; missing {missing}. Got: {list(first.columns)}"
        )
    out = first[["Symbol", "Date added"]].copy()
    out["Symbol"] = out["Symbol"].astype(str).str.replace(".", "-", regex=False)
    return out


def fetch_sp500_constituents_table(
    *,
    wiki_url: str = DEFAULT_WIKI_URL,
    wiki_timeout: float = 30.0,
    exa_backend: Optional["Sp500ExaBackend"] = None,
    exa_mode: Optional["_ExaRunMode"] = None,
    on_wiki_failure: str = "raise",
) -> pd.DataFrame:
    """
    Return a DataFrame with columns ``Symbol`` and ``Date added`` (datetime64).

    Parameters
    ----------
    wiki_url
        Wikipedia page URL (override for tests or mirrors).
    exa_backend
        If set and ``on_wiki_failure == "exa"``, used when Wikipedia fails.
    exa_mode
        Required when using Exa fallback: ``ExaRunMode.LIVE`` or ``SIMULATION``.
    on_wiki_failure
        ``"raise"`` — propagate Wikipedia errors.
        ``"exa"`` — call ``exa_backend``; raises if backend is None or ``exa_mode`` is None.
    """
    from pairs_eda.exa_fallback import ExaRunMode  # local import for runtime

    try:
        html = _fetch_wikipedia_html(wiki_url, timeout=wiki_timeout)
        return _normalize_dates(_parse_wikipedia_sp500_table(html))
    except Exception as exc:
        logger.warning("Wikipedia S&P 500 fetch/parse failed: %s", exc)
        if on_wiki_failure != "exa":
            raise WikipediaSp500Error(str(exc)) from exc
        if exa_backend is None or exa_mode is None:
            raise WikipediaSp500Error(
                "Wikipedia failed and Exa fallback requested but "
                "`exa_backend` or `exa_mode` is missing."
            ) from exc
        symbols = [str(s).strip().upper().replace(".", "-") for s in exa_backend.list_sp500_symbols(mode=exa_mode)]
        symbols = [s for s in symbols if s]
        if not symbols:
            raise WikipediaSp500Error("Exa fallback returned no symbols.") from exc
        df = pd.DataFrame({"Symbol": symbols, "Date added": pd.Timestamp("1900-01-01")})
        return _normalize_dates(df)


def _normalize_dates(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Date added"] = out["Date added"].fillna("1900-01-01")
    out["Date added"] = pd.to_datetime(out["Date added"], errors="coerce", format="%Y-%m-%d")
    return out
