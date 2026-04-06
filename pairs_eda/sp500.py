"""
Fetch S&P 500 constituents from any URL with intelligent column detection.

Parsing pipeline:
1. Heuristic: match columns by common names (Symbol/Ticker/Stock, Date added/…).
2. If heuristics fail + GOOGLE_API_KEY is set: ask Gemini to map columns.
3. User can always override with ``symbol_column=`` / ``date_column=``.

Fallback: if the URL itself fails and an ``Sp500ExaBackend`` is provided,
retrieve tickers via Gemini Search Grounding (see ``exa_fallback.py``).
"""

from __future__ import annotations

import io
import json
import logging
import os
import warnings
from typing import TYPE_CHECKING, Optional

import pandas as pd
import requests

if TYPE_CHECKING:
    from pairs_eda.exa_fallback import ExaRunMode as _ExaRunMode
    from pairs_eda.exa_fallback import Sp500ExaBackend

logger = logging.getLogger(__name__)

DEFAULT_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

DEFAULT_HEADERS = {
    "User-Agent": "crypto-trading-rs/1.0 (pairs-trading EDA; https://github.com/danielyouk/crypto-trading-rs)",
    "Accept-Language": "en-US,en;q=0.9",
}

_SYMBOL_HINTS = {"symbol", "ticker", "stock", "code", "stock symbol", "ticker symbol"}
_DATE_HINTS = {"date added", "date", "added", "date_added", "listing date"}


class Sp500FetchError(RuntimeError):
    """Raised when fetch or parse fails and no fallback is available."""


# ---------------------------------------------------------------------------
# HTML fetch
# ---------------------------------------------------------------------------

def _fetch_html(url: str, *, timeout: float | tuple[float, float]) -> str:
    resp = requests.get(url, headers=DEFAULT_HEADERS, timeout=timeout)
    resp.raise_for_status()
    return resp.text


# ---------------------------------------------------------------------------
# Table extraction (all tables from page)
# ---------------------------------------------------------------------------

def _extract_tables(html: str, url: str) -> list[pd.DataFrame]:
    """Return all HTML tables from a page as DataFrames."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "lxml")

    if "wikipedia.org" in url:
        node = soup.find("table", attrs={"id": "constituents"})
        if node is not None:
            return pd.read_html(io.StringIO(str(node)))

    tables = pd.read_html(io.StringIO(html))
    return tables


# ---------------------------------------------------------------------------
# Heuristic column matching
# ---------------------------------------------------------------------------

def _heuristic_symbol_col(cols: list[str]) -> Optional[str]:
    for c in cols:
        if c.lower().strip() in _SYMBOL_HINTS:
            return c
    return None


def _heuristic_date_col(cols: list[str]) -> Optional[str]:
    for c in cols:
        if c.lower().strip() in _DATE_HINTS:
            return c
    return None


def _pick_best_table(
    tables: list[pd.DataFrame],
    symbol_column: Optional[str],
    date_column: Optional[str],
) -> tuple[pd.DataFrame, str, Optional[str]]:
    """
    Return (table, resolved_symbol_col, resolved_date_col).
    Picks the table with the most rows that has a recognisable symbol column.
    """
    candidates: list[tuple[int, pd.DataFrame, str, Optional[str]]] = []

    for tbl in tables:
        cols = [str(c) for c in tbl.columns]

        sym = symbol_column or _heuristic_symbol_col(cols)
        dt = date_column or _heuristic_date_col(cols)

        if sym and sym in cols and len(tbl) > 10:
            candidates.append((len(tbl), tbl, sym, dt if dt and dt in cols else None))

    if not candidates:
        return pd.DataFrame(), "", None

    candidates.sort(key=lambda x: -x[0])
    _, best_tbl, best_sym, best_dt = candidates[0]
    return best_tbl, best_sym, best_dt


# ---------------------------------------------------------------------------
# LLM column mapping (Gemini)
# ---------------------------------------------------------------------------

def _llm_map_columns(
    columns: list[str],
    *,
    needed: dict[str, str],
    verbose: bool,
) -> dict[str, Optional[str]]:
    """
    Ask Gemini to map table columns to our needed fields.

    Parameters
    ----------
    columns
        Actual column names from the HTML table.
    needed
        ``{"symbol": "description", "date_added": "description"}``
    verbose
        Print the mapping decision.

    Returns
    -------
    ``{"symbol": "matched_col_or_None", "date_added": "matched_col_or_None"}``
    """
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        return {k: None for k in needed}

    try:
        from google import genai
        from google.genai import types
    except ImportError:
        return {k: None for k in needed}

    prompt = (
        "You are a data-parsing assistant. Given HTML table column names, "
        "map each requested field to the best matching column name.\n\n"
        f"Available columns: {json.dumps(columns)}\n\n"
        "Requested fields:\n"
    )
    for key, desc in needed.items():
        prompt += f'  - "{key}": {desc}\n'
    prompt += (
        "\nReturn ONLY valid JSON like: "
        '{"symbol": "Ticker", "date_added": "Date added"}\n'
        "If no column matches a field, use null."
    )

    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction="Return only valid JSON. No markdown fences.",
            ),
        )
        text = (response.text or "").strip()
        text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        mapping = json.loads(text)

        for key in needed:
            val = mapping.get(key)
            if val and val not in columns:
                mapping[key] = None

        if verbose:
            _vb(True, f"LLM column mapping: {mapping}")
        return mapping
    except Exception as exc:
        logger.warning("LLM column mapping failed: %s", exc)
        return {k: None for k in needed}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _vb(verbose: bool, msg: str) -> None:
    if verbose:
        print(f"[pairs_eda] {msg}", flush=True)


def fetch_sp500_constituents_table(
    *,
    url: str = DEFAULT_URL,
    timeout: float | tuple[float, float] = (15.0, 60.0),
    symbol_column: Optional[str] = None,
    date_column: Optional[str] = None,
    exa_backend: Optional["Sp500ExaBackend"] = None,
    exa_mode: Optional["_ExaRunMode"] = None,
    on_failure: str = "raise",
    verbose: bool = False,
    # Legacy aliases
    wiki_url: Optional[str] = None,
    wiki_timeout: Optional[float | tuple[float, float]] = None,
    on_wiki_failure: Optional[str] = None,
) -> pd.DataFrame:
    """
    Fetch S&P 500 constituents from any URL.

    Returns a DataFrame with columns ``Symbol`` and ``Date added`` (datetime64).

    Parameters
    ----------
    url
        Web page containing an HTML table with S&P 500 constituents.
        Default: Wikipedia. You can point to any site (e.g. stockanalysis.com).
    timeout
        Request timeout (seconds or (connect, read) tuple).
    symbol_column
        Explicit column name for ticker symbols. If None, auto-detected
        (heuristic first, then LLM if GOOGLE_API_KEY is set).
    date_column
        Explicit column name for listing date. If None, auto-detected.
        Many non-Wikipedia sources don't have this — that's fine, defaults to 1900-01-01.
    exa_backend
        Gemini Search Grounding fallback backend (see ``default_gemini_backend()``).
    exa_mode
        ``ExaRunMode.LIVE`` or ``SIMULATION``.
    on_failure
        ``"raise"`` — propagate errors.
        ``"exa"`` — call ``exa_backend`` when fetch/parse fails.
    verbose
        Print progress and column-mapping decisions.
    """
    # Legacy compat
    if wiki_url is not None:
        url = wiki_url
    if wiki_timeout is not None:
        timeout = wiki_timeout
    if on_wiki_failure is not None:
        on_failure = on_wiki_failure

    try:
        _vb(verbose, f"Fetching {url} …")
        html = _fetch_html(url, timeout=timeout)

        _vb(verbose, "Extracting tables …")
        tables = _extract_tables(html, url)
        if not tables:
            raise Sp500FetchError(f"No HTML tables found at {url}")

        table, sym_col, dt_col = _pick_best_table(tables, symbol_column, date_column)

        if table.empty or not sym_col:
            all_cols = []
            for t in tables:
                all_cols.append([str(c) for c in t.columns])

            _vb(verbose, f"Heuristic failed. Tables found with columns: {all_cols}")

            mapping = _llm_map_columns(
                [str(c) for c in tables[0].columns],
                needed={
                    "symbol": "stock ticker symbol (e.g. AAPL, MSFT)",
                    "date_added": "date the stock was added to the index",
                },
                verbose=verbose,
            )

            sym_col = mapping.get("symbol") or symbol_column
            dt_col = mapping.get("date_added") or date_column

            if sym_col:
                for t in tables:
                    if sym_col in [str(c) for c in t.columns] and len(t) > 10:
                        table = t
                        break

            if table.empty or not sym_col:
                raise Sp500FetchError(
                    f"Could not identify a symbol column in tables at {url}. "
                    f"Columns found: {all_cols}. "
                    f"Try passing symbol_column='YourColumnName' explicitly."
                )

        if sym_col != "Symbol" or (dt_col and dt_col != "Date added"):
            _vb(verbose, f"Column mapping: symbol='{sym_col}', date='{dt_col or '(none)'}'")
            warnings.warn(
                f"[pairs_eda] Using column mapping: symbol='{sym_col}', "
                f"date='{dt_col or '(none)'}'. "
                f"If wrong, pass symbol_column= / date_column= explicitly.",
                UserWarning,
                stacklevel=2,
            )

        out = pd.DataFrame()
        out["Symbol"] = table[sym_col].astype(str).str.strip().str.upper().str.replace(".", "-", regex=False)

        if dt_col and dt_col in table.columns:
            out["Date added"] = table[dt_col]
        else:
            out["Date added"] = pd.NaT

        out = _normalize_dates(out)
        _vb(verbose, f"Done ({len(out)} rows from {url}).")
        return out

    except Exception as exc:
        logger.warning("Fetch/parse failed for %s: %s", url, exc)
        _vb(verbose, f"Fetch/parse failed: {exc}")

        if on_failure != "exa":
            raise Sp500FetchError(str(exc)) from exc
        if exa_backend is None or exa_mode is None:
            raise Sp500FetchError(
                f"Fetch failed for {url} and Exa fallback requested but "
                "`exa_backend` or `exa_mode` is missing."
            ) from exc

        _vb(verbose, "Falling back to Gemini Search Grounding …")
        symbols = [
            str(s).strip().upper().replace(".", "-")
            for s in exa_backend.list_sp500_symbols(mode=exa_mode)
        ]
        symbols = [s for s in symbols if s]
        if not symbols:
            raise Sp500FetchError("Exa fallback returned no symbols.") from exc
        _vb(verbose, f"Gemini fallback: {len(symbols)} tickers.")
        df = pd.DataFrame({"Symbol": symbols, "Date added": pd.NaT})
        return _normalize_dates(df)


def _normalize_dates(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Date added"] = pd.to_datetime(out["Date added"], errors="coerce", format="mixed")
    return out


def fetch_sp500_sector_map(
    *,
    url: str = DEFAULT_URL,
    timeout: float | tuple[float, float] = (15.0, 60.0),
    verbose: bool = False,
) -> dict[str, str]:
    """Return a {ticker: GICS_sector} mapping from the S&P 500 Wikipedia table.

    Falls back to an empty dict if the sector column is not found.
    """
    _SECTOR_HINTS = {"gics sector", "sector", "gics_sector"}

    try:
        html = _fetch_html(url, timeout=timeout)
        tables = _extract_tables(html, url)
        if not tables:
            return {}

        table, sym_col, _ = _pick_best_table(tables, None, None)
        if table.empty or not sym_col:
            return {}

        cols_lower = {str(c).lower().strip(): str(c) for c in table.columns}
        sector_col = None
        for hint in _SECTOR_HINTS:
            if hint in cols_lower:
                sector_col = cols_lower[hint]
                break

        if sector_col is None:
            _vb(verbose, "No GICS Sector column found in table.")
            return {}

        mapping: dict[str, str] = {}
        for _, row in table.iterrows():
            ticker = str(row[sym_col]).strip().upper().replace(".", "-")
            sector = str(row[sector_col]).strip()
            if ticker and sector and sector != "nan":
                mapping[ticker] = sector

        _vb(verbose, f"Sector map: {len(mapping)} tickers across {len(set(mapping.values()))} sectors.")
        return mapping

    except Exception as exc:
        logger.warning("Failed to fetch sector map: %s", exc)
        return {}


# Backward compat alias
WikipediaSp500Error = Sp500FetchError
