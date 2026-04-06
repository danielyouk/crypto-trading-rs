"""Download missing delisted/bankrupt stock prices from EODHD.

Fills the gap left by Yahoo Finance which doesn't serve data
for companies that have been delisted or gone bankrupt.

Usage::

    from pairs_eda.eodhd_download import download_missing_from_eodhd
    new_prices = download_missing_from_eodhd(missing_tickers, api_key="...")
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)

EODHD_BASE = "https://eodhd.com/api/eod"


def _candidate_symbols(hanshof_ticker: str) -> list[str]:
    """Generate EODHD symbol candidates for a hanshof ticker.

    hanshof uses Yahoo-style tickers (e.g. WAMUQ, BRK-B).
    EODHD uses {SYMBOL}.US format with original pre-bankruptcy names.
    """
    t = hanshof_ticker.strip().upper()
    candidates = [f"{t}.US"]

    if t.endswith("Q") and len(t) > 2:
        candidates.append(f"{t[:-1]}.US")

    if "-" in t:
        candidates.append(f"{t.replace('-', '.')}.US")
        candidates.append(f"{t.replace('-', '')}.US")

    return candidates


def download_missing_from_eodhd(
    missing_tickers: list[str],
    api_key: str,
    *,
    start: str = "1990-01-01",
    end: str = "2026-12-31",
    delay: float = 0.12,
    verbose: bool = True,
) -> tuple[pd.DataFrame, dict[str, str], list[str]]:
    """Download price data for missing tickers from EODHD.

    Returns:
        (prices_df, ticker_map, still_missing)
        - prices_df: DataFrame (dates x hanshof_tickers) with adjusted close
        - ticker_map: {hanshof_ticker: eodhd_symbol} for found tickers
        - still_missing: tickers not found on EODHD either
    """
    found: dict[str, pd.Series] = {}
    ticker_map: dict[str, str] = {}
    still_missing: list[str] = []

    total = len(missing_tickers)

    for i, hanshof_tk in enumerate(missing_tickers):
        candidates = _candidate_symbols(hanshof_tk)
        got_data = False

        for symbol in candidates:
            url = (
                f"{EODHD_BASE}/{symbol}"
                f"?api_token={api_key}&fmt=json"
                f"&from={start}&to={end}&period=d"
            )
            try:
                r = requests.get(url, timeout=30)
                if r.status_code == 200:
                    data = r.json()
                    if isinstance(data, list) and len(data) > 0:
                        df = pd.DataFrame(data)
                        df["date"] = pd.to_datetime(df["date"])
                        df = df.set_index("date").sort_index()

                        if "adjusted_close" in df.columns:
                            series = df["adjusted_close"].astype(float)
                        else:
                            series = df["close"].astype(float)

                        series.name = hanshof_tk
                        found[hanshof_tk] = series
                        ticker_map[hanshof_tk] = symbol

                        if verbose and (i + 1) % 20 == 0:
                            print(
                                f"  [{i+1}/{total}] {hanshof_tk} → {symbol}  "
                                f"({len(data)} days, "
                                f"{data[0]['date']} to {data[-1]['date']})"
                            )
                        got_data = True
                        break
            except Exception as exc:
                logger.debug("Failed %s: %s", symbol, exc)

            time.sleep(delay)

        if not got_data:
            still_missing.append(hanshof_tk)
            if verbose and (i + 1) % 50 == 0:
                print(f"  [{i+1}/{total}] {hanshof_tk} → NOT FOUND")

        time.sleep(delay)

    if verbose:
        print(f"\nEODHD download complete:")
        print(f"  Found:   {len(found)}/{total}")
        print(f"  Missing: {len(still_missing)}/{total}")

    if found:
        prices_df = pd.DataFrame(found)
        prices_df.index = pd.DatetimeIndex(prices_df.index)
        prices_df = prices_df.sort_index()
    else:
        prices_df = pd.DataFrame()

    return prices_df, ticker_map, still_missing
