"""
Gemini Search Grounding for S&P 500 ticker retrieval.

Uses the same pattern as GeminiProvider in google-agents-api-gen
(exa_ai.SimulationEngine.gemini_provider): Gemini + GoogleSearch tool
→ grounding_metadata → extract source URLs → scrape pages → parse tickers.

The LLM cannot list all ~500 tickers in one response, so we:
1. Ask Gemini with Google Search Grounding to find pages listing S&P 500 constituents.
2. Extract source URLs from grounding_metadata.grounding_chunks.
3. Fetch those pages with requests + BeautifulSoup.
4. Parse tickers from HTML tables (same as Wikipedia flow, but on whatever source Google found).

Requirements:
    - GOOGLE_API_KEY in environment (load .env yourself before calling)
    - pip install google-genai
"""

from __future__ import annotations

import io
import os
import re
import logging
from typing import Sequence

import pandas as pd
import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

_SP500_SEARCH_QUERY = (
    "Find a complete list of all current S&P 500 constituent companies "
    "with their ticker symbols. I need the full list of approximately 500 stocks."
)

_TICKER_RE = re.compile(r"\b[A-Z]{1,5}(?:\.[A-Z])?\b")

_FETCH_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
}

_NOISE_WORDS = {
    "THE", "AND", "FOR", "ARE", "NOT", "BUT", "ALL", "CAN", "HAD", "HER",
    "WAS", "ONE", "OUR", "OUT", "HAS", "HIS", "HOW", "ITS", "MAY", "NEW",
    "NOW", "OLD", "SEE", "WAY", "WHO", "DID", "GET", "HIM", "LET", "SAY",
    "SHE", "TOO", "USE", "YES", "EACH", "FROM", "HAVE", "INTO", "LIST",
    "MORE", "ONLY", "SOME", "THAN", "THAT", "THEM", "THEN", "THEY", "THIS",
    "WERE", "WHAT", "WHEN", "WILL", "WITH", "YOUR", "ALSO", "BEEN", "JUST",
    "HERE", "MOST", "MUCH", "VERY", "WELL", "SP", "NYSE", "NASDAQ", "ETF",
    "USD", "RETURN", "COMMA", "INDEX", "INC", "CORP", "LTD", "CO", "CLASS",
    "TABLE", "DATE", "ADDED", "SYMBOL", "NAME", "SECTOR", "COMPANY",
}


def _get_gemini_client():
    """Lazy import + init so google-genai is only needed when actually called."""
    try:
        from google import genai
    except ImportError as e:
        raise ImportError(
            "google-genai is required for Gemini fallback. "
            "Install: pip install 'google-genai>=1.0'"
        ) from e

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GOOGLE_API_KEY not set. Load your .env before calling, e.g.:\n"
            "  from dotenv import load_dotenv; load_dotenv()"
        )
    return genai.Client(api_key=api_key)


def _extract_grounding_urls(response) -> list[str]:
    """Extract source URLs from Gemini grounding metadata (same as GeminiProvider)."""
    urls: list[str] = []
    try:
        if response.candidates:
            candidate = response.candidates[0]
            meta = getattr(candidate, "grounding_metadata", None)
            if meta and hasattr(meta, "grounding_chunks") and meta.grounding_chunks:
                for chunk in meta.grounding_chunks:
                    web = getattr(chunk, "web", None)
                    if web:
                        uri = getattr(web, "uri", "")
                        if uri and uri.startswith("http"):
                            resolved = _resolve_redirect(uri)
                            urls.append(resolved)
    except Exception:
        pass
    return urls


def _resolve_redirect(url: str) -> str:
    """Resolve Google grounding redirect URLs."""
    if "vertexaisearch.cloud.google.com/grounding-api-redirect" not in url:
        return url
    try:
        resp = requests.head(url, allow_redirects=True, timeout=5)
        return resp.url
    except Exception:
        return url


def _fetch_and_parse_tickers_from_url(url: str) -> list[str]:
    """Fetch a web page and try to extract ticker symbols from HTML tables."""
    try:
        resp = requests.get(url, headers=_FETCH_HEADERS, timeout=15)
        resp.raise_for_status()
    except Exception as exc:
        logger.debug("Failed to fetch %s: %s", url, exc)
        return []

    tickers: list[str] = []

    try:
        tables = pd.read_html(io.StringIO(resp.text))
        for table in tables:
            for col in table.columns:
                col_lower = str(col).lower()
                if any(kw in col_lower for kw in ("symbol", "ticker", "stock")):
                    vals = table[col].dropna().astype(str).str.strip().str.upper()
                    vals = vals[vals.str.match(r"^[A-Z]{1,5}(\.[A-Z])?$")]
                    tickers.extend(vals.str.replace(".", "-", regex=False).tolist())
                    break
            if tickers:
                break
    except Exception:
        pass

    if not tickers:
        soup = BeautifulSoup(resp.text, "lxml")
        text = soup.get_text(" ", strip=True)
        raw = _TICKER_RE.findall(text)
        seen: set[str] = set()
        for t in raw:
            t = t.replace(".", "-")
            if t not in seen and t not in _NOISE_WORDS:
                seen.add(t)
                tickers.append(t)

    return tickers


def search_sp500_via_gemini(*, model: str = "gemini-2.5-flash") -> Sequence[str]:
    """
    Use Gemini + Google Search Grounding to retrieve S&P 500 tickers.

    Pipeline (same as exa_answer in live mode):
    1. Gemini call with GoogleSearch tool → grounding_metadata
    2. Extract source URLs from grounding_chunks
    3. Fetch those pages and parse ticker tables
    4. Dedupe and return

    Returns:
        List of uppercase ticker strings (e.g. ['AAPL', 'MSFT', ...]).
    """
    from google.genai import types

    client = _get_gemini_client()

    config = types.GenerateContentConfig(
        tools=[types.Tool(google_search=types.GoogleSearch())],
        system_instruction=(
            "You are a financial data assistant. Find web pages that contain "
            "the complete list of S&P 500 constituent stocks with ticker symbols."
        ),
    )

    logger.info("Gemini search grounding: querying S&P 500 sources …")
    response = client.models.generate_content(
        model=model,
        contents=_SP500_SEARCH_QUERY,
        config=config,
    )

    source_urls = _extract_grounding_urls(response)
    logger.info("Gemini returned %d grounding source URLs.", len(source_urls))

    all_tickers: list[str] = []
    seen: set[str] = set()

    for url in source_urls:
        logger.info("Fetching tickers from: %s", url)
        found = _fetch_and_parse_tickers_from_url(url)
        for t in found:
            if t not in seen:
                seen.add(t)
                all_tickers.append(t)
        if len(all_tickers) >= 400:
            break

    if len(all_tickers) < 400 and (response.text or ""):
        raw = _TICKER_RE.findall(response.text or "")
        for t in raw:
            t = t.replace(".", "-")
            if t not in seen and t not in _NOISE_WORDS:
                seen.add(t)
                all_tickers.append(t)

    if len(all_tickers) < 100:
        logger.warning(
            "Gemini fallback returned only %d tickers (expected ~500). "
            "Results may be incomplete.",
            len(all_tickers),
        )

    logger.info("Gemini fallback total: %d unique tickers.", len(all_tickers))
    return all_tickers
