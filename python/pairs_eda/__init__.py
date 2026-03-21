"""Pairs-trading EDA helpers (universe fetch, optional Exa fallback)."""

from pairs_eda.exa_fallback import ExaRunMode, Sp500ExaBackend, create_exa_backend
from pairs_eda.sp500 import WikipediaSp500Error, fetch_sp500_constituents_table
from pairs_eda.yfinance_tools import adj_close_or_close_panel

__all__ = [
    "ExaRunMode",
    "Sp500ExaBackend",
    "WikipediaSp500Error",
    "adj_close_or_close_panel",
    "create_exa_backend",
    "fetch_sp500_constituents_table",
]
