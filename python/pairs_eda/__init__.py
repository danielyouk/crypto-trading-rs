"""Pairs-trading EDA helpers (universe fetch, optional Exa fallback)."""

from pairs_eda.exa_fallback import (
    ExaRunMode,
    Sp500ExaBackend,
    create_exa_backend,
    default_gemini_backend,
)
from pairs_eda.sp500 import (
    Sp500FetchError,
    WikipediaSp500Error,
    fetch_sp500_constituents_table,
)
from pairs_eda.correlation import (
    compute_pairwise_return_correlations,
    find_candidate_pairs,
)
from pairs_eda.visualization import plot_correlation_histogram
from pairs_eda.yfinance_tools import adj_close_or_close_panel, download_with_retry

__all__ = [
    "ExaRunMode",
    "Sp500ExaBackend",
    "Sp500FetchError",
    "WikipediaSp500Error",
    "adj_close_or_close_panel",
    "compute_pairwise_return_correlations",
    "find_candidate_pairs",
    "download_with_retry",
    "create_exa_backend",
    "default_gemini_backend",
    "fetch_sp500_constituents_table",
    "plot_correlation_histogram",
]
