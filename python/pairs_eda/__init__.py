"""Pairs-trading EDA helpers (universe fetch, optional Exa fallback, backtesting)."""

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
    filter_volatile_tickers,
    find_candidate_pairs,
    find_cointegrated_pairs,
)
from pairs_eda.visualization import plot_correlation_histogram
from pairs_eda.yfinance_tools import adj_close_or_close_panel, download_with_retry
from pairs_eda.backtesting import (
    PairPipelineState,
    backtest_pair_intraday,
    calculate_margin,
    compute_signals,
    compute_zscore,
    compute_zscore_intraday,
    grid_search_pair,
    run_pair_pipeline,
    summarize_signals,
)
from pairs_eda.display import (
    print_margin_summary,
    print_signal_distribution,
    print_signal_groups,
    print_zscore_summary,
)
from pairs_eda.vectorized_backtest import (
    PairsBacktestInput,
    PairsBacktestOutput,
    run_grid_search_optimization,
    run_pairs_backtest_vectorized,
)
from pairs_eda.rolling_phase2 import (
    RebalanceWindow,
    RollingPhase2Config,
    RollingPhase2Input,
    RollingPhase2Output,
    apply_sticky_watchlist,
    build_rolling_timeline,
    compute_robust_pair_scores,
    filter_cointegrated_cached,
    run_phase2_rolling,
)

__all__ = [
    "ExaRunMode",
    "Sp500ExaBackend",
    "Sp500FetchError",
    "WikipediaSp500Error",
    "PairPipelineState",
    "adj_close_or_close_panel",
    "backtest_pair_intraday",
    "calculate_margin",
    "compute_pairwise_return_correlations",
    "compute_signals",
    "compute_zscore",
    "compute_zscore_intraday",
    "filter_volatile_tickers",
    "find_candidate_pairs",
    "find_cointegrated_pairs",
    "download_with_retry",
    "create_exa_backend",
    "default_gemini_backend",
    "fetch_sp500_constituents_table",
    "grid_search_pair",
    "plot_correlation_histogram",
    "print_margin_summary",
    "print_signal_distribution",
    "print_signal_groups",
    "print_zscore_summary",
    "run_pair_pipeline",
    "summarize_signals",
    "PairsBacktestInput",
    "PairsBacktestOutput",
    "RebalanceWindow",
    "RollingPhase2Config",
    "RollingPhase2Input",
    "RollingPhase2Output",
    "apply_sticky_watchlist",
    "build_rolling_timeline",
    "compute_robust_pair_scores",
    "filter_cointegrated_cached",
    "run_grid_search_optimization",
    "run_pairs_backtest_vectorized",
    "run_phase2_rolling",
]
