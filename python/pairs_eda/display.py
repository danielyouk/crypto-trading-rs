"""Display helpers for notebook backtesting walkthrough output.

These functions extract verbose print-formatting from notebook cells so the
cell body stays readable and high level.
"""

from __future__ import annotations

from typing import Any, cast

import pandas as pd


def print_zscore_summary(pair_stats: pd.DataFrame) -> None:
    """Print the Step 1 z-score table preview and range summary.

    Shows the last 8 rows of ratio/ma/msd/zscore columns (signal excluded so
    the table stays narrow), followed by the full zscore range.

    ASCII preview:
        ======================================================================
        Step 1: Z-Score  —  log(stock_a/stock_b), rolling MA/MSD, zscore
        ======================================================================
        <tail(8) of stock1/stock2/ratio/ma/msd/zscore>
        Z-score range: [min, max]

    Args:
        pair_stats: DataFrame from `run_pair_pipeline` with columns
            stock1/stock2/ratio/ma/msd/zscore/signal.

    Example:
        >>> print_zscore_summary(pipeline_state.pair_stats)
    """
    print("=" * 70)
    print("Step 1: Z-Score  —  log(stock_a/stock_b), rolling MA/MSD, zscore")
    print("=" * 70)
    zscore_cols = ["stock1", "stock2", "ratio", "ma", "msd", "zscore"]
    print(pair_stats[zscore_cols].dropna().tail(8).to_string())
    zscore = cast(pd.Series, pair_stats["zscore"])
    print(f"\nZ-score range: [{zscore.min():.2f}, {zscore.max():.2f}]")


def print_signal_distribution(pair_stats: pd.DataFrame, threshold: float) -> None:
    """Print the Step 2 signal count distribution for -1/0/+1 states.

    ASCII preview:
        ======================================================================
        Step 2: Signals  —  threshold = 2.0
        ======================================================================
        Signal distribution:
          -1 (short A / long B): <count>
           0 (neutral):         <count>
          +1 (long A / short B):<count>

    Args:
        pair_stats: DataFrame from `run_pair_pipeline` with a `signal` column.
        threshold: Entry threshold shown in the section header.

    Example:
        >>> print_signal_distribution(pipeline_state.pair_stats, threshold=2.0)
    """
    print("\n" + "=" * 70)
    print(f"Step 2: Signals  —  threshold = {threshold}")
    print("=" * 70)
    signal = cast(pd.Series, pair_stats["signal"])
    counts = signal.value_counts().sort_index()
    print(f"Signal distribution:\n  -1 (short A / long B): {counts.get(-1, 0)}")
    print(f"   0 (neutral):         {counts.get(0, 0)}")
    print(f"  +1 (long A / short B):{counts.get(1, 0)}")


def print_signal_groups(
    signal_summary: pd.DataFrame,
    max_rows: int = 10,
) -> None:
    """Print the Step 3 grouped-signal date ranges and price transitions.

    ASCII preview:
        ======================================================================
        Step 3: Signal Grouping  —  date ranges per signal period
        ======================================================================
        Total periods: <N>,  Active trades (signal != 0): <M>
        #  sig  start  end  s1_open  s1_close  s2_open  s2_close
        ...

    Args:
        signal_summary: DataFrame from `run_pair_pipeline.signal_summary`.
            One row per contiguous signal period.
        max_rows: Number of rows to print from the top of the table.

    Example:
        >>> print_signal_groups(pipeline_state.signal_summary, max_rows=10)
    """
    print("\n" + "=" * 70)
    print("Step 3: Signal Grouping  —  date ranges per signal period")
    print("=" * 70)
    active_trades = int((signal_summary["signal"] != 0).sum())
    print(
        f"Total periods: {len(signal_summary)},  "
        f"Active trades (signal != 0): {active_trades}\n"
    )
    print(
        f"{'#':>3}  {'sig':>4}  {'start':>12}  {'end':>12}  "
        f"{'s1_open':>9}  {'s1_close':>9}  {'s2_open':>9}  {'s2_close':>9}"
    )
    print("-" * 90)
    for row_index, row in enumerate(signal_summary.head(max_rows).itertuples(index=False)):
        print(
            f"{row_index:3d}  {row.signal:+4.0f}  "
            f"{str(row.time_start)[:10]:>12}  {str(row.time_end)[:10]:>12}  "
            f"{row.stock1_start_price:9.2f}  {row.stock1_final_price:9.2f}  "
            f"{row.stock2_start_price:9.2f}  {row.stock2_final_price:9.2f}"
        )
    if len(signal_summary) > max_rows:
        print(f"  ... ({len(signal_summary) - max_rows} more periods)")


def print_margin_summary(
    signal_summary: pd.DataFrame,
    margin_init: float,
    margin_final: float,
    liquidation_date: str | None = None,
    max_trades: int = 5,
) -> None:
    """Print the Step 4 margin/P&L section and first active-trade rows.

    Reads trade columns (pnl/commission/units/margin_after) directly from
    `signal_summary`. Neutral periods (signal == 0) and post-liquidation periods
    have NaN in those columns and are skipped automatically via `dropna`.

    ASCII preview:
        ======================================================================
        Step 4: Margin Calculation  —  fractional shares, IB commissions
        ======================================================================
        Initial margin:   <x>
        Final margin:     <y>
        Total P&L:        <y - x>
        Trade count:      <n>
        Liquidated:       <date>  ← only shown when account went bankrupt
        First 5 trades:
        ...

    Args:
        signal_summary: DataFrame from `run_pair_pipeline` with pnl/commission columns.
        margin_init: Initial margin/collateral used for the run.
        margin_final: Final margin from `pipeline_state.margin_final`.
        liquidation_date: Date the account was liquidated, from
            `pipeline_state.liquidation_date`. Pass None (default) when solvent.
        max_trades: Number of active-trade rows to print.

    Example:
        >>> print_margin_summary(
        ...     signal_summary, MARGIN_INIT, pipeline_state.margin_final,
        ...     pipeline_state.liquidation_date,
        ... )
    """
    active_trades = signal_summary.dropna(subset=["pnl"])
    print("\n" + "=" * 70)
    print("Step 4: Margin Calculation  —  fractional shares, IB commissions")
    print("=" * 70)
    print(f"Initial margin:  {margin_init}")
    print(f"Final margin:    {margin_final:.2f}")
    print(f"Total P&L:       {margin_final - margin_init:+.2f}")
    print(f"Trade count:     {len(active_trades)}")
    if liquidation_date is not None:
        liq_str = str(liquidation_date)[:10]
        print(f"Liquidated:      {liq_str}  (margin < $2 — simulation stopped)")
    print()
    print("First 5 trades:")
    print(
        f"{'sig':>4}  {'start':>12}  {'end':>12}  {'s1_units':>10}  "
        f"{'s2_units':>10}  {'pnl':>10}  {'comm':>8}  {'margin':>10}"
    )
    print("-" * 95)
    for row in active_trades.head(max_trades).itertuples(index=False):
        print(
            f"{row.signal:+4.0f}  {str(row.time_start)[:10]:>12}  "
            f"{str(row.time_end)[:10]:>12}  {row.stock1_units:10.2f}  "
            f"{row.stock2_units:10.2f}  {row.pnl:+10.2f}  "
            f"{row.commission:8.2f}  {row.margin_after:10.2f}"
        )
