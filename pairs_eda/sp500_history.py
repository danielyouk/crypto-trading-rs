"""Point-in-time S&P 500 membership utilities.

Uses the hanshof/sp500_constituents dataset (MIT license) which provides
daily-granularity S&P 500 membership from 1996-01-02 to present.

Typical usage::

    from pairs_eda.sp500_history import Sp500History

    hist = Sp500History.from_csv("data/sp500_historical_components.csv")
    members_2008 = hist.members_as_of(pd.Timestamp("2008-09-15"))
    all_tickers = hist.all_unique_tickers()
"""

from __future__ import annotations

import bisect
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional

import pandas as pd

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

_DEFAULT_CSV = Path(__file__).resolve().parent.parent / "data" / "sp500_historical_components.csv"


class Sp500History:
    """Fast point-in-time S&P 500 membership lookup.

    Internally stores a sorted list of (date, frozenset[ticker]) snapshots.
    ``members_as_of(date)`` does a binary search to find the most recent
    snapshot on or before ``date``.
    """

    def __init__(self, snapshots: list[tuple[pd.Timestamp, frozenset[str]]]):
        self._snapshots = sorted(snapshots, key=lambda x: x[0])
        self._dates = [s[0] for s in self._snapshots]
        self._members = [s[1] for s in self._snapshots]

        self._all_tickers: Optional[frozenset[str]] = None

    @classmethod
    def from_csv(cls, path: str | Path = _DEFAULT_CSV) -> "Sp500History":
        """Load from hanshof CSV (``date,tickers`` where tickers is comma-separated)."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"Historical S&P 500 CSV not found at {path}. "
                "Download from https://github.com/hanshof/sp500_constituents"
            )

        snapshots: list[tuple[pd.Timestamp, frozenset[str]]] = []
        with open(path, "r") as f:
            header = f.readline()
            if "date" not in header.lower():
                raise ValueError(f"Unexpected CSV header: {header.strip()}")

            for line in f:
                line = line.strip()
                if not line:
                    continue
                date_str, tickers_str = line.split(",", 1)
                tickers_str = tickers_str.strip('"')
                tickers = frozenset(
                    t.strip().upper().replace(".", "-")
                    for t in tickers_str.split(",")
                    if t.strip()
                )
                snapshots.append((pd.Timestamp(date_str), tickers))

        logger.info(
            "Loaded %d S&P 500 snapshots (%s to %s)",
            len(snapshots),
            snapshots[0][0].date() if snapshots else "?",
            snapshots[-1][0].date() if snapshots else "?",
        )
        return cls(snapshots)

    def members_as_of(self, date: pd.Timestamp) -> frozenset[str]:
        """Return S&P 500 tickers as of ``date`` (forward-filled from nearest prior snapshot)."""
        idx = bisect.bisect_right(self._dates, date) - 1
        if idx < 0:
            return self._members[0]
        return self._members[idx]

    def all_unique_tickers(self) -> frozenset[str]:
        """Union of all tickers that ever appeared in any snapshot."""
        if self._all_tickers is None:
            result: set[str] = set()
            for members in self._members:
                result |= members
            self._all_tickers = frozenset(result)
        return self._all_tickers

    @property
    def first_date(self) -> pd.Timestamp:
        return self._dates[0]

    @property
    def last_date(self) -> pd.Timestamp:
        return self._dates[-1]

    @property
    def n_snapshots(self) -> int:
        return len(self._snapshots)

    def universe_fn(self) -> "Callable[[pd.Timestamp], frozenset[str]]":
        """Return a callable suitable for passing to ``RollingPhase2Input.universe_fn``."""
        return self.members_as_of

    def summary(self) -> dict[str, object]:
        """Quick stats for logging/display."""
        all_t = self.all_unique_tickers()
        first_members = self._members[0] if self._members else frozenset()
        last_members = self._members[-1] if self._members else frozenset()
        return {
            "snapshots": self.n_snapshots,
            "date_range": f"{self.first_date.date()} to {self.last_date.date()}",
            "unique_tickers_ever": len(all_t),
            "first_snapshot_size": len(first_members),
            "last_snapshot_size": len(last_members),
            "tickers_added_over_time": len(all_t - first_members),
            "tickers_removed_over_time": len(first_members - last_members),
        }


_DEFAULT_PARQUET = Path(__file__).resolve().parent.parent / "data" / "sp500_all_prices.parquet"


def download_all_historical_prices(
    history: Sp500History,
    *,
    cache_path: str | Path = _DEFAULT_PARQUET,
    start: str = "1990-01-01",
    force: bool = False,
    verbose: bool = True,
    batch_size: int = 200,
) -> pd.DataFrame:
    """Download prices for ALL tickers that ever appeared in S&P 500, with parquet cache.

    On first call, downloads ~1,100 tickers from Yahoo Finance (takes ~10 min).
    Subsequent calls load from parquet in < 2 seconds.

    Returns:
        DataFrame (dates x tickers), adjusted close prices.
    """
    from pairs_eda.yfinance_tools import download_with_retry

    cache_path = Path(cache_path)

    if cache_path.exists() and not force:
        if verbose:
            print(f"[sp500_history] Loading cached prices from {cache_path}")
        panel = pd.read_parquet(cache_path)
        panel.index = pd.DatetimeIndex(panel.index)
        if verbose:
            print(f"[sp500_history]   {panel.shape[1]} tickers, {panel.shape[0]} days")
        return panel

    all_tickers = sorted(history.all_unique_tickers())
    if verbose:
        print(f"[sp500_history] Downloading prices for {len(all_tickers)} unique historical tickers...")

    import datetime

    end = datetime.datetime.today()

    all_panels: list[pd.DataFrame] = []
    failed_tickers: list[str] = []

    for i in range(0, len(all_tickers), batch_size):
        batch = all_tickers[i : i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(all_tickers) + batch_size - 1) // batch_size

        if verbose:
            print(f"[sp500_history] Batch {batch_num}/{total_batches}: {len(batch)} tickers...")

        try:
            panel = download_with_retry(
                batch, start=start, end=end,
                interval="1d", progress=False, threads=True,
                auto_adjust=False, max_retries=2,
            )
            all_panels.append(panel)
            if verbose:
                print(f"[sp500_history]   Got {panel.shape[1]} tickers")
        except Exception as exc:
            logger.warning("Batch %d failed: %s", batch_num, exc)
            failed_tickers.extend(batch)

    if not all_panels:
        raise RuntimeError("All download batches failed")

    combined = pd.concat(all_panels, axis=1)
    combined = combined.sort_index()

    all_nan_cols = combined.columns[combined.isna().all()]
    if len(all_nan_cols) > 0:
        failed_tickers.extend(all_nan_cols.tolist())
        combined = combined.drop(columns=all_nan_cols)

    if verbose:
        print(f"[sp500_history] Combined: {combined.shape[1]} tickers, {combined.shape[0]} days")
        if failed_tickers:
            print(f"[sp500_history] {len(set(failed_tickers))} tickers with no data (delisted/unavailable)")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(cache_path)
    if verbose:
        size_mb = cache_path.stat().st_size / 1024 / 1024
        print(f"[sp500_history] Saved cache: {cache_path} ({size_mb:.1f} MB)")

    return combined
