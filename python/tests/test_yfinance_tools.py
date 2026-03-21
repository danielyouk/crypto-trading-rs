"""Tests for adj_close_or_close_panel and download_with_retry."""

import warnings
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from pairs_eda.yfinance_tools import adj_close_or_close_panel, download_with_retry

# ---------------------------------------------------------------------------
# Helpers to build fake yfinance-style DataFrames
# ---------------------------------------------------------------------------

DATES = pd.bdate_range("2024-01-02", periods=60)
TICKERS = ["AAPL", "MSFT", "GOOG"]


def _random_prices(n_dates: int, n_tickers: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = rng.uniform(100, 400, size=n_tickers)
    returns = rng.normal(0, 0.01, size=(n_dates, n_tickers))
    return base * np.cumprod(1 + returns, axis=0)


def make_daily_panel(*, with_split: bool = True) -> pd.DataFrame:
    """
    Simulate yf.download(..., auto_adjust=False) for daily data.
    MultiIndex columns: (price_type, ticker).

    If with_split=True, AAPL has a 4:1 split at row 30 — making
    Adj Close differ from Close in the pre-split period.
    """
    prices = _random_prices(len(DATES), len(TICKERS))
    close = pd.DataFrame(prices, index=DATES, columns=TICKERS)
    adj_close = close.copy()

    if with_split:
        adj_close.iloc[:30, 0] = close.iloc[:30, 0] / 4.0

    data = {
        ("Adj Close", t): adj_close[t] for t in TICKERS
    } | {
        ("Close", t): close[t] for t in TICKERS
    } | {
        ("Volume", t): np.random.default_rng(7).integers(1_000_000, 50_000_000, size=len(DATES))
        for t in TICKERS
    }
    df = pd.DataFrame(data, index=DATES)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df


def make_intraday_panel() -> pd.DataFrame:
    """Simulate yf.download(..., interval='5m') — only Close, no Adj Close."""
    idx = pd.date_range("2024-06-01 09:30", periods=78, freq="5min")
    prices = _random_prices(78, len(TICKERS), seed=99)
    close = pd.DataFrame(prices, index=idx, columns=TICKERS)

    data = {("Close", t): close[t] for t in TICKERS} | {
        ("Volume", t): np.random.default_rng(3).integers(10_000, 500_000, size=78)
        for t in TICKERS
    }
    df = pd.DataFrame(data, index=idx)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAdjCloseWithSplit:
    """Daily data where Adj Close differs from Close (split scenario)."""

    def test_selects_adj_close(self, capsys: pytest.CaptureFixture[str]) -> None:
        dl = make_daily_panel(with_split=True)
        result = adj_close_or_close_panel(dl, verbose=True)

        assert result.shape == (60, 3)
        assert list(result.columns) == TICKERS
        out = capsys.readouterr().out
        assert "Adj Close" in out
        assert "adjustments detected" in out

    def test_adj_close_differs_from_close(self) -> None:
        dl = make_daily_panel(with_split=True)
        result = adj_close_or_close_panel(dl, verbose=False)
        close_panel = pd.DataFrame(
            {t: dl[("Close", t)] for t in TICKERS}, index=DATES
        )
        assert not np.allclose(
            np.asarray(result["AAPL"]), np.asarray(close_panel["AAPL"])
        ), "AAPL should differ due to 4:1 split adjustment"


class TestAdjCloseIdenticalToClose:
    """Daily data where Adj Close == Close (no corporate actions)."""

    def test_warns_and_uses_close(self) -> None:
        dl = make_daily_panel(with_split=False)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = adj_close_or_close_panel(dl, verbose=False)
            user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
            assert len(user_warnings) >= 1
            assert "identical" in str(user_warnings[0].message).lower()

        assert result.shape == (60, 3)


class TestIntradayNoAdjClose:
    """Intraday data — only Close available (no Adj Close column)."""

    def test_selects_close(self, capsys: pytest.CaptureFixture[str]) -> None:
        dl = make_intraday_panel()
        result = adj_close_or_close_panel(dl, verbose=True)

        assert result.shape == (78, 3)
        out = capsys.readouterr().out
        assert "Close" in out
        assert "intraday" in out.lower() or "no Adj Close" in out


class TestExplicitPrefer:
    """User explicitly requests a specific column via prefer=."""

    def test_prefer_close_skips_adj(self, capsys: pytest.CaptureFixture[str]) -> None:
        dl = make_daily_panel(with_split=True)
        result = adj_close_or_close_panel(dl, prefer="Close", verbose=True)

        close_panel = pd.DataFrame(
            {t: dl[("Close", t)] for t in TICKERS}, index=DATES
        )
        pd.testing.assert_frame_equal(result, close_panel)
        out = capsys.readouterr().out
        assert "explicitly requested" in out

    def test_prefer_missing_column_raises(self) -> None:
        dl = make_daily_panel(with_split=True)
        with pytest.raises(KeyError, match="Open"):
            adj_close_or_close_panel(dl, prefer="Open")


class TestFlatColumns:
    """Single-ticker download: flat columns (not MultiIndex)."""

    def test_single_ticker_adj_close(self) -> None:
        dates = pd.bdate_range("2024-01-02", periods=30)
        rng = np.random.default_rng(11)
        close = rng.uniform(150, 160, size=30)
        adj = close * 0.98  # simulate a small dividend adjustment
        dl = pd.DataFrame(
            {"Adj Close": adj, "Close": close, "Volume": 1_000_000},
            index=dates,
        )
        result = adj_close_or_close_panel(dl, verbose=False)
        np.testing.assert_array_almost_equal(np.asarray(result["Adj Close"]), adj)


class TestEdgeCases:
    def test_none_raises_type_error(self) -> None:
        with pytest.raises(TypeError):
            adj_close_or_close_panel(None)  # type: ignore[arg-type]

    def test_empty_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            adj_close_or_close_panel(pd.DataFrame())

    def test_no_price_columns_raises_key_error(self) -> None:
        dl = pd.DataFrame({"Volume": [1, 2, 3]})
        with pytest.raises(KeyError, match="(?i)neither.*nor"):
            adj_close_or_close_panel(dl, verbose=False)


# ---------------------------------------------------------------------------
# Tests for download_with_retry
# ---------------------------------------------------------------------------

def _make_multi_dl(tickers: list[str], dates: pd.DatetimeIndex, *, nan_tickers: list[str] | None = None) -> pd.DataFrame:
    """Build a fake yf.download result (MultiIndex) with optional all-NaN tickers."""
    rng = np.random.default_rng(42)
    data: dict[tuple[str, str], np.ndarray] = {}
    for t in tickers:
        prices = rng.uniform(100, 300, size=len(dates))
        if nan_tickers and t in nan_tickers:
            prices = np.full(len(dates), np.nan)
        data[("Adj Close", t)] = prices
        data[("Close", t)] = prices
    df = pd.DataFrame(data, index=dates)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df


class TestDownloadWithRetry:
    """Mock yf.download to test retry logic without network calls."""

    @patch("pairs_eda.yfinance_tools.yf")
    def test_no_failures_no_retry(self, mock_yf: object, capsys: pytest.CaptureFixture[str]) -> None:
        dates = pd.bdate_range("2024-01-02", periods=30)
        tickers = ["AAPL", "MSFT"]
        mock_yf.download = lambda *a, **kw: _make_multi_dl(tickers, dates)  # type: ignore[attr-defined]

        result = download_with_retry(tickers, start="2024-01-02", end="2024-02-15", verbose=True)
        assert result.shape == (30, 2)
        out = capsys.readouterr().out
        assert "Retry" not in out

    @patch("pairs_eda.yfinance_tools.yf")
    def test_retry_recovers_failed_ticker(self, mock_yf: object, capsys: pytest.CaptureFixture[str]) -> None:
        dates = pd.bdate_range("2024-01-02", periods=30)
        tickers = ["AAPL", "MSFT", "GOOG"]
        call_count = {"n": 0}

        def fake_download(*args: object, **kwargs: object) -> pd.DataFrame:
            call_count["n"] += 1
            requested = args[0] if args else kwargs.get("tickers", tickers)
            if call_count["n"] == 1:
                return _make_multi_dl(tickers, dates, nan_tickers=["GOOG"])
            return _make_multi_dl(list(requested), dates)

        mock_yf.download = fake_download  # type: ignore[attr-defined]

        result = download_with_retry(
            tickers, start="2024-01-02", end="2024-02-15",
            max_retries=2, retry_delay=0, verbose=True,
        )
        assert "GOOG" in result.columns, "GOOG should be recovered after retry"
        assert result.shape == (30, 3)
        out = capsys.readouterr().out
        assert "recovered" in out.lower()

    @patch("pairs_eda.yfinance_tools.yf")
    def test_permanent_failure_drops_ticker(self, mock_yf: object, capsys: pytest.CaptureFixture[str]) -> None:
        dates = pd.bdate_range("2024-01-02", periods=30)
        tickers = ["AAPL", "BADTICKER"]

        mock_yf.download = lambda *a, **kw: _make_multi_dl(tickers, dates, nan_tickers=["BADTICKER"])  # type: ignore[attr-defined]

        result = download_with_retry(
            tickers, start="2024-01-02", end="2024-02-15",
            max_retries=2, retry_delay=0, verbose=True,
        )
        assert "BADTICKER" not in result.columns, "Permanently failed ticker should be dropped"
        assert result.shape == (30, 1)
        out = capsys.readouterr().out
        assert "Dropping" in out
