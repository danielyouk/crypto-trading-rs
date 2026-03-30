"""Unit tests for pairs_eda.backtesting (functional backtesting pipeline)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from pairs_eda.backtesting import (
    PairPipelineState,
    _MIN_VIABLE_MARGIN,
    calculate_margin,
    compute_signals,
    compute_zscore,
    grid_search_pair,
    run_pair_pipeline,
    summarize_signals,
)


# ---------------------------------------------------------------------------
# Fixtures — small synthetic price data
# ---------------------------------------------------------------------------

def _make_prices(n: int = 200, seed: int = 42) -> tuple[pd.Series, pd.Series]:
    """Two correlated random-walk price series with a DatetimeIndex."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n)
    noise_a = rng.normal(0, 0.01, n)
    noise_b = rng.normal(0, 0.01, n)
    common = rng.normal(0, 0.005, n)

    prices_a = pd.Series(
        100.0 * np.exp(np.cumsum(noise_a + common)),
        index=dates,
        name="AAAA",
    )
    prices_b = pd.Series(
        80.0 * np.exp(np.cumsum(noise_b + common)),
        index=dates,
        name="BBBB",
    )
    return prices_a, prices_b


@pytest.fixture
def prices():
    return _make_prices()


# ---------------------------------------------------------------------------
# compute_zscore
# ---------------------------------------------------------------------------

class TestComputeZscore:
    def test_output_columns(self, prices):
        pa, pb = prices
        df = compute_zscore(pa, pb, window=10)
        assert set(df.columns) == {"stock1", "stock2", "ratio", "ma", "msd", "zscore"}

    def test_output_length(self, prices):
        pa, pb = prices
        df = compute_zscore(pa, pb, window=10)
        assert len(df) == len(pa)

    def test_first_row_nan(self, prices):
        """First row of MA should be NaN (Kalman not yet initialised)."""
        pa, pb = prices
        window = 10
        df = compute_zscore(pa, pb, window=window)
        assert pd.isna(df["ma"].iloc[0])

    def test_ratio_is_log(self, prices):
        pa, pb = prices
        df = compute_zscore(pa, pb, window=5)
        expected = np.log(pa / pb)
        np.testing.assert_allclose(df["ratio"].values, expected.values)

    def test_zscore_formula(self, prices):
        """Z-score = (ratio - ma) / msd where ma and msd are not NaN."""
        pa, pb = prices
        df = compute_zscore(pa, pb, window=5)
        manual = (df["ratio"] - df["ma"]) / df["msd"]
        # Both must be non-NaN for the comparison
        valid = df["zscore"].notna() & manual.notna()
        np.testing.assert_allclose(
            df["zscore"][valid].values, manual[valid].values
        )


# ---------------------------------------------------------------------------
# compute_signals
# ---------------------------------------------------------------------------

class TestComputeSignals:
    def test_only_valid_values(self, prices):
        pa, pb = prices
        df = compute_zscore(pa, pb, window=10)
        sig = compute_signals(df["zscore"], threshold=2.0)
        assert set(sig.unique()).issubset({-1, 0, 1})

    def test_extreme_zscore_holds(self):
        """Z-score >= 5 should NOT trigger a signal (NaN -> ffill)."""
        idx = pd.date_range("2020-01-01", periods=5)
        zscore = pd.Series([0.0, 0.5, 6.0, 6.5, 0.5], index=idx)
        sig = compute_signals(zscore, threshold=2.0)
        assert sig.iloc[0] == 0
        assert sig.iloc[2] == 0  # extreme -> holds previous (0)
        assert sig.iloc[4] == 0  # back in neutral zone

    def test_threshold_triggers(self):
        idx = pd.date_range("2020-01-01", periods=4)
        zscore = pd.Series([0.0, 3.0, -3.0, 0.0], index=idx)
        sig = compute_signals(zscore, threshold=2.0)
        assert sig.iloc[1] == -1  # zscore > threshold -> short A
        assert sig.iloc[2] == 1   # zscore < -threshold -> long A
        assert sig.iloc[3] == 0   # neutral zone


# ---------------------------------------------------------------------------
# summarize_signals
# ---------------------------------------------------------------------------

class TestSummarizeSignals:
    def test_basic_structure(self, prices):
        pa, pb = prices
        df = compute_zscore(pa, pb, window=10)
        sig = compute_signals(df["zscore"], threshold=2.0)
        summary = summarize_signals(df["stock1"], df["stock2"], sig)
        assert isinstance(summary, list)
        assert len(summary) > 0
        required_keys = {
            "signal", "time_start", "time_end",
            "stock1_start_price", "stock1_final_price",
            "stock2_start_price", "stock2_final_price",
        }
        assert required_keys == set(summary[0].keys())

    def test_periods_cover_full_range(self, prices):
        """First period starts at first date, last period ends at last date."""
        pa, pb = prices
        df = compute_zscore(pa, pb, window=10)
        sig = compute_signals(df["zscore"], threshold=2.0)
        summary = summarize_signals(df["stock1"], df["stock2"], sig)
        assert summary[0]["time_start"] == pa.index[0]
        assert summary[-1]["time_end"] == pa.index[-1]

    def test_end_price_equals_next_start(self, prices):
        """Final price of period N = start price of period N+1."""
        pa, pb = prices
        df = compute_zscore(pa, pb, window=10)
        sig = compute_signals(df["zscore"], threshold=2.0)
        summary = summarize_signals(df["stock1"], df["stock2"], sig)
        for i in range(len(summary) - 1):
            assert summary[i]["stock1_final_price"] == pytest.approx(
                summary[i + 1]["stock1_start_price"]
            )


# ---------------------------------------------------------------------------
# calculate_margin
# ---------------------------------------------------------------------------

class TestCalculateMargin:
    def _run_margin(self, prices, fractional: bool) -> dict:
        pa, pb = prices
        df = compute_zscore(pa, pb, window=10)
        sig = compute_signals(df["zscore"], threshold=2.0)
        summary = summarize_signals(df["stock1"], df["stock2"], sig)
        return calculate_margin(summary, 3000.0, 0.25, fractional=fractional)

    def test_returns_dict_keys(self, prices):
        result = self._run_margin(prices, fractional=True)
        assert "margin" in result
        assert "trade_count" in result
        assert "trade_log" in result

    def test_no_trades_keeps_init(self):
        """If all signals are 0, margin should stay at init."""
        summary = [
            {
                "signal": 0,
                "time_start": pd.Timestamp("2020-01-01"),
                "time_end": pd.Timestamp("2020-06-01"),
                "stock1_start_price": 100.0,
                "stock1_final_price": 110.0,
                "stock2_start_price": 80.0,
                "stock2_final_price": 85.0,
            }
        ]
        result = calculate_margin(summary, 3000.0, 0.25)
        assert result["margin"] == 3000.0
        assert result["trade_count"] == 0

    def test_fractional_vs_integer(self, prices):
        """Fractional mode should generally differ from integer mode."""
        frac = self._run_margin(prices, fractional=True)
        integ = self._run_margin(prices, fractional=False)
        assert frac["margin"] != pytest.approx(integ["margin"], abs=0.01)

    def test_trade_log_length_matches_count(self, prices):
        result = self._run_margin(prices, fractional=True)
        assert len(result["trade_log"]) == result["trade_count"]



# ---------------------------------------------------------------------------
# run_pair_pipeline (stateful intermediates)
# ---------------------------------------------------------------------------

class TestRunPairPipeline:
    def test_returns_state_object(self, prices):
        pa, pb = prices
        pipeline_state = run_pair_pipeline(
            pa,
            pb,
            window=10,
            zscore_threshold=2.0,
            margin_init=3000.0,
            margin_ratio=0.25,
        )
        assert isinstance(pipeline_state, PairPipelineState)
        # pair_stats has zscore stats + signal in one DataFrame
        pair_stats_cols = {"stock1", "stock2", "ratio", "ma", "msd", "zscore", "signal"}
        assert pair_stats_cols.issubset(set(pipeline_state.pair_stats.columns))
        assert len(pipeline_state.pair_stats) == len(pa)
        # signal_summary is a DataFrame with merged P&L columns for active trades
        assert isinstance(pipeline_state.signal_summary, pd.DataFrame)
        assert "signal" in pipeline_state.signal_summary.columns
        assert "pnl" in pipeline_state.signal_summary.columns
        assert "commission" in pipeline_state.signal_summary.columns
        assert "margin_after" in pipeline_state.signal_summary.columns
        # neutral periods (signal == 0) have NaN pnl; active trades have values
        neutral_mask = pipeline_state.signal_summary["signal"] == 0
        assert pipeline_state.signal_summary.loc[neutral_mask, "pnl"].isna().all()
        assert isinstance(pipeline_state.margin_final, float)
        # liquidation_date is None when account stays solvent
        assert pipeline_state.liquidation_date is None or isinstance(
            pipeline_state.liquidation_date, str
        )
        # n_stops is 0 when stop_loss_pct is disabled (default)
        assert isinstance(pipeline_state.n_stops, int)
        assert pipeline_state.n_stops == 0

    def test_margin_final_matches_manual_pipeline(self, prices):
        """margin_final must equal the result of manually chaining all four steps."""
        pa, pb = prices
        zdf = compute_zscore(pa, pb, window=10)
        sig = compute_signals(zdf["zscore"], threshold=2.0)
        summary = summarize_signals(zdf["stock1"], zdf["stock2"], sig)
        manual = calculate_margin(summary, 3000.0, 0.25, fractional=True)

        state = run_pair_pipeline(
            pa, pb,
            window=10,
            zscore_threshold=2.0,
            margin_init=3000.0,
            margin_ratio=0.25,
            pair=("AAAA", "BBBB"),
        )
        assert state.margin_final == pytest.approx(manual["margin"])
        assert state.pair == ("AAAA", "BBBB")
        assert state.window == 10
        assert state.zscore_threshold == 2.0

    def test_commission_in_active_trades(self, prices):
        pa, pb = prices
        pipeline_state = run_pair_pipeline(
            pa,
            pb,
            window=10,
            zscore_threshold=2.0,
            margin_init=3000.0,
            margin_ratio=0.25,
        )
        active = pipeline_state.signal_summary.dropna(subset=["commission"])
        assert len(active) > 0
        assert (active["commission"] > 0).all()

    def test_bankruptcy_stops_simulation(self):
        """With a tiny margin, account goes bankrupt quickly and stops cleanly."""
        # Use tiny margin so the account bankrupts within a handful of trades.
        pa, pb = _make_prices(n=500, seed=7)
        pipeline_state = run_pair_pipeline(
            pa,
            pb,
            window=10,
            zscore_threshold=1.0,   # low threshold → many trades → fast bankruptcy
            margin_init=5.0,        # tiny margin (just above _MIN_VIABLE_MARGIN=2)
            margin_ratio=0.25,
        )
        # liquidation_date must be set when account goes bankrupt
        assert pipeline_state.liquidation_date is not None
        # final margin must be below _MIN_VIABLE_MARGIN (was too small to trade further)
        assert pipeline_state.margin_final < 2.0
        # no e-16 residual units in signal_summary — executed trades only
        executed = pipeline_state.signal_summary.dropna(subset=["stock1_units"])
        assert len(executed) > 0
        assert (executed["stock1_units"] > 1e-10).all()


# ---------------------------------------------------------------------------
# grid_search_pair
# ---------------------------------------------------------------------------

class TestGridSearchPair:
    def test_result_count(self, prices):
        pa, pb = prices
        windows = range(5, 8)       # 3 values
        thresholds = [2.0, 3.0]     # 2 values
        results = grid_search_pair(
            pa, pb,
            windows=windows,
            zscore_thresholds=thresholds,
            margin_init=3000,
            margin_ratio=0.25,
        )
        assert len(results) == 3 * 2

    def test_sorted_descending(self, prices):
        pa, pb = prices
        results = grid_search_pair(
            pa, pb,
            windows=range(5, 10),
            zscore_thresholds=[2.0, 2.5, 3.0],
            margin_init=3000,
            margin_ratio=0.25,
        )
        margins = [r["margin"] for r in results]
        assert margins == sorted(margins, reverse=True)

    def test_n_stops_in_compact_result(self, prices):
        pa, pb = prices
        results = grid_search_pair(
            pa, pb,
            windows=[10],
            zscore_thresholds=[2.0],
            margin_init=3000,
            margin_ratio=0.25,
        )
        assert "n_stops" in results[0]
        assert isinstance(results[0]["n_stops"], int)


# ---------------------------------------------------------------------------
# stop-loss & cooldown
# ---------------------------------------------------------------------------

def _make_stop_loss_scenario(
    margin_init: float = 3000.0,
    margin_ratio: float = 0.25,
) -> tuple[list[dict], pd.DataFrame]:
    """Controlled scenario where stop-loss must trigger on trade 1 day 3.

    Setup:
        margin=3000, margin_ratio=0.25 → buying_power=12000, half_bp=6000
        s1_units = 6000 / 100 = 60  (stock1 starts at $100)
        s2_units = 6000 /  80 = 75  (stock2 stays at $80, unchanged)
        signal = 1: long stock1, short stock2

        Day 0 (entry): stock1=100, unrealized=0
        Day 1        : stock1=100, unrealized=0
        Day 2        : stock1=88  → unrealized = (88-100)*60 = -720
                       5% of buying_power = 0.05 * 12000 = 600
                       -720 < -600 → stop-loss triggered
    """
    dates = pd.bdate_range("2023-01-02", periods=10)
    s1 = [100.0, 100.0, 88.0] + [88.0] * 7   # drops on day 2
    s2 = [80.0] * 10

    pair_stats = pd.DataFrame({"stock1": s1, "stock2": s2}, index=dates)

    signal_summary = [
        {
            "signal": 1,
            "time_start": dates[0],
            "time_end": dates[-1],
            "stock1_start_price": 100.0,
            "stock1_final_price": s1[-1],
            "stock2_start_price": 80.0,
            "stock2_final_price": s2[-1],
        }
    ]
    return signal_summary, pair_stats


class TestStopLoss:
    def test_stop_loss_triggers_on_large_drop(self):
        """Stop-loss fires when unrealized loss exceeds 5% of buying power."""
        summary, pair_stats = _make_stop_loss_scenario()
        result = calculate_margin(
            summary, 3000.0, 0.25,
            fractional=True,
            stop_loss_pct=0.05,
            pair_stats=pair_stats,
        )
        assert result["n_stops"] == 1
        trade = result["trade_log"][0]
        assert trade["stopped_out"] is True
        assert trade["stop_date"] is not None
        # Exit at stop-day price ($88), not the original final price
        assert trade["stock1_final_price"] == pytest.approx(88.0)

    def test_stop_loss_disabled_when_pct_zero(self):
        """stop_loss_pct=0.0 must not change behaviour vs. no stop-loss."""
        summary, pair_stats = _make_stop_loss_scenario()
        with_stop = calculate_margin(
            summary, 3000.0, 0.25, fractional=True,
            stop_loss_pct=0.0, pair_stats=pair_stats,
        )
        without_stop = calculate_margin(summary, 3000.0, 0.25, fractional=True)
        assert with_stop["margin"] == pytest.approx(without_stop["margin"])
        assert with_stop["n_stops"] == 0

    def test_stop_loss_ignored_without_pair_stats(self):
        """stop_loss_pct > 0 but pair_stats=None → behaves like no stop-loss."""
        summary, _ = _make_stop_loss_scenario()
        with_data = calculate_margin(
            summary, 3000.0, 0.25, fractional=True,
            stop_loss_pct=0.05, pair_stats=None,
        )
        without = calculate_margin(summary, 3000.0, 0.25, fractional=True)
        assert with_data["margin"] == pytest.approx(without["margin"])

    def test_cooldown_skips_subsequent_trade(self):
        """After stop-loss, trades within cooldown_days are skipped entirely."""
        dates = pd.bdate_range("2023-01-02", periods=20)
        # Trade 1 (days 0–4): stock1 drops on day 2 → stop-loss
        # Trade 2 (days 5–9): starts immediately after trade 1 ends
        # With cooldown_days=7 the stop-loss on day 2 blocks until day 9,
        # so trade 2 (starting day 5) should be skipped.
        s1 = [100.0, 100.0, 88.0] + [88.0] * 17
        s2 = [80.0] * 20
        pair_stats = pd.DataFrame({"stock1": s1, "stock2": s2}, index=dates)

        summary = [
            {
                "signal": 1,
                "time_start": dates[0],
                "time_end": dates[5],
                "stock1_start_price": 100.0,
                "stock1_final_price": s1[5],
                "stock2_start_price": 80.0,
                "stock2_final_price": s2[5],
            },
            {
                "signal": 1,
                "time_start": dates[5],
                "time_end": dates[10],
                "stock1_start_price": s1[5],
                "stock1_final_price": s1[10],
                "stock2_start_price": s2[5],
                "stock2_final_price": s2[10],
            },
        ]
        result_with_cooldown = calculate_margin(
            summary, 3000.0, 0.25, fractional=True,
            stop_loss_pct=0.05, pair_stats=pair_stats, cooldown_days=7,
        )
        # Only 1 trade executed (trade 2 skipped by cooldown)
        assert result_with_cooldown["trade_count"] == 1

        result_no_cooldown = calculate_margin(
            summary, 3000.0, 0.25, fractional=True,
            stop_loss_pct=0.05, pair_stats=pair_stats, cooldown_days=0,
        )
        # Both trades execute (no cooldown)
        assert result_no_cooldown["trade_count"] == 2

    def test_run_pair_pipeline_stop_loss_reduces_n_stops(self):
        """With stop_loss_pct=0.05, n_stops ≥ 0 and signal_summary has stop columns."""
        pa, pb = _make_prices(n=300, seed=99)
        state = run_pair_pipeline(
            pa, pb,
            window=10,
            zscore_threshold=2.0,
            margin_init=3000.0,
            margin_ratio=0.25,
            stop_loss_pct=0.05,
        )
        assert isinstance(state.n_stops, int)
        assert state.n_stops >= 0
        assert "stopped_out" in state.signal_summary.columns
        assert "stop_date" in state.signal_summary.columns
