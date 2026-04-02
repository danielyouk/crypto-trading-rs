"""Tests for find_candidate_pairs and find_cointegrated_pairs."""

import numpy as np
import pandas as pd
import pytest

from pairs_eda.correlation import find_candidate_pairs, find_cointegrated_pairs

DATES = pd.bdate_range("2024-01-02", periods=120)

# All existing tests use short data (120 days), so min_overlap_years=0
# disables the overlap filter. New tests below exercise overlap/recency.
_NO_OVERLAP = dict(min_overlap_pct=0.0)


def _make_correlated_panel() -> pd.DataFrame:
    """
    Build fake price data where:
    - A and B are highly correlated (B = A * 1.5 + noise)
    - C is independent random walk
    - D tracks A inversely (negatively correlated)
    """
    rng = np.random.default_rng(42)
    a = 100 + np.cumsum(rng.normal(0.1, 1, len(DATES)))
    b = a * 1.5 + rng.normal(0, 0.5, len(DATES))
    c = 200 + np.cumsum(rng.normal(0, 2, len(DATES)))
    d = 300 - a * 0.8 + rng.normal(0, 0.5, len(DATES))
    return pd.DataFrame({"A": a, "B": b, "C": c, "D": d}, index=DATES)


class TestReturnsCorrelation:
    """Default mode: use_returns=True (correlate daily % changes)."""

    def test_returns_dict_sorted_descending(self) -> None:
        data = _make_correlated_panel()
        result = find_candidate_pairs(data, min_correlation=-1.0, max_correlation=1.0, **_NO_OVERLAP)
        assert isinstance(result, dict)
        values = list(result.values())
        assert values == sorted(values, reverse=True)

    def test_ab_is_top_pair(self) -> None:
        data = _make_correlated_panel()
        result = find_candidate_pairs(data, min_correlation=-1.0, max_correlation=1.0, **_NO_OVERLAP)
        first_pair = next(iter(result))
        assert first_pair == ("A", "B")
        assert result[first_pair] > 0.8

    def test_returns_correlation_lower_than_price(self) -> None:
        """Returns correlation should be lower than price correlation for trending data."""
        data = _make_correlated_panel()
        ret_corr = find_candidate_pairs(data, min_correlation=-1.0, max_correlation=1.0, **_NO_OVERLAP)
        price_corr = find_candidate_pairs(data, min_correlation=-1.0, max_correlation=1.0, use_returns=False, **_NO_OVERLAP)
        assert ret_corr[("A", "B")] < price_corr[("A", "B")]


class TestPriceCorrelation:
    """Legacy mode: use_returns=False (correlate raw price levels)."""

    def test_ab_highly_correlated(self) -> None:
        data = _make_correlated_panel()
        result = find_candidate_pairs(data, min_correlation=-1.0, max_correlation=1.0, use_returns=False, **_NO_OVERLAP)
        assert result[("A", "B")] > 0.99

    def test_max_correlation_excludes_ab(self) -> None:
        data = _make_correlated_panel()
        result = find_candidate_pairs(data, min_correlation=-1.0, max_correlation=0.98, use_returns=False, **_NO_OVERLAP)
        assert ("A", "B") not in result
        assert len(result) > 0


class TestSharedBehavior:
    """Tests that apply regardless of use_returns setting."""

    def test_top_n_limits_output(self) -> None:
        data = _make_correlated_panel()
        result = find_candidate_pairs(data, top_n=2, min_correlation=-1.0, max_correlation=1.0, **_NO_OVERLAP)
        assert len(result) == 2

    def test_top_n_none_returns_all(self) -> None:
        data = _make_correlated_panel()
        result = find_candidate_pairs(data, min_correlation=-1.0, max_correlation=1.0, **_NO_OVERLAP)
        assert len(result) == 6  # C(4,2) = 6

    def test_min_correlation_filters(self) -> None:
        data = _make_correlated_panel()
        all_pairs = find_candidate_pairs(data, min_correlation=-1.0, max_correlation=1.0, **_NO_OVERLAP)
        high_pairs = find_candidate_pairs(data, min_correlation=0.5, max_correlation=1.0, **_NO_OVERLAP)
        assert len(high_pairs) < len(all_pairs)

    def test_correlation_band_inclusive(self) -> None:
        data = _make_correlated_panel()
        result = find_candidate_pairs(data, min_correlation=0.0, max_correlation=0.95, **_NO_OVERLAP)
        for pair, val in result.items():
            assert 0.0 <= val <= 0.95, f"{pair} has r={val}, outside [0.0, 0.95]"

    def test_start_end_slices_data(self) -> None:
        data = _make_correlated_panel()
        result = find_candidate_pairs(data, start=DATES[60], end=DATES[-1], min_correlation=-1.0, max_correlation=1.0, **_NO_OVERLAP)
        assert len(result) > 0

    def test_too_few_tickers_raises(self) -> None:
        data = pd.DataFrame({"A": [1, 2, 3]}, index=pd.bdate_range("2024-01-02", periods=3))
        with pytest.raises(ValueError, match="at least 2"):
            find_candidate_pairs(data, **_NO_OVERLAP)

    def test_no_duplicate_pairs(self) -> None:
        data = _make_correlated_panel()
        result = find_candidate_pairs(data, min_correlation=-1.0, max_correlation=1.0, **_NO_OVERLAP)
        pair_set = set()
        for pair in result:
            key = tuple(sorted(pair))
            assert key not in pair_set, f"Duplicate pair: {key}"
            pair_set.add(key)


class TestOverlapFilter:
    """Tests for min_overlap_pct parameter."""

    def test_insufficient_overlap_excluded(self) -> None:
        """120 trading days with 90% overlap means min_periods=108."""
        data = _make_correlated_panel()
        # With min_overlap_pct=0.90, it needs 108 days.
        # Let's create a gap so it fails.
        data_gap = data.copy()
        data_gap.iloc[0:20, 0] = np.nan # A has only 100 days
        result = find_candidate_pairs(data_gap, min_correlation=-1.0, max_correlation=1.0, min_overlap_pct=0.90)
        assert ("A", "B") not in result

    def test_sufficient_overlap_included(self) -> None:
        """120 trading days with 50% overlap means min_periods=60."""
        data = _make_correlated_panel()
        data_gap = data.copy()
        data_gap.iloc[0:20, 0] = np.nan # A has 100 days
        result = find_candidate_pairs(data_gap, min_correlation=-1.0, max_correlation=1.0, min_overlap_pct=0.50)
        assert ("A", "B") in result or ("B", "A") in result

    def test_partial_overlap_tickers(self) -> None:
        """Ticker E starts halfway through — only pairs with enough overlap survive."""
        rng = np.random.default_rng(99)
        long_dates = pd.bdate_range("2020-01-02", periods=600)
        a = 100 + np.cumsum(rng.normal(0.1, 1, 600))
        b = a * 1.2 + rng.normal(0, 0.5, 600)
        e_prices = np.full(600, np.nan)
        e_prices[400:] = 50 + np.cumsum(rng.normal(0.1, 1, 200))
        data = pd.DataFrame({"A": a, "B": b, "E": e_prices}, index=long_dates)

        # min_overlap_pct=0.50 → need 300 days overlap
        # A-B: 600 days overlap → pass
        # A-E, B-E: 200 days overlap → fail
        result = find_candidate_pairs(data, min_correlation=-1.0, max_correlation=1.0, min_overlap_pct=0.50)
        assert ("A", "B") in result or ("B", "A") in result
        assert ("A", "E") not in result and ("E", "A") not in result


class TestInputValidation:
    """Fail-fast on invalid parameters."""

    def test_min_gt_max_raises(self) -> None:
        data = _make_correlated_panel()
        with pytest.raises(ValueError, match="min_correlation.*>.*max_correlation"):
            find_candidate_pairs(data, min_correlation=0.9, max_correlation=0.4, **_NO_OVERLAP)

    def test_min_correlation_out_of_range(self) -> None:
        data = _make_correlated_panel()
        with pytest.raises(ValueError, match="min_correlation.*outside"):
            find_candidate_pairs(data, min_correlation=-1.5, max_correlation=1.0, **_NO_OVERLAP)

    def test_max_correlation_out_of_range(self) -> None:
        data = _make_correlated_panel()
        with pytest.raises(ValueError, match="max_correlation.*outside"):
            find_candidate_pairs(data, min_correlation=-1.0, max_correlation=1.5, **_NO_OVERLAP)

    def test_invalid_overlap_pct_raises(self) -> None:
        data = _make_correlated_panel()
        with pytest.raises(ValueError, match="min_overlap_pct"):
            find_candidate_pairs(data, min_overlap_pct=-0.1)
        with pytest.raises(ValueError, match="min_overlap_pct"):
            find_candidate_pairs(data, min_overlap_pct=1.1)

    def test_negative_top_n_raises(self) -> None:
        data = _make_correlated_panel()
        with pytest.raises(ValueError, match="top_n"):
            find_candidate_pairs(data, top_n=-1, **_NO_OVERLAP)


class TestEdgeCases:
    """Missing data, unsorted index, and other edge cases."""

    def test_nan_gaps_do_not_create_false_returns(self) -> None:
        """Gaps in price data should remain NaN in returns, not become 0."""
        dates = pd.bdate_range("2020-01-02", periods=100)
        rng = np.random.default_rng(55)
        a = 100 + np.cumsum(rng.normal(0.1, 1, 100))
        b = a * 1.2 + rng.normal(0, 0.5, 100)
        b[30:40] = np.nan  # gap in B

        data = pd.DataFrame({"A": a, "B": b}, index=dates)
        result = find_candidate_pairs(data, min_correlation=-1.0, max_correlation=1.0, **_NO_OVERLAP)
        assert ("A", "B") in result
        # With NaN gap, correlation uses fewer observations but should still work
        assert result[("A", "B")] > 0.5

    def test_unsorted_index_still_correct(self) -> None:
        """Shuffled index should produce same results as sorted."""
        data = _make_correlated_panel()
        shuffled = data.sample(frac=1, random_state=42)  # shuffle rows
        assert not shuffled.index.is_monotonic_increasing

        sorted_result = find_candidate_pairs(data, min_correlation=-1.0, max_correlation=1.0, **_NO_OVERLAP)
        shuffled_result = find_candidate_pairs(shuffled, min_correlation=-1.0, max_correlation=1.0, **_NO_OVERLAP)

        assert sorted_result.keys() == shuffled_result.keys()
        for pair in sorted_result:
            assert abs(sorted_result[pair] - shuffled_result[pair]) < 1e-10


# ---------------------------------------------------------------------------
# find_cointegrated_pairs
# ---------------------------------------------------------------------------

def _make_cointegrated_panel(n: int = 600) -> pd.DataFrame:
    """Build price data with one cointegrated pair (A-B) and one non-cointegrated pair (A-C).

    A and B share a common stochastic trend plus small stationary noise,
    so the spread is mean-reverting. C is an independent random walk.
    """
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2020-01-02", periods=n)
    common = np.cumsum(rng.normal(0.05, 0.5, n))
    a = 100.0 + common + rng.normal(0, 0.3, n)
    b = 80.0 + common * 0.8 + rng.normal(0, 0.3, n)
    c = 120.0 + np.cumsum(rng.normal(0.02, 0.6, n))
    return pd.DataFrame({"A": a, "B": b, "C": c}, index=dates)


class TestFindCointegratedPairs:
    def test_cointegrated_pair_passes(self) -> None:
        prices = _make_cointegrated_panel()
        pairs = [("A", "B"), ("A", "C"), ("B", "C")]
        result = find_cointegrated_pairs(pairs, prices, significance=0.05)
        assert ("A", "B") in result

    def test_independent_pair_excluded(self) -> None:
        prices = _make_cointegrated_panel()
        pairs = [("A", "C")]
        result = find_cointegrated_pairs(pairs, prices, significance=0.05)
        assert ("A", "C") not in result

    def test_missing_ticker_skipped(self) -> None:
        prices = _make_cointegrated_panel()
        pairs = [("A", "MISSING"), ("A", "B")]
        result = find_cointegrated_pairs(pairs, prices, significance=0.05)
        assert ("A", "MISSING") not in result
        assert ("A", "B") in result

    def test_short_overlap_skipped(self) -> None:
        """Pairs with < 252 overlapping days are skipped."""
        prices = _make_cointegrated_panel(n=200)  # 200 < 252
        pairs = [("A", "B")]
        result = find_cointegrated_pairs(pairs, prices, significance=0.05)
        assert len(result) == 0

    def test_preserves_input_order(self) -> None:
        prices = _make_cointegrated_panel()
        pairs = [("B", "C"), ("A", "B"), ("A", "C")]
        result = find_cointegrated_pairs(pairs, prices, significance=0.50)
        # With a very loose threshold, A-B should pass; check order matches input
        if len(result) >= 2:
            input_order = [p for p in pairs if p in result]
            assert result == input_order

    def test_empty_input_returns_empty(self) -> None:
        prices = _make_cointegrated_panel()
        result = find_cointegrated_pairs([], prices, significance=0.05)
        assert result == []
