"""Tests for find_candidate_pairs."""

import numpy as np
import pandas as pd
import pytest

from pairs_eda.correlation import find_candidate_pairs

DATES = pd.bdate_range("2024-01-02", periods=120)


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
        result = find_candidate_pairs(data, min_correlation=-1.0, max_correlation=1.0)
        assert isinstance(result, dict)
        values = list(result.values())
        assert values == sorted(values, reverse=True)

    def test_ab_is_top_pair(self) -> None:
        data = _make_correlated_panel()
        result = find_candidate_pairs(data, min_correlation=-1.0, max_correlation=1.0)
        first_pair = next(iter(result))
        assert first_pair == ("A", "B")
        assert result[first_pair] > 0.8

    def test_returns_correlation_lower_than_price(self) -> None:
        """Returns correlation should be lower than price correlation for trending data."""
        data = _make_correlated_panel()
        ret_corr = find_candidate_pairs(data, min_correlation=-1.0, max_correlation=1.0)
        price_corr = find_candidate_pairs(data, min_correlation=-1.0, max_correlation=1.0, use_returns=False)
        assert ret_corr[("A", "B")] < price_corr[("A", "B")]


class TestPriceCorrelation:
    """Legacy mode: use_returns=False (correlate raw price levels)."""

    def test_ab_highly_correlated(self) -> None:
        data = _make_correlated_panel()
        result = find_candidate_pairs(data, min_correlation=-1.0, max_correlation=1.0, use_returns=False)
        assert result[("A", "B")] > 0.99

    def test_max_correlation_excludes_ab(self) -> None:
        data = _make_correlated_panel()
        result = find_candidate_pairs(data, min_correlation=-1.0, max_correlation=0.98, use_returns=False)
        assert ("A", "B") not in result
        assert len(result) > 0


class TestSharedBehavior:
    """Tests that apply regardless of use_returns setting."""

    def test_top_n_limits_output(self) -> None:
        data = _make_correlated_panel()
        result = find_candidate_pairs(data, top_n=2, min_correlation=-1.0, max_correlation=1.0)
        assert len(result) == 2

    def test_top_n_none_returns_all(self) -> None:
        data = _make_correlated_panel()
        result = find_candidate_pairs(data, min_correlation=-1.0, max_correlation=1.0)
        assert len(result) == 6  # C(4,2) = 6

    def test_min_correlation_filters(self) -> None:
        data = _make_correlated_panel()
        all_pairs = find_candidate_pairs(data, min_correlation=-1.0, max_correlation=1.0)
        high_pairs = find_candidate_pairs(data, min_correlation=0.5, max_correlation=1.0)
        assert len(high_pairs) < len(all_pairs)

    def test_correlation_band_inclusive(self) -> None:
        data = _make_correlated_panel()
        result = find_candidate_pairs(data, min_correlation=0.0, max_correlation=0.95)
        for pair, val in result.items():
            assert 0.0 <= val <= 0.95, f"{pair} has r={val}, outside [0.0, 0.95]"

    def test_start_end_slices_data(self) -> None:
        data = _make_correlated_panel()
        result = find_candidate_pairs(data, start=DATES[60], end=DATES[-1], min_correlation=-1.0, max_correlation=1.0)
        assert len(result) > 0

    def test_too_few_tickers_raises(self) -> None:
        data = pd.DataFrame({"A": [1, 2, 3]}, index=pd.bdate_range("2024-01-02", periods=3))
        with pytest.raises(ValueError, match="at least 2"):
            find_candidate_pairs(data)

    def test_no_duplicate_pairs(self) -> None:
        data = _make_correlated_panel()
        result = find_candidate_pairs(data, min_correlation=-1.0, max_correlation=1.0)
        pair_set = set()
        for pair in result:
            key = tuple(sorted(pair))
            assert key not in pair_set, f"Duplicate pair: {key}"
            pair_set.add(key)
