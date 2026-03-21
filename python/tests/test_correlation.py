"""Tests for find_top_pairs."""

import numpy as np
import pandas as pd
import pytest

from pairs_eda.correlation import find_top_pairs

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


class TestFindTopPairs:
    def test_returns_pairs_sorted_by_correlation(self) -> None:
        data = _make_correlated_panel()
        pairs = find_top_pairs(data, top_n=6)
        assert len(pairs) <= 6
        assert all(isinstance(p, tuple) and len(p) == 2 for p in pairs)
        assert pairs[0] == ("A", "B"), "A-B should be the most correlated pair"

    def test_top_n_limits_output(self) -> None:
        data = _make_correlated_panel()
        pairs = find_top_pairs(data, top_n=2)
        assert len(pairs) == 2

    def test_min_correlation_filters(self) -> None:
        data = _make_correlated_panel()
        pairs_all = find_top_pairs(data, top_n=100, min_correlation=-1.0)
        pairs_high = find_top_pairs(data, top_n=100, min_correlation=0.9)
        assert len(pairs_high) < len(pairs_all)

    def test_start_end_slices_data(self) -> None:
        data = _make_correlated_panel()
        pairs_full = find_top_pairs(data, top_n=3)
        pairs_half = find_top_pairs(data, start=DATES[60], end=DATES[-1], top_n=6, min_correlation=-1.0)
        assert len(pairs_half) > 0
        # Results may differ because correlation changes with window

    def test_too_few_tickers_raises(self) -> None:
        data = pd.DataFrame({"A": [1, 2, 3]}, index=pd.bdate_range("2024-01-02", periods=3))
        with pytest.raises(ValueError, match="at least 2"):
            find_top_pairs(data)

    def test_no_duplicate_pairs(self) -> None:
        data = _make_correlated_panel()
        pairs = find_top_pairs(data, top_n=100)
        pair_set = set()
        for a, b in pairs:
            key = tuple(sorted([a, b]))
            assert key not in pair_set, f"Duplicate pair: {key}"
            pair_set.add(key)
