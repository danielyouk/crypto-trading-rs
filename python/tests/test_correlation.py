"""Tests for find_candidate_pairs."""

import numpy as np
import pandas as pd
import pytest

from pairs_eda.correlation import find_candidate_pairs

DATES = pd.bdate_range("2024-01-02", periods=120)

# All existing tests use short data (120 days), so min_overlap_years=0
# disables the overlap filter. New tests below exercise overlap/recency.
_NO_OVERLAP = dict(min_overlap_years=0, recent_years=0)


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
    """Tests for min_overlap_years parameter."""

    def test_insufficient_overlap_excluded(self) -> None:
        """120 trading days < 1 year (252 days) → all pairs excluded with min_overlap_years=1."""
        data = _make_correlated_panel()
        result = find_candidate_pairs(data, min_correlation=-1.0, max_correlation=1.0, min_overlap_years=1.0, recent_years=0)
        assert len(result) == 0

    def test_sufficient_overlap_included(self) -> None:
        """120 trading days > 0.4 years (101 days) → pairs included."""
        data = _make_correlated_panel()
        result = find_candidate_pairs(data, min_correlation=-1.0, max_correlation=1.0, min_overlap_years=0.4, recent_years=0)
        assert len(result) > 0

    def test_partial_overlap_tickers(self) -> None:
        """Ticker E starts halfway through — only pairs with enough overlap survive."""
        rng = np.random.default_rng(99)
        long_dates = pd.bdate_range("2020-01-02", periods=600)
        a = 100 + np.cumsum(rng.normal(0.1, 1, 600))
        b = a * 1.2 + rng.normal(0, 0.5, 600)
        e_prices = np.full(600, np.nan)
        e_prices[400:] = 50 + np.cumsum(rng.normal(0.1, 1, 200))
        data = pd.DataFrame({"A": a, "B": b, "E": e_prices}, index=long_dates)

        # min_overlap_years=1.5 → need 378 days overlap
        # A-B: 600 days overlap → pass
        # A-E, B-E: 200 days overlap → fail
        result = find_candidate_pairs(data, min_correlation=-1.0, max_correlation=1.0, min_overlap_years=1.5, recent_years=0)
        assert ("A", "B") in result or ("B", "A") in result
        assert ("A", "E") not in result and ("E", "A") not in result


class TestRecentCorrelation:
    """Tests for the dual condition (full + recent)."""

    def test_degraded_relationship_excluded(self) -> None:
        """A-B correlated in first half, uncorrelated in second half → excluded by recency check."""
        rng = np.random.default_rng(77)
        long_dates = pd.bdate_range("2020-01-02", periods=800)

        a = 100 + np.cumsum(rng.normal(0.1, 1, 800))

        # B tracks A for first 500 days, then diverges
        b = np.empty(800)
        b[:500] = a[:500] * 1.3 + rng.normal(0, 0.5, 500)
        b[500:] = 200 + np.cumsum(rng.normal(0, 2, 300))

        c = 150 + np.cumsum(rng.normal(0, 1, 800))
        data = pd.DataFrame({"A": a, "B": b, "C": c}, index=long_dates)

        # Without recency check → A-B likely passes on full period
        no_recent = find_candidate_pairs(
            data, min_correlation=-1.0, max_correlation=1.0,
            min_overlap_years=0, recent_years=0,
        )

        # With recency check (last ~1 year = 252 days, within the diverged zone)
        with_recent = find_candidate_pairs(
            data, min_correlation=0.3, max_correlation=1.0,
            min_overlap_years=0, recent_years=1.0,
        )

        ab_in_no_recent = ("A", "B") in no_recent
        ab_in_with_recent = ("A", "B") in with_recent

        assert ab_in_no_recent, "A-B should appear without recency check"
        assert not ab_in_with_recent, "A-B should be excluded by recency check (relationship degraded)"

    def test_recent_years_zero_disables_check(self) -> None:
        data = _make_correlated_panel()
        result = find_candidate_pairs(data, min_correlation=-1.0, max_correlation=1.0, min_overlap_years=0, recent_years=0)
        assert len(result) == 6
