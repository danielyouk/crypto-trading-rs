"""Tests for Kalman filter drift recovery in compute_zscore.

These tests verify that after a structural break (price jump), the
Kalman-based Z-score normalises significantly faster than the legacy
SMA approach — the core design requirement from the pairs-trading pipeline.

Empirical recovery benchmarks (window=60, daily_vol=0.005, +20 % jump):
    ┌─────────────────────────────────────────────────┐
    │  Method          │  Recovery (days to |z| ≤ 1)  │
    │──────────────────┼──────────────────────────────│
    │  SMA             │  21–35 days                  │
    │  EMA             │  12–20 days                  │
    │  Kalman (auto Q) │  10–20 days                  │
    │  Kalman (5× Q)   │   5–10 days   ← target      │
    └─────────────────────────────────────────────────┘

Synthetic data strategy:
    Two perfectly cointegrated random walks share a common factor.
    At a known date, stock A receives a permanent +20 % level shift.
    We measure how many days until |z_score| ≤ 1.0 after the jump.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pairs_eda.backtesting import KalmanParams, compute_zscore


# ---------------------------------------------------------------------------
# Helpers — synthetic data with controlled structural break
# ---------------------------------------------------------------------------

def _make_cointegrated_pair_with_jump(
    n_days: int = 300,
    jump_day: int = 150,
    jump_pct: float = 0.20,
    daily_vol: float = 0.005,
    seed: int = 42,
) -> tuple[pd.Series, pd.Series, int]:
    """Two cointegrated price series where stock A jumps on ``jump_day``.

    Before jump:  A and B move together (shared common factor + idio noise).
    On jump_day:  A receives a permanent +``jump_pct`` level shift.
    After jump:   A continues from the new level with the same volatility.

    Args:
        daily_vol: Per-stock idiosyncratic volatility.  0.005 yields a ratio
                   std ≈ 0.007/day, making a 20 % jump clearly visible
                   (peak |z| ≈ 8–12).

    Returns:
        (prices_a, prices_b, jump_idx)
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_days)

    common = rng.normal(0, 0.005, n_days)
    noise_a = rng.normal(0, daily_vol, n_days)
    noise_b = rng.normal(0, daily_vol, n_days)

    log_ret_a = noise_a + common
    log_ret_b = noise_b + common

    # Permanent level shift on jump_day
    log_ret_a[jump_day] += np.log1p(jump_pct)

    prices_a = pd.Series(
        100.0 * np.exp(np.cumsum(log_ret_a)), index=dates, name="JUMP_A",
    )
    prices_b = pd.Series(
        100.0 * np.exp(np.cumsum(log_ret_b)), index=dates, name="STABLE_B",
    )
    return prices_a, prices_b, jump_day


def _days_until_zscore_normalises(
    zscore: pd.Series,
    jump_idx: int,
    threshold: float = 1.0,
) -> int | None:
    """Count trading days after ``jump_idx`` until |z_score| ≤ threshold.

    Returns None if z_score never normalises within the available data.
    """
    post_jump = zscore.iloc[jump_idx + 1:]
    for offset, val in enumerate(post_jump):
        if np.isfinite(val) and abs(val) <= threshold:
            return offset + 1
    return None


def _compute_sma_zscore(
    prices_a: pd.Series,
    prices_b: pd.Series,
    window: int,
) -> pd.Series:
    """Legacy SMA-based Z-score (the method Kalman replaced).

    ┌────────────────────────────────────────────────────┐
    │  ratio  = log(A / B)                               │
    │  ma     = ratio.rolling(window).mean().shift(1)    │
    │  msd    = ratio.rolling(window).std().shift(1)     │
    │  zscore = (ratio - ma) / msd                       │
    └────────────────────────────────────────────────────┘
    """
    ratio = np.log(prices_a / prices_b)
    ma = ratio.rolling(window=window).mean().shift(1)
    msd = ratio.rolling(window=window).std().shift(1)
    return (ratio - ma) / msd


def _compute_ema_zscore(
    prices_a: pd.Series,
    prices_b: pd.Series,
    window: int,
) -> pd.Series:
    """EMA-based Z-score — a simpler adaptive alternative to Kalman.

    Uses exponential moving average (span=window) for the mean and
    exponential moving std for the volatility estimate.  Both are
    shifted by 1 to avoid lookahead.

    EMA reacts faster than SMA because recent observations have
    exponentially higher weight, but it lacks the Bayesian uncertainty
    tracking that lets the Kalman filter absorb jumps adaptively.
    """
    ratio = np.log(prices_a / prices_b)
    ema_mean = ratio.ewm(span=window, adjust=False).mean().shift(1)
    ema_std = ratio.ewm(span=window, adjust=False).std().shift(1)
    return (ratio - ema_mean) / ema_std


def _make_aggressive_kalman_params(
    prices_a: pd.Series,
    prices_b: pd.Series,
    window: int,
    q_multiplier: float = 5.0,
) -> KalmanParams:
    """Create KalmanParams with Q scaled up for faster drift tracking.

    Default auto-tune: Q = R / window.
    With q_multiplier=5: Q = 5 × R / window → recovery in 5–10 days.
    """
    ratio = np.log(prices_a / prices_b).values
    seg = ratio[:window]
    valid = seg[np.isfinite(seg)]
    r_auto = float(np.var(valid, ddof=1)) if len(valid) >= 2 else 1e-12
    q_tuned = r_auto / float(window) * q_multiplier
    return KalmanParams(process_variance=q_tuned, measurement_variance=r_auto)


# ---------------------------------------------------------------------------
# Tests — Kalman drift recovery (core requirement)
# ---------------------------------------------------------------------------

class TestKalmanDriftRecovery:
    """Verify that Kalman Z-score normalises faster than SMA after a jump.

    Auto-tuned Kalman (Q = R/window) recovers in ~10–20 days.
    With tuned Q (5× multiplier), recovery hits the 5–10 day target.
    """

    WINDOW = 60
    MAX_RECOVERY_DAYS_AUTO = 20  # auto-tuned Kalman
    MAX_RECOVERY_DAYS_TUNED = 10  # Q × 5 multiplier

    def test_zscore_spikes_on_jump_day(self):
        """On the jump day itself, |z_score| should be large (≥ 2)."""
        pa, pb, jump_idx = _make_cointegrated_pair_with_jump(seed=42)
        df = compute_zscore(pa, pb, window=self.WINDOW)
        jump_z = df["zscore"].iloc[jump_idx]
        assert np.isfinite(jump_z)
        assert abs(jump_z) >= 2.0, (
            f"Expected spike on jump day, got z={jump_z:.2f}"
        )

    def test_auto_tuned_recovery_within_20_days(self):
        """Auto-tuned Kalman: recovery within 20 days (default Q = R/window)."""
        pa, pb, jump_idx = _make_cointegrated_pair_with_jump(seed=42)
        df = compute_zscore(pa, pb, window=self.WINDOW)
        days = _days_until_zscore_normalises(df["zscore"], jump_idx)
        assert days is not None, "Z-score never normalised after jump"
        assert days <= self.MAX_RECOVERY_DAYS_AUTO, (
            f"Auto-tuned took {days} days (max {self.MAX_RECOVERY_DAYS_AUTO})"
        )

    def test_tuned_q_recovery_within_10_days(self):
        """With Q × 5 multiplier, recovery hits the 5–10 day target."""
        pa, pb, jump_idx = _make_cointegrated_pair_with_jump(seed=42)
        params = _make_aggressive_kalman_params(pa, pb, self.WINDOW, q_multiplier=5.0)
        df = compute_zscore(pa, pb, window=self.WINDOW, kalman_params=params)
        days = _days_until_zscore_normalises(df["zscore"], jump_idx)
        assert days is not None, "Z-score never normalised after jump"
        assert days <= self.MAX_RECOVERY_DAYS_TUNED, (
            f"Tuned Kalman took {days} days (max {self.MAX_RECOVERY_DAYS_TUNED})"
        )

    @pytest.mark.parametrize("jump_pct", [0.10, 0.15, 0.20, 0.30])
    def test_auto_recovery_across_jump_sizes(self, jump_pct: float):
        """Auto-tuned Kalman recovers within 20 days for various jump sizes."""
        pa, pb, jump_idx = _make_cointegrated_pair_with_jump(
            jump_pct=jump_pct, seed=42,
        )
        df = compute_zscore(pa, pb, window=self.WINDOW)
        days = _days_until_zscore_normalises(df["zscore"], jump_idx)
        assert days is not None, f"Never normalised for {jump_pct:.0%} jump"
        assert days <= self.MAX_RECOVERY_DAYS_AUTO, (
            f"{jump_pct:.0%} jump: auto recovery took {days} days"
        )

    @pytest.mark.parametrize("jump_pct", [0.10, 0.15, 0.20, 0.30])
    def test_tuned_recovery_across_jump_sizes(self, jump_pct: float):
        """With Q × 5, recovery ≤ 10 days for various jump sizes."""
        pa, pb, jump_idx = _make_cointegrated_pair_with_jump(
            jump_pct=jump_pct, seed=42,
        )
        params = _make_aggressive_kalman_params(pa, pb, self.WINDOW, q_multiplier=5.0)
        df = compute_zscore(pa, pb, window=self.WINDOW, kalman_params=params)
        days = _days_until_zscore_normalises(df["zscore"], jump_idx)
        assert days is not None, f"Never normalised for {jump_pct:.0%} jump"
        assert days <= self.MAX_RECOVERY_DAYS_TUNED, (
            f"{jump_pct:.0%} jump: tuned recovery took {days} days"
        )

    @pytest.mark.parametrize("seed", [42, 77, 123, 999, 2024])
    def test_auto_recovery_across_random_seeds(self, seed: int):
        """Consistent auto-tuned recovery across different noise realisations."""
        pa, pb, jump_idx = _make_cointegrated_pair_with_jump(seed=seed)
        df = compute_zscore(pa, pb, window=self.WINDOW)
        days = _days_until_zscore_normalises(df["zscore"], jump_idx)
        assert days is not None, f"Never normalised (seed={seed})"
        assert days <= self.MAX_RECOVERY_DAYS_AUTO, (
            f"seed={seed}: auto recovery took {days} days"
        )

    def test_negative_jump_recovery(self):
        """A -20 % crash should also recover within 20 days (auto-tuned)."""
        pa, pb, jump_idx = _make_cointegrated_pair_with_jump(jump_pct=-0.20)
        df = compute_zscore(pa, pb, window=self.WINDOW)
        days = _days_until_zscore_normalises(df["zscore"], jump_idx)
        assert days is not None, "Z-score never normalised after crash"
        assert days <= self.MAX_RECOVERY_DAYS_AUTO


# ---------------------------------------------------------------------------
# Tests — Kalman vs SMA comparison (Kalman must be strictly faster)
# ---------------------------------------------------------------------------

class TestKalmanVsSmaRecovery:
    """Kalman filter must recover from drift faster than SMA."""

    WINDOW = 60

    def test_kalman_beats_sma_on_20pct_jump(self):
        """Kalman recovers from +20 % jump strictly faster than SMA."""
        pa, pb, jump_idx = _make_cointegrated_pair_with_jump(seed=42)

        kalman_df = compute_zscore(pa, pb, window=self.WINDOW)
        kalman_days = _days_until_zscore_normalises(
            kalman_df["zscore"], jump_idx,
        )

        sma_zscore = _compute_sma_zscore(pa, pb, window=self.WINDOW)
        sma_days = _days_until_zscore_normalises(sma_zscore, jump_idx)

        assert kalman_days is not None, "Kalman never normalised"
        assert sma_days is not None, "SMA never normalised"
        assert kalman_days < sma_days, (
            f"Kalman ({kalman_days}d) should beat SMA ({sma_days}d)"
        )

    @pytest.mark.parametrize("seed", [42, 77, 123, 999, 2024])
    def test_kalman_beats_sma_across_seeds(self, seed: int):
        """Kalman consistently recovers faster than SMA across noise seeds."""
        pa, pb, jump_idx = _make_cointegrated_pair_with_jump(seed=seed)
        kalman_days = _days_until_zscore_normalises(
            compute_zscore(pa, pb, window=self.WINDOW)["zscore"], jump_idx,
        )
        sma_days = _days_until_zscore_normalises(
            _compute_sma_zscore(pa, pb, self.WINDOW), jump_idx,
        )
        assert kalman_days is not None and sma_days is not None
        assert kalman_days < sma_days, (
            f"seed={seed}: Kalman ({kalman_days}d) should beat SMA ({sma_days}d)"
        )

    def test_sma_ghost_persists_longer_than_kalman(self):
        """SMA takes at least 1.5× longer than Kalman to recover.

        With daily_vol=0.005 and a +20 % jump, SMA typically takes
        21–35 days vs Kalman's 10–15 days.
        """
        pa, pb, jump_idx = _make_cointegrated_pair_with_jump(seed=42)
        kalman_days = _days_until_zscore_normalises(
            compute_zscore(pa, pb, window=60)["zscore"], jump_idx,
        )
        sma_days = _days_until_zscore_normalises(
            _compute_sma_zscore(pa, pb, 60), jump_idx,
        )
        assert kalman_days is not None and sma_days is not None
        ratio = sma_days / kalman_days
        assert ratio >= 1.5, (
            f"SMA/Kalman ratio = {ratio:.1f}× (expected ≥ 1.5×)"
        )


# ---------------------------------------------------------------------------
# Tests — Kalman vs EMA comparison
# ---------------------------------------------------------------------------

class TestKalmanVsEmaRecovery:
    """EMA is a simpler adaptive alternative. Kalman should match or beat it."""

    WINDOW = 60

    def test_ema_recovers_faster_than_sma(self):
        """EMA should beat SMA (sanity check for the EMA baseline)."""
        pa, pb, jump_idx = _make_cointegrated_pair_with_jump(seed=42)
        ema_days = _days_until_zscore_normalises(
            _compute_ema_zscore(pa, pb, self.WINDOW), jump_idx,
        )
        sma_days = _days_until_zscore_normalises(
            _compute_sma_zscore(pa, pb, self.WINDOW), jump_idx,
        )
        assert ema_days is not None, "EMA never normalised"
        assert sma_days is not None, "SMA never normalised"
        assert ema_days < sma_days

    def test_kalman_recovers_no_slower_than_ema(self):
        """Auto-tuned Kalman should be comparable to or faster than EMA."""
        pa, pb, jump_idx = _make_cointegrated_pair_with_jump(seed=42)

        kalman_days = _days_until_zscore_normalises(
            compute_zscore(pa, pb, window=self.WINDOW)["zscore"], jump_idx,
        )
        ema_days = _days_until_zscore_normalises(
            _compute_ema_zscore(pa, pb, self.WINDOW), jump_idx,
        )

        assert kalman_days is not None
        assert ema_days is not None
        assert kalman_days <= ema_days, (
            f"Kalman ({kalman_days}d) should be ≤ EMA ({ema_days}d)"
        )

    def test_all_three_methods_summary(self):
        """Print recovery comparison for visual inspection (always passes)."""
        pa, pb, jump_idx = _make_cointegrated_pair_with_jump(seed=42)
        params_tuned = _make_aggressive_kalman_params(pa, pb, 60, q_multiplier=5.0)

        kalman_auto_days = _days_until_zscore_normalises(
            compute_zscore(pa, pb, window=60)["zscore"], jump_idx,
        )
        kalman_tuned_days = _days_until_zscore_normalises(
            compute_zscore(pa, pb, window=60, kalman_params=params_tuned)["zscore"],
            jump_idx,
        )
        ema_days = _days_until_zscore_normalises(
            _compute_ema_zscore(pa, pb, 60), jump_idx,
        )
        sma_days = _days_until_zscore_normalises(
            _compute_sma_zscore(pa, pb, 60), jump_idx,
        )
        print(
            f"\n{'Method':<20} {'Recovery (days)':>15}\n"
            f"{'─' * 36}\n"
            f"{'Kalman (Q×5)':<20} {str(kalman_tuned_days):>15}\n"
            f"{'Kalman (auto)':<20} {str(kalman_auto_days):>15}\n"
            f"{'EMA':<20} {str(ema_days):>15}\n"
            f"{'SMA':<20} {str(sma_days):>15}"
        )


# ---------------------------------------------------------------------------
# Tests — Kalman steady-state behaviour (no drift)
# ---------------------------------------------------------------------------

class TestKalmanSteadyState:
    """When there is no structural break, Z-scores should behave normally."""

    def test_zscore_mostly_within_2(self):
        """Under normal conditions, >90 % of Z-scores should be in [-2, 2]."""
        rng = np.random.default_rng(42)
        n = 500
        dates = pd.bdate_range("2020-01-01", periods=n)
        common = rng.normal(0, 0.005, n)

        pa = pd.Series(
            100 * np.exp(np.cumsum(rng.normal(0, 0.01, n) + common)),
            index=dates,
        )
        pb = pd.Series(
            100 * np.exp(np.cumsum(rng.normal(0, 0.01, n) + common)),
            index=dates,
        )
        df = compute_zscore(pa, pb, window=60)
        valid_z = df["zscore"].dropna()
        pct_within_2 = (valid_z.abs() <= 2.0).mean()
        assert pct_within_2 >= 0.90, (
            f"Only {pct_within_2:.1%} of Z-scores within ±2 (expected ≥ 90%)"
        )

    def test_zscore_mean_near_zero(self):
        """Mean Z-score should be approximately 0 under stationary conditions."""
        rng = np.random.default_rng(123)
        n = 1000
        dates = pd.bdate_range("2018-01-01", periods=n)
        common = rng.normal(0, 0.005, n)
        pa = pd.Series(
            100 * np.exp(np.cumsum(rng.normal(0, 0.01, n) + common)),
            index=dates,
        )
        pb = pd.Series(
            100 * np.exp(np.cumsum(rng.normal(0, 0.01, n) + common)),
            index=dates,
        )
        df = compute_zscore(pa, pb, window=60)
        valid_z = df["zscore"].dropna()
        assert abs(valid_z.mean()) < 0.5, (
            f"Mean Z-score = {valid_z.mean():.3f}, expected near 0"
        )


# ---------------------------------------------------------------------------
# Tests — KalmanParams custom override
# ---------------------------------------------------------------------------

class TestKalmanParamsOverride:
    """Verify that explicit Q/R overrides are respected."""

    def test_high_process_variance_tracks_faster(self):
        """Higher Q → faster tracking → quicker recovery after jump.

        Very low Q (1e-8) may never recover within 150 remaining bars,
        which is the expected behaviour (SMA-like sluggishness).
        """
        pa, pb, jump_idx = _make_cointegrated_pair_with_jump(seed=42)

        fast_params = KalmanParams(
            process_variance=1e-3, measurement_variance=1e-3,
        )
        fast_df = compute_zscore(pa, pb, window=60, kalman_params=fast_params)
        fast_days = _days_until_zscore_normalises(fast_df["zscore"], jump_idx)

        slow_params = KalmanParams(
            process_variance=1e-6, measurement_variance=1e-3,
        )
        slow_df = compute_zscore(pa, pb, window=60, kalman_params=slow_params)
        slow_days = _days_until_zscore_normalises(slow_df["zscore"], jump_idx)

        assert fast_days is not None
        # slow_days may be None (never recovers) — that's even stronger proof
        if slow_days is not None:
            assert fast_days <= slow_days, (
                f"High-Q ({fast_days}d) should recover ≤ low-Q ({slow_days}d)"
            )
        # If slow never recovers, fast must still recover
        assert fast_days <= 10

    def test_custom_params_output_shape_unchanged(self):
        """Custom KalmanParams should not change output shape or columns."""
        pa, pb, _ = _make_cointegrated_pair_with_jump()
        default_df = compute_zscore(pa, pb, window=60)
        custom_df = compute_zscore(
            pa, pb, window=60,
            kalman_params=KalmanParams(
                process_variance=1e-4, measurement_variance=1e-3,
            ),
        )
        assert list(default_df.columns) == list(custom_df.columns)
        assert len(default_df) == len(custom_df)

    def test_q_multiplier_monotonic(self):
        """Higher Q multiplier → equal or faster recovery (monotonic)."""
        pa, pb, jump_idx = _make_cointegrated_pair_with_jump(seed=42)
        prev_days = float("inf")
        for q_mult in [1, 2, 5, 10]:
            params = _make_aggressive_kalman_params(
                pa, pb, 60, q_multiplier=q_mult,
            )
            df = compute_zscore(pa, pb, window=60, kalman_params=params)
            days = _days_until_zscore_normalises(df["zscore"], jump_idx)
            assert days is not None
            assert days <= prev_days, (
                f"Q×{q_mult} recovery ({days}d) > Q×{q_mult-1} ({prev_days}d)"
            )
            prev_days = days


# ---------------------------------------------------------------------------
# Tests — Edge cases
# ---------------------------------------------------------------------------

class TestKalmanEdgeCases:
    """Edge cases: NaN handling, short series, identical prices."""

    def test_nan_in_prices_handled(self):
        """NaN values in input should not crash the filter."""
        pa, pb, _ = _make_cointegrated_pair_with_jump(n_days=200)
        pa.iloc[80] = np.nan
        pa.iloc[81] = np.nan
        df = compute_zscore(pa, pb, window=60)
        assert len(df) == 200
        assert "zscore" in df.columns

    def test_identical_prices_zero_zscore(self):
        """When A == B, ratio = log(1) = 0, Z-score should be ~0 or NaN."""
        n = 200
        dates = pd.bdate_range("2020-01-01", periods=n)
        pa = pd.Series(100.0 * np.ones(n), index=dates)
        pb = pd.Series(100.0 * np.ones(n), index=dates)
        df = compute_zscore(pa, pb, window=10)
        valid = df["zscore"].dropna()
        if len(valid) > 0:
            assert valid.abs().max() < 1e-6

    def test_short_series_no_crash(self):
        """Series shorter than window should not crash (all NaN Z-scores)."""
        n = 20
        dates = pd.bdate_range("2020-01-01", periods=n)
        pa = pd.Series(100.0 + np.arange(n) * 0.1, index=dates)
        pb = pd.Series(80.0 + np.arange(n) * 0.08, index=dates)
        df = compute_zscore(pa, pb, window=60)
        assert len(df) == n
        assert df["zscore"].isna().all()
