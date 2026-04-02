"""Unit tests for rolling Phase 2 portfolio simulation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pairs_eda.rolling_phase2 import (
    RollingPhase2Config,
    RollingPhase2Input,
    _CointCacheEntry,
    build_rolling_timeline,
    compute_robust_pair_scores,
    filter_cointegrated_cached,
    run_phase2_rolling,
)


def _make_daily_panel(n_days: int = 900, seed: int = 11) -> pd.DataFrame:
    """Synthetic daily Adj Close panel with two mean-reverting pairs.

    Prices are constructed via log-ratio mean reversion so that the
    z-score based strategy generates consistent positive returns in
    both train and validation windows (passing the consistency gate).
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2018-01-01", periods=n_days)

    t = np.arange(n_days, dtype=float)

    def _pair(base: float, drift: float, cycle_len: int, amp: float) -> tuple[np.ndarray, np.ndarray]:
        trend = np.log(base) + drift * t / 252.0
        spread = amp * np.sin(2 * np.pi * t / cycle_len)
        noise_a = rng.normal(0, 0.002, n_days)
        noise_b = rng.normal(0, 0.002, n_days)
        log_a = trend + spread / 2 + noise_a
        log_b = trend - spread / 2 + noise_b
        return np.exp(log_a), np.exp(log_b)

    a, b = _pair(100.0, 0.05, 40, 0.06)
    c, d = _pair(80.0, 0.04, 35, 0.05)

    prices = pd.DataFrame({"AAA": a, "BBB": b, "CCC": c, "DDD": d}, index=idx)
    return prices.sort_index()


def _base_config() -> RollingPhase2Config:
    return RollingPhase2Config(
        training_months=12,
        validation_days=42,
        windows=(15, 20, 30),
        zscore_thresholds=(1.2, 1.5, 2.0),
        watchlist_size=2,
        max_slots=1,
        top_n_candidates=20,
        min_overlap_pct=0.5,
    )


class TestRollingTimeline:
    def test_builds_monthly_schedule_with_daily_phase2_ranges(self):
        prices = _make_daily_panel()
        cfg = _base_config()
        inp = RollingPhase2Input(prices=prices, initial_capital=7000.0, config=cfg)

        schedule = build_rolling_timeline(inp)
        assert len(schedule) > 0
        assert all(w.phase1_start < w.phase1_end for w in schedule)
        assert all(w.phase2_start <= w.phase2_end for w in schedule)
        assert all(w.phase2_start.day == 1 for w in schedule)


class TestRobustScoring:
    def test_computes_robustness_columns(self):
        """Scored df has correct columns; may be empty if consistency gate rejects all."""
        prices = _make_daily_panel()
        cfg = _base_config()
        train = prices.iloc[:500]

        scored, coint_stats = compute_robust_pair_scores(
            train,
            cfg,
            pair_universe=[("AAA", "BBB"), ("CCC", "DDD")],
        )
        expected_cols = {
            "pair_key",
            "window",
            "entry_zscore",
            "dist_window",
            "validation_margin",
            "diff_margin",
            "base_score",
            "final_score",
        }
        assert expected_cols.issubset(set(scored.columns))

    def test_consistency_gate_rejects_negative_validation(self):
        """Pairs where validation margin < 0 are hard-rejected. 0 is allowed if no trades."""
        from pairs_eda.rolling_phase2 import _evaluate_pair_surface

        prices = _make_daily_panel()
        cfg = _base_config()
        train = prices.iloc[:500]
        result = _evaluate_pair_surface(("AAA", "BBB"), train, cfg)
        if result is not None:
            assert result["train_margin"] > 0.0
            assert result["validation_margin"] >= 0.0


class TestRollingStateMachine:
    def test_respects_slot_limit_and_outputs_metrics(self):
        prices = _make_daily_panel()
        cfg = _base_config()
        inp = RollingPhase2Input(
            prices=prices,
            initial_capital=7000.0,
            config=cfg,
            pair_universe=[("AAA", "BBB"), ("CCC", "DDD")],
        )
        out = run_phase2_rolling(inp)

        assert len(out.daily_equity) > 0
        assert out.summary["max_open_slots"] <= float(cfg.max_slots)
        assert "slot_utilization" in out.summary
        assert "sharpe_ratio" in out.summary
        assert "max_drawdown" in out.summary
        assert "turnover_per_year" in out.summary
        assert "twr_total" in out.summary
        assert "coint_cache_hit_rate" in out.summary


class TestCointCaching:
    """Tests for cointegration cache logic."""

    def test_filter_uses_cache_for_deep_pass(self):
        prices = _make_daily_panel()
        cache: dict[str, _CointCacheEntry] = {
            "AAA|BBB": _CointCacheEntry(p_value=0.001, passed=True, streak=1, tested_at="t1"),
        }
        passed, _, stats = filter_cointegrated_cached(
            [("AAA", "BBB")],
            prices,
            cache,
            significance=0.05,
        )
        assert ("AAA", "BBB") in passed
        assert stats["cache_hit"] == 1
        assert stats["tested"] == 0

    def test_filter_uses_cache_for_deep_fail(self):
        prices = _make_daily_panel()
        cache: dict[str, _CointCacheEntry] = {
            "AAA|BBB": _CointCacheEntry(p_value=0.50, passed=False, streak=1, tested_at="t1"),
        }
        passed, _, stats = filter_cointegrated_cached(
            [("AAA", "BBB")],
            prices,
            cache,
            significance=0.05,
        )
        assert ("AAA", "BBB") not in passed
        assert stats["cache_hit"] == 1
        assert stats["tested"] == 0

    def test_filter_retests_borderline_pair(self):
        prices = _make_daily_panel()
        cache: dict[str, _CointCacheEntry] = {
            "AAA|BBB": _CointCacheEntry(p_value=0.045, passed=True, streak=1, tested_at="t1"),
        }
        _, updated_cache, stats = filter_cointegrated_cached(
            [("AAA", "BBB")],
            prices,
            cache,
            significance=0.05,
        )
        assert stats["tested"] == 1
        assert stats["cache_hit"] == 0
        assert "AAA|BBB" in updated_cache
        assert updated_cache["AAA|BBB"].streak == 1

    def test_filter_retests_old_pair(self):
        prices = _make_daily_panel()
        cache: dict[str, _CointCacheEntry] = {
            "AAA|BBB": _CointCacheEntry(p_value=0.001, passed=True, streak=6, tested_at="t1"),
        }
        _, updated_cache, stats = filter_cointegrated_cached(
            [("AAA", "BBB")],
            prices,
            cache,
            significance=0.05,
            max_streak=6,
        )
        assert stats["tested"] == 1
        assert stats["cache_hit"] == 0
        assert "AAA|BBB" in updated_cache
        assert updated_cache["AAA|BBB"].streak == 1

    def test_filter_evicts_missing_pairs(self):
        prices = _make_daily_panel()
        cache: dict[str, _CointCacheEntry] = {
            "AAA|BBB": _CointCacheEntry(p_value=0.001, passed=True, streak=1, tested_at="t1"),
            "CCC|DDD": _CointCacheEntry(p_value=0.001, passed=True, streak=1, tested_at="t1"),
        }
        # Only AAA|BBB is in the input pairs
        passed, updated_cache, stats = filter_cointegrated_cached(
            [("AAA", "BBB")],
            prices,
            cache,
            significance=0.05,
        )
        assert "AAA|BBB" in updated_cache
        assert "CCC|DDD" not in updated_cache

    def test_filter_tests_new_pair(self):
        prices = _make_daily_panel()
        cache: dict[str, _CointCacheEntry] = {}
        _, updated_cache, stats = filter_cointegrated_cached(
            [("AAA", "BBB")],
            prices,
            cache,
            significance=0.05,
        )
        assert stats["tested"] == 1
        assert "AAA|BBB" in updated_cache

    def test_filter_skips_missing_ticker(self):
        prices = _make_daily_panel()
        cache: dict[str, _CointCacheEntry] = {}
        passed, _, stats = filter_cointegrated_cached(
            [("AAA", "MISSING")],
            prices,
            cache,
            significance=0.05,
        )
        assert len(passed) == 0
        assert stats["tested"] == 0

