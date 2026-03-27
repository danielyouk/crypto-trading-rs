"""Unit tests for rolling Phase 2 portfolio simulation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pairs_eda.rolling_phase2 import (
    RollingPhase2Config,
    RollingPhase2Input,
    _CointCacheEntry,
    _PairStickinessState,
    _should_retest,
    apply_sticky_watchlist,
    build_rolling_timeline,
    compute_robust_pair_scores,
    filter_cointegrated_cached,
    run_phase2_rolling,
)


def _make_daily_panel(n_days: int = 900, seed: int = 11) -> pd.DataFrame:
    """Synthetic daily Adj Close panel with two correlated pairs."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2018-01-01", periods=n_days)

    base_1 = np.cumsum(rng.normal(0, 0.6, n_days)) + 100.0
    base_2 = np.cumsum(rng.normal(0, 0.5, n_days)) + 80.0

    a = pd.Series(base_1 + rng.normal(0, 0.3, n_days), index=idx)
    b = pd.Series(base_1 + rng.normal(0, 0.35, n_days), index=idx)
    c = pd.Series(base_2 + rng.normal(0, 0.3, n_days), index=idx)
    d = pd.Series(base_2 + rng.normal(0, 0.35, n_days), index=idx)

    prices = pd.DataFrame(
        {
            "AAA": np.exp(np.log(np.maximum(a, 1.0)) / 4.6),
            "BBB": np.exp(np.log(np.maximum(b, 1.0)) / 4.6),
            "CCC": np.exp(np.log(np.maximum(c, 1.0)) / 4.4),
            "DDD": np.exp(np.log(np.maximum(d, 1.0)) / 4.4),
        },
        index=idx,
    )
    return prices.sort_index()


def _base_config() -> RollingPhase2Config:
    return RollingPhase2Config(
        training_months=12,
        validation_days=42,
        windows=(15, 20, 30),
        zscore_thresholds=(1.2, 1.5, 2.0),
        watchlist_size=2,
        watchlist_retention_buffer=1,
        max_slots=1,
        top_n_candidates=20,
        min_overlap_years=0.5,
        recent_years=0.5,
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


class TestRobustScoringAndStickiness:
    def test_computes_robustness_columns(self):
        prices = _make_daily_panel()
        cfg = _base_config()
        train = prices.iloc[:500]
        state: dict[str, _PairStickinessState] = {}

        scored, coint_stats = compute_robust_pair_scores(
            train,
            cfg,
            state,
            pair_universe=[("AAA", "BBB"), ("CCC", "DDD")],
        )
        assert not scored.empty
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
        assert "tested" in coint_stats

    def test_sticky_watchlist_retains_buffer_pair(self):
        cfg = _base_config()
        state = {"AAA|BBB": _PairStickinessState(keep_streak=2, drop_streak=0)}
        previous_watchlist = {"AAA|BBB"}

        scored_pairs = pd.DataFrame(
            [
                {"pair_key": "CCC|DDD", "final_score": 1.00},
                {"pair_key": "AAA|BBB", "final_score": 0.95},
                {"pair_key": "EEE|FFF", "final_score": 0.80},
            ]
        )
        active, next_state = apply_sticky_watchlist(scored_pairs, previous_watchlist, state, cfg)
        assert "AAA|BBB" in set(active["pair_key"].tolist())
        assert next_state["AAA|BBB"].keep_streak >= 1


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

    def test_should_retest_borderline_pair(self):
        entry = _CointCacheEntry(p_value=0.042, passed=True, tested_at="2020-Q1")
        assert _should_retest(entry, significance=0.05) is True

    def test_should_not_retest_deep_pass(self):
        entry = _CointCacheEntry(p_value=0.001, passed=True, tested_at="2020-Q1")
        assert _should_retest(entry, significance=0.05) is False

    def test_should_not_retest_deep_fail(self):
        entry = _CointCacheEntry(p_value=0.30, passed=False, tested_at="2020-Q1")
        assert _should_retest(entry, significance=0.05) is False

    def test_should_retest_borderline_fail(self):
        entry = _CointCacheEntry(p_value=0.058, passed=False, tested_at="2020-Q1")
        assert _should_retest(entry, significance=0.05) is True

    def test_filter_uses_cache_for_deep_pass(self):
        prices = _make_daily_panel()
        cache: dict[str, _CointCacheEntry] = {
            "AAA|BBB": _CointCacheEntry(p_value=0.001, passed=True, tested_at="t1"),
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
            "AAA|BBB": _CointCacheEntry(p_value=0.50, passed=False, tested_at="t1"),
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
            "AAA|BBB": _CointCacheEntry(p_value=0.045, passed=True, tested_at="t1"),
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

