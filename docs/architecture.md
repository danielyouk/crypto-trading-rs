# Pairs Trading System — Architecture & Analysis

## 1. System Overview

```
crypto-trading-rs/
├── python/pairs_eda/          # Core library (pip-installable package)
│   ├── rolling_phase2.py      # WFA engine — the heart of the system (1124 lines)
│   ├── backtesting.py         # Single-pair pipeline & grid search (725 lines)
│   ├── correlation.py         # Pair universe filtering (364 lines)
│   ├── sp500.py               # S&P 500 constituent/sector fetching (402 lines)
│   ├── vectorized_backtest.py # NumPy-vectorized backtest (325 lines)
│   ├── yfinance_tools.py      # Data download with retry + Adj Close (213 lines)
│   ├── display.py             # Pretty-print helpers for notebook (181 lines)
│   ├── visualization.py       # Correlation histogram (76 lines)
│   ├── gemini_search.py       # LLM fallback for S&P 500 list (211 lines)
│   ├── exa_fallback.py        # Search API fallback (90 lines)
│   └── __init__.py            # 42 public symbols exported (100 lines)
│
├── python/tests/              # Unit tests (85 tests, 4 modules)
│   ├── test_backtesting.py    # Single-pair pipeline tests (484 lines)
│   ├── test_correlation.py    # Pair filtering tests (314 lines)
│   ├── test_rolling_phase2.py # WFA engine tests (242 lines)
│   └── test_yfinance_tools.py # Data download tests (249 lines)
│
├── reference/python_pairstrading/
│   └── stock-trading-eda-scheduled_eng.ipynb   # Main notebook (88 cells)
│
├── docs/
│   ├── architecture.md        # This file
│   ├── lecture-ideas.md       # Course content & teaching notes (1057 lines)
│   ├── pipeline-backlog.md    # Feature backlog & design decisions
│   ├── wfa-tuning-log.md      # Tuning rationale & experiment template
│   ├── wfa-run-log.csv        # General WFA run log
│   ├── wfa-sensitivity-log.csv # Parameter sweep results
│   ├── wfa-selection-log.csv  # Robust spot selection log
│   └── wfa-holdout-log.csv    # Final holdout evaluation log
│
└── .cursor/rules/             # 10 Cursor rules (commit, refactor, quant commands, etc.)
```

**Total library code**: ~3,700 lines Python  
**Total test code**: ~1,290 lines Python  
**Test count**: 85 tests, all passing

---

## 2. Data Flow Architecture

```
                        ┌─────────────────────────────┐
                        │   S&P 500 Constituents       │
                        │   (Wikipedia + LLM fallback) │
                        └──────────────┬──────────────┘
                                       │
                                       ▼
                        ┌─────────────────────────────┐
                        │   yfinance Daily Prices      │
                        │   (Adj Close, with retry)    │
                        └──────────────┬──────────────┘
                                       │
                          ┌────────────┴────────────┐
                          │                         │
                          ▼                         ▼
                ┌─────────────────┐     ┌─────────────────────────┐
                │  EDA / Demo     │     │  Walk-Forward Analysis   │
                │  (single pair)  │     │  (the real backtest)     │
                └─────────────────┘     └────────────┬────────────┘
                                                     │
                                        ┌────────────┼────────────┐
                                        │            │            │
                                        ▼            ▼            ▼
                                   Sensitivity    Robust Spot   Holdout
                                     Sweep        Selection    Evaluation
                                        │            │            │
                                        ▼            ▼            ▼
                                   CSV Logs      CSV Logs      CSV Logs
```

---

## 3. WFA Engine — Core Loop (`rolling_phase2.py`)

This is the most complex and important module. Here is its internal flow:

```
build_rolling_timeline()
    │
    ▼
For each rebalance window:
    │
    ├── 1. filter_volatile_tickers()      ← per-rebalance, no look-ahead
    ├── 2. find_candidate_pairs()         ← correlation band [0.40, 0.85]
    ├── 3. filter_cointegrated_cached()   ← Engle-Granger + smart cache
    ├── 4. compute_robust_pair_scores()   ← grid search over (window × zscore)
    │       ├── compute_zscore()          ← Kalman Filter (handles structural breaks)
    │       └── _evaluate_pair_surface()  ← train/validation split
    │             ├── train_per_trade > 0? ← hard gate (past profit)
    │             ├── val_per_trade > 0?   ← hard gate (recent profit)
    │             ├── val_per_trade < 5x?  ← hard gate (reject luck/structural break)
    │             └── stable region median → best_window, best_z
    │
    ▼
Daily simulation loop (sim_dates):
    │
    ├── Check exits:
    │     ├── mean_reversion (z crosses exit_threshold) ← adaptive exit, blocked by min_holding_days
    │     └── stop_loss (5% of slot_notional) ← always allowed
    │
    ├── Circuit breaker check:
    │     └── total_equity dropped > cb_pct from peak → close all, 5-day cooldown
    │
    ├── New entries:
    │     ├── max_new_entries_per_day cap
    │     ├── min_entry_score gate
    │     ├── sector diversification (max_sector_slots)
    │     ├── min_spread_range_pct gate
    │     ├── max_zscore gate (reject extreme jumps > 5.0 as structural breaks)
    │     └── z-score trigger → open position
    │
    └── Record equity (realized + unrealized)
```

### Key Config Parameters (current active)

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `training_months` | 36 | Rolling window size |
| `validation_days` | 180 | Phase 2a consistency check (~6 months) |
| `rebalance_frequency` | "MS" | Monthly rebalance |
| `leverage` | 3.0 | Position sizing multiplier |
| `max_slots` | 7 | Max concurrent positions |
| `max_new_entries_per_day` | 2 | Smooths equity fluctuation |
| `min_holding_days` | 3 | Prevents ultra-fast churn |
| `circuit_breaker_pct` | 0.12 | Portfolio-level tail risk defense |
| `max_drop_quantile` | 0.90 | Drop worst-10% volatile tickers per rebalance |
| `entry_zscore_default` | 2.0 | Default entry threshold |
| `exit_zscore` | 0.0 | Exit when z-score reverts to 0 |
| `max_zscore` | 5.0 | Rejects extreme jumps as structural breaks |
| `stop_loss_pct` | 0.05 | Per-trade loss limit |
| `min_entry_score` | 0.50 | Quality gate |
| `max_sector_slots` | 2 | Sector concentration limit |

---

## 4. Hybrid Strategy — Macro Regime Switching

Pairs trading is a bear-market hedge, not a growth engine. The hybrid
strategy holds the S&P 500 in normal markets and switches to Pairs
Trading only during corrections:

```
            S&P 500 drawdown from peak
            │
            │  Normal: hold S&P 500
            │
    ────────┼──────── EXIT_DD = -5% ─── (switch BACK to S&P 500)
            │
            │  Bear: run Pairs Trading
            │
    ────────┼──────── ENTRY_DD = -10% ── (switch TO Pairs Trading)
            │
            ▼ deeper drawdown
```

**Rules:**
- When S&P 500 drawdown hits **-10%**, sell S&P 500 and switch 100% to Pairs Trading.
- When drawdown recovers to **-5%** (having passed through -10%), switch back to S&P 500.
- The -5% exit is **conditional** on already being in bear mode — it's never triggered in a normal bull market.
- On switch day, the WFA engine runs immediately on existing historical data (no warm-up delay).

**On-demand WFA architecture (`run_hybrid_backtest`):**

The WFA engine is expensive (176 grid combos × 200 pairs × monthly rebalances).
Since pairs trading is only active ~10-15% of the time, we avoid running WFA
over the full 20+ year history:

```
Step 1: find_bear_episodes()        ← scan S&P 500 drawdown (milliseconds)
        → [(2000-09, 2003-05), (2007-11, 2009-08), (2020-02, 2020-05), ...]

Step 2: For each bear episode ONLY:
        → run_phase2_rolling()      ← expensive, but only for 2-3 year windows

Step 3: Bull periods between:
        → compound S&P 500 returns  ← zero computation

Step 4: Stitch equity curves
```

This cuts WFA computation by **~85-90%** compared to running it full-period.

---

## 5. Notebook Flow (Active Path)

The notebook has 88 cells, but the **active execution path** is:

| Step | Cells | What it does |
|------|-------|-------------|
| Setup | 0-5 | Imports, S&P 500 list, sector map, download prices |
| EDA | 10-11 | Correlation histogram, interactive pair chart |
| Demo | 17-28 | Single-pair pipeline walkthrough (educational) |
| **WFA Config** | **72** | Define `RollingPhase2Config` + `RollingPhase2Input` |
| **WFA Run** | **74** | Full-period `run_phase2_rolling()` |
| **Hybrid** | **76** | S&P 500 + Pairs hedge equity curve (regime switching) |
| Diagnostics | 78-83 | Period table, heatmap, filter impact |
| Trade/Rebalance | 85-87 | Trade stats, rebalance history, cointegration cache |

Cells 32-70 are **legacy static backtest stubs** (all commented out, superseded by WFA).

---

## 6. Code Strengths

### Architecture
- **Clean separation**: library (`pairs_eda/`) vs. notebook vs. docs vs. rules
- **Pydantic models** for all config/input/output — type-safe, self-documenting, serializable
- **42 public symbols** cleanly exported through `__init__.py`
- **85 unit tests** covering backtesting, correlation, WFA, and data download
- **Per-iteration CSV logging** — experiments are reproducible and resumable
- **Cointegration caching** — reduces redundant computation by ~80-90%

### Strategy Design
- **Walk-Forward Analysis** eliminates look-ahead bias (train → validate → simulate)
- **Multi-layer risk control**: per-trade stop-loss, min holding days, circuit breaker, sector limits, entry quality gate
- **Sensitivity analysis framework** with robust-spot selection (not just picking the best point)
- **Holdout protocol** — final 2 years reserved for blind evaluation
- **Returns correlation** (not price correlation) — statistically correct
- **Adj Close** used consistently — accounts for splits/dividends

### Development Workflow
- **10 Cursor rules** enforce consistency (commit style, refactor rules, bilingual response, etc.)
- **Lecture-driven development** — `docs/lecture-ideas.md` captures teaching insights alongside code
- **Experiment logging** at 4 levels: run, sensitivity, selection, holdout

---

## 7. Code Weaknesses & Technical Debt

### Architecture
- **Notebook is too large** (88 cells, ~6500 lines with outputs). Hard to navigate. Cells 32-70 are dead code that should be removed entirely.
- **`rolling_phase2.py` is monolithic** (1124 lines). The daily simulation loop alone is ~300 lines. Should be split into: config, timeline, scoring, simulation, output.
- **No CLI / script entry point** — everything runs through notebook. Cannot run overnight sweeps headlessly (e.g., `python -m pairs_eda.sweep --config config.yaml`).
- **Helper functions defined inside notebook cells** (`append_wfa_run_log`, `run_local_sensitivity_sweep`, `select_robust_spot`, `evaluate_holdout_with_row`) should be in the library, not the notebook.
- **No type stubs for pandas** — many Pyright workarounds (`pd.DataFrame(...)` wrapping, `.to_dict()` pattern). Fragile.
- **`vectorized_backtest.py` vs `backtesting.py`** — two parallel backtest implementations. Confusing.

### Performance
- **Sensitivity sweep is extremely slow** (~15+ hours for 27 configs). Root cause: each `run_phase2_rolling` call rebuilds everything from scratch. No shared cointegration cache across sweep iterations.
- **Sequential sweep** — `run_local_sensitivity_sweep` runs configs one-by-one. Could use `joblib` or `multiprocessing`.
- **Feature cache rebuilt per pair** — `_create_feature_cache` is called repeatedly for the same pair across days.

### Testing
- **No integration test** for the full WFA pipeline end-to-end on synthetic data.
- **No test for sensitivity sweep / robust selection / holdout** functions.
- **No test for notebook helper functions** (`append_wfa_run_log` etc.).
- **Test data is synthetic** — no regression tests against known historical results.

### Logging / Observability
- **No progress callback** in `run_phase2_rolling` — cannot monitor 15-hour runs from outside.
- **No elapsed-time logging per rebalance** — hard to identify bottlenecks.
- **CSV logging path is relative** — breaks if notebook working directory changes.

---

## 8. Investment Strategy Strengths

- **Market-neutral by design** — long/short pairs hedge out market beta
- **Grounded in statistics** — cointegration + z-score mean-reversion is well-established
- **Multi-layer defense** — stop-loss (5%), circuit breaker (12%), sector limits, entry quality gate, and extreme jump rejection (`max_zscore`).
- **Adaptive Regime Handling** — Kalman Filter quickly resets the baseline after structural breaks, preventing the system from being trapped by "ghost signals".
- **Robust parameter selection** — not chasing the single best point, but the stable center of a high-performance region
- **Walk-forward validation** — avoids the classic trap of in-sample overfitting
- **Capital compounding** — slot notional recalculated each rebalance based on current equity
- **Historical performance** — ~18% CAGR, Sharpe ~1.3, across 30+ years (subject to overfitting caveat)

---

## 9. Investment Strategy Weaknesses

### Structural Limitations
- **Survivorship bias** — uses current S&P 500 constituents for the entire historical period. Delisted/removed tickers are missing from the universe.
- **Daily granularity** — stop-loss checked once per day. Intraday gaps can cause realized losses > 5%.
- **Equal-weight S&P 500 proxy** — not a true S&P 500 benchmark (cap-weighted).

### Risk Concerns
- **Max drawdown still high** (~22-35% depending on config) — uncomfortable for a "market-neutral" strategy.
- **Leverage amplifies tail risk** — 3x leverage means a 12% portfolio drawdown is effectively a 36% gross-exposure shock.
- **Correlation breakdown in crises** — COVID 2020 showed that all pairs can move together during extreme events, breaking the "market-neutral" assumption.
- **Stop-loss whipsaw** — pairs that trigger stop-loss and re-enter quickly can compound losses.
- **Low recent returns** (2020-2024 period underperforms vs S&P 500 buy-and-hold in a strong bull market).

### Overfitting Risk
- **Many tunable parameters** (training_months, validation_days, circuit_breaker, min_entry_score, max_sector_slots, min_spread_range, min_holding_days, max_new_entries_per_day, windows, zscore_thresholds...) — high-dimensional optimization surface.
- **Iterative manual tuning** — each run's results influenced the next parameter choice. Even with holdout, the "which holdout period to use" is itself a choice.
- **Exceptional historical performance** (~18% CAGR, Sharpe 1.3) should be treated with skepticism until confirmed out-of-sample in live trading.

### Operational Gaps (for live trading)
- **No real-time execution engine** — the system is backtest-only.
- **No order management** — no IBKR/TWS integration yet.
- **No position reconciliation** — no way to verify simulated vs actual positions.
- **No earnings blackout** — trades may open around earnings dates (high-risk).
- **No handling of halted/delisted tickers** in real-time.

---

## 10. Recommended Next Steps (Priority Order)

1. **Clean up notebook** — remove 40 dead cells (32-70), move helper functions to library
2. **Split `rolling_phase2.py`** — config, timeline, scoring, simulation as separate modules
3. **Add progress callback** to `run_phase2_rolling` — enable monitoring of long runs
4. **Share cointegration cache across sweep iterations** — could reduce sweep time from 15h to 2-3h
5. **Add CLI entry point** — enable headless overnight runs
6. **Holdout blind evaluation** — run once, record, and freeze
7. **Live trading bridge** — IBKR TWS API integration (separate module)
