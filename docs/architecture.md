# Pairs Trading System — Architecture & Analysis

> **Last updated**: April 2026 — includes Hybrid Strategy, Point-in-Time (PIT)
> survivorship bias fix, EODHD delisted data integration, and Streamlit dashboards.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Repository Structure](#2-repository-structure)
3. [Data Pipeline](#3-data-pipeline)
4. [Core Library (`pairs_eda/`)](#4-core-library-pairs_eda)
5. [WFA Engine — `rolling_phase2.py`](#5-wfa-engine--rolling_phase2py)
6. [Kalman Filter Z-Score](#6-kalman-filter-z-score)
7. [Pair Scoring & Robustness](#7-pair-scoring--robustness)
8. [Daily Portfolio Simulation](#8-daily-portfolio-simulation)
9. [Hybrid Strategy — Macro Regime Switching](#9-hybrid-strategy--macro-regime-switching)
10. [Point-in-Time (PIT) Survivorship Bias Fix](#10-point-in-time-pit-survivorship-bias-fix)
11. [Runners & Dashboards](#11-runners--dashboards)
12. [Configuration Reference](#12-configuration-reference)
13. [Code Strengths](#13-code-strengths)
14. [Code Weaknesses & Technical Debt](#14-code-weaknesses--technical-debt)
15. [Investment Strategy Strengths](#15-investment-strategy-strengths)
16. [Investment Strategy Weaknesses](#16-investment-strategy-weaknesses)
17. [Recommended Next Steps](#17-recommended-next-steps)

---

## 1. System Overview

This is a **statistical pairs trading** system built on Walk-Forward Analysis (WFA).
The core idea: find pairs of stocks whose price ratio is *cointegrated* (statistically
linked), compute a Kalman-filtered z-score, trade when the spread deviates, and exit
when it reverts.

The system operates in three modes:

| Mode | Description | Entry Point |
|------|-------------|-------------|
| **Full Pairs** | All-period pairs trading | `run_pairs_only.py` |
| **Hybrid** | S&P 500 in bull, Pairs in bear | `run_wfa.py` |
| **PIT Pairs** | Survivorship-bias-free pairs | `run_pairs_pit.py` |

**Total library code**: ~4,500 lines Python (13 modules)
**Total test code**: ~1,290 lines Python (4 test modules, 85 tests)

---

## 2. Repository Structure

```
crypto-trading-rs/
│
├── python/pairs_eda/                    # ── Core Library ──────────────────
│   ├── __init__.py                      #   Public API exports (42 symbols)
│   ├── rolling_phase2.py          1644  #   WFA engine — THE heart of the system
│   ├── backtesting.py              975  #   Single-pair pipeline, Kalman filter, grid search
│   ├── correlation.py              544  #   Volatility filter, return correlation, cointegration
│   ├── sp500.py                    402  #   S&P 500 constituent/sector fetching (Wikipedia + Gemini)
│   ├── sp500_history.py            220  #   Point-in-time membership (hanshof dataset)
│   ├── eodhd_download.py           100  #   EODHD API for delisted stock prices
│   ├── yfinance_tools.py           213  #   Download retry + Adj Close panel selection
│   ├── vectorized_backtest.py      325  #   NumPy-vectorized backtest (legacy, for EDA)
│   ├── display.py                  181  #   Pretty-print helpers for notebook
│   ├── visualization.py             76  #   Correlation histogram (matplotlib)
│   ├── gemini_search.py            211  #   LLM fallback for S&P 500 list
│   └── exa_fallback.py              90  #   Search API protocol/backends
│
├── python/tests/                        # ── Unit Tests ────────────────────
│   ├── test_backtesting.py         484  #   Single-pair pipeline tests
│   ├── test_correlation.py         314  #   Pair filtering tests
│   ├── test_rolling_phase2.py      242  #   WFA engine tests
│   └── test_yfinance_tools.py      249  #   Data download tests
│
├── reference/python_pairstrading/       # ── Runners & Dashboards ──────────
│   ├── run_wfa.py                       #   Hybrid WFA runner (S&P 500 + Pairs)
│   ├── run_pairs_only.py                #   Full pairs-only WFA runner
│   ├── run_pairs_pit.py                 #   PIT pairs WFA runner (bias-free)
│   ├── wfa_dashboard.py                 #   Streamlit: hybrid WFA (port 8501)
│   ├── pairs_dashboard.py               #   Streamlit: full pairs (port 8502)
│   ├── pairs_pit_dashboard.py           #   Streamlit: PIT comparison (port 8503)
│   ├── run_all.sh                       #   tmux launcher: hybrid
│   ├── run_pairs_only.sh                #   tmux launcher: full pairs
│   ├── run_pairs_pit.sh                 #   tmux launcher: PIT pairs
│   └── stock-trading-eda-scheduled_eng.ipynb  # Main notebook (88 cells)
│
├── scripts/                             # ── Operational Scripts ───────────
│   └── download_eodhd_missing.py        #   EODHD gap-fill for delisted tickers
│
├── data/                                # ── Cached Data (gitignored) ──────
│   ├── sp500_historical_components.csv  #   hanshof daily S&P 500 membership
│   ├── sp500_all_prices.parquet         #   Price panel: 1,108 tickers × 9,135 days (36 MB)
│   ├── eodhd_download_report.json       #   EODHD download report
│   └── missing_tickers.txt              #   Tickers still missing after download
│
├── docs/                                # ── Documentation ─────────────────
│   ├── architecture.md                  #   THIS FILE
│   ├── lecture-ideas.md                 #   Course content & teaching notes
│   ├── pipeline-backlog.md              #   Feature backlog & design decisions
│   ├── wfa-tuning-log.md               #   Tuning rationale
│   ├── wfa-*.csv                        #   Experiment logs (sensitivity, selection, holdout)
│   └── pairs-*.json / wfa-*.json        #   Live progress files for dashboards
│
├── src/                                 # ── Rust Crate (crypto trading) ───
│   ├── main.rs                          #   CLI: Stream, Backtest, Trade
│   ├── strategy/pairs.rs                #   Rust pairs z-spread strategy
│   └── ...                              #   config, data, execution, risk, monitoring
│
└── .cursor/                             # ── Cursor AI Configuration ───────
    ├── rules/                           #   10 rules (commit, bilingual, quant, etc.)
    ├── commands/                         #   /quant-coding, /quant-brainstorm, /quant-refactor
    ├── agents/                           #   quant-perf-optimizer, quant-risk-manager, etc.
    └── skills/pairs-trading/             #   SKILL.md + reference docs
```

---

## 3. Data Pipeline

### Data Sources

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           DATA ACQUISITION                              │
│                                                                         │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────────┐  │
│  │    Wikipedia      │  │  hanshof/sp500_  │  │    EODHD API         │  │
│  │  S&P 500 table    │  │  constituents    │  │  (paid, $19.99/mo)   │  │
│  │  (current list)   │  │  (1996-present)  │  │  delisted stock data │  │
│  └────────┬─────────┘  └────────┬─────────┘  └──────────┬───────────┘  │
│           │                     │                        │              │
│           ▼                     ▼                        ▼              │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────────┐  │
│  │  sp500.py         │  │  sp500_history.py│  │  eodhd_download.py   │  │
│  │  fetch_sp500_     │  │  Sp500History    │  │  download_missing_   │  │
│  │  constituents_    │  │  .from_csv()     │  │  from_eodhd()        │  │
│  │  table()          │  │  .members_as_of()│  │                      │  │
│  └────────┬─────────┘  └────────┬─────────┘  └──────────┬───────────┘  │
│           │                     │                        │              │
│           ▼                     └────────────┬───────────┘              │
│  ┌──────────────────┐                        │                          │
│  │  Yahoo Finance   │                        ▼                          │
│  │  (yfinance_tools)│            ┌──────────────────────┐              │
│  │  download_with_  │            │  sp500_all_prices    │              │
│  │  retry()         │            │  .parquet            │              │
│  └────────┬─────────┘            │  1,108 tickers       │              │
│           │                      │  9,135 trading days   │              │
│           ▼                      │  36 MB (disk cache)   │              │
│  ┌──────────────────┐            └──────────────────────┘              │
│  │  adj_close_or_   │                        │                          │
│  │  close_panel()   │ ◄──────────────────────┘                          │
│  │  → prices DF     │                                                   │
│  └──────────────────┘                                                   │
└─────────────────────────────────────────────────────────────────────────┘
```

### Data Coverage

| Source | Tickers | Date Range | Notes |
|--------|---------|------------|-------|
| Yahoo Finance | 785 | 1990–2026 | Current S&P 500 + some historical |
| EODHD (paid) | 323 | 1990–2026 | Delisted/bankrupt tickers |
| Combined cache | **1,108** | 1990–2026 | **98.4% coverage** of all S&P 500 members |
| Still missing | 18 | — | Very old pre-1996 companies, no digital records |

Key bankruptcies now included:
- **Lehman Brothers** (LEHMQ): 2,695 days, last price $0.13 on 2008-09-17
- **Enron** (ENRNQ): 1,731 days, last price $0.04 on 2004-11-17
- **Washington Mutual** (WAMUQ): 3,578 days, last price $0.00 on 2012-03-20
- **Ambac Financial** (ABKFQ): 5,051 days, last price $1.60 on 2018-01-30

---

## 4. Core Library (`pairs_eda/`)

### Module Dependency Graph

```
                    ┌─────────────────────┐
                    │  rolling_phase2.py   │  ← orchestrator
                    │  (WFA engine)        │
                    └──┬──────┬────────┬──┘
                       │      │        │
            ┌──────────┘      │        └──────────────┐
            ▼                 ▼                        ▼
   ┌─────────────────┐  ┌────────────────┐  ┌─────────────────────┐
   │  correlation.py  │  │ backtesting.py │  │  sp500_history.py   │
   │  - filter_       │  │ - compute_     │  │  - Sp500History     │
   │    volatile      │  │   zscore       │  │  - members_as_of    │
   │  - find_         │  │ - Kalman       │  │  - universe_fn      │
   │    candidate_    │  │   filter       │  └──────────┬──────────┘
   │    pairs         │  └────────────────┘             │
   │  - cointegration │                        ┌────────┴────────┐
   └─────────────────┘                         │ eodhd_download  │
            │                                  └─────────────────┘
            ▼
   ┌─────────────────┐
   │  yfinance_tools  │
   │  - download_     │
   │    with_retry    │
   │  - adj_close_or_ │
   │    close_panel   │
   └─────────────────┘
```

### Key Classes

```python
# Configuration — all 35+ parameters with validation
class RollingPhase2Config(BaseModel):
    training_months: int = 36           # rolling lookback
    validation_days: int = 180          # Phase 2a consistency check
    rebalance_frequency: str = "MS"     # monthly rebalance
    windows: tuple[int, ...] = (10, 12, 14, ..., 30)    # 11 values
    zscore_thresholds: tuple[float, ...] = (1.0, 1.1, ..., 2.5)  # 16 values
    leverage: float = 3.0
    max_slots: int = 10
    stop_loss_pct: float = 0.08
    circuit_breaker_pct: float = 0.12
    ...  # 20+ more fields with Pydantic validation

# Input — prices + config + optional PIT universe function
class RollingPhase2Input(BaseModel):
    prices: pd.DataFrame              # (dates × tickers) Adj Close panel
    initial_capital: float = 10_000.0
    config: RollingPhase2Config
    universe_fn: Optional[Callable[[pd.Timestamp], frozenset[str]]]
    #           ↑ PIT hook: returns valid tickers for a given date

# Output — full backtest results
class RollingPhase2Output(BaseModel):
    schedule: list[RebalanceWindow]   # rebalance timeline
    trades: list[dict]                # every trade with P&L
    daily_equity: pd.Series           # equity curve
    daily_return: pd.Series           # daily return series
    summary: dict                     # CAGR, Sharpe, MDD, etc.
```

---

## 5. WFA Engine — `rolling_phase2.py`

This is the most complex and important module (1,644 lines). Here is the complete
internal flow from start to finish:

### Phase 1: Rebalance Scoring Loop

```
build_rolling_timeline(inp)           ← creates monthly RebalanceWindows
    │
    ▼
For each rebalance window W:
    │
    │   phase1 = prices[W.phase1_start : W.phase1_end]
    │
    │   ┌── PIT filter (if universe_fn provided) ──────────────────────┐
    │   │   pit_members = universe_fn(W.rebalance_date)                │
    │   │   phase1 = phase1[columns ∩ pit_members]                     │
    │   └──────────────────────────────────────────────────────────────┘
    │
    ├── 1. filter_volatile_tickers(phase1)
    │       → removes tickers with extreme single-day moves
    │       → per-rebalance, NO look-ahead (uses only trailing data)
    │
    ├── 2. find_candidate_pairs(filtered, min_corr=0.40, max_corr=0.85)
    │       → pairwise RETURNS correlation (not price correlation!)
    │       → top_n=200 candidates sorted by correlation
    │
    ├── 3. filter_cointegrated_cached(pairs, prices, cache)
    │       → Engle-Granger ADF test with SMART CACHING:
    │         • deep-pass (p << 0.05): reuse for up to 6 rebalances
    │         • deep-fail (p >> 0.05): reuse for up to 6 rebalances
    │         • borderline (|p - 0.05| < margin): always retest
    │         → reduces computation by ~80%
    │
    ├── 4. compute_robust_pair_scores(cointegrated_pairs)
    │       → for each pair: _evaluate_pair_surface()
    │         (see Section 7 for the full scoring pipeline)
    │       → output: scored DataFrame sorted by final_score
    │
    └── Store watchlist_by_rebalance[W.date] = top scored pairs
```

### Phase 2: Daily Portfolio Simulation

```
sim_dates = all trading days in [first_phase2_start, last_phase2_end]

For each trading day:
    │
    ├── Advance rebalance pointer if needed
    │     → update current_watchlist
    │     → recalculate slot_notional = equity × leverage / max_slots
    │
    ├── CHECK EXITS (for each open position):
    │     │
    │     ├── Mean reversion exit:
    │     │     z crosses exit_threshold (e.g. z=0.0)
    │     │     BLOCKED if holding_days < min_holding_days
    │     │
    │     └── Stop-loss exit:
    │           unrealized P&L <= -(stop_loss_pct × slot_notional)
    │           ALWAYS allowed (overrides min_holding_days)
    │
    ├── CIRCUIT BREAKER:
    │     total_equity (realized + unrealized) dropped > cb_pct from peak
    │     → close ALL positions immediately
    │     → 5-day cooldown (no new entries)
    │
    ├── NEW ENTRIES (from current watchlist):
    │     │
    │     ├── max_new_entries_per_day cap (3)
    │     ├── min_entry_score gate (0.3)
    │     ├── sector diversification (max_sector_slots=3)
    │     ├── min_spread_range_pct gate (reject illiquid pairs)
    │     ├── max_zscore gate (>5.0 = structural break, reject)
    │     │
    │     └── z-score trigger → open position:
    │           z > entry_threshold → short the spread
    │           z < -entry_threshold → long the spread
    │
    └── Record equity = realized + sum(unrealized for all open positions)
```

---

## 6. Kalman Filter Z-Score

The z-score is the core trading signal. We use a **1-D local-level Kalman filter**
instead of a Simple Moving Average (SMA) because it adapts to structural breaks
within 1–3 bars instead of `window` bars.

### Why Not SMA?

```
SMA problem after a structural jump:

Price ratio │     ┌──────── 20% jump (e.g. earnings surprise)
            │     │
            │  ───┘         ← SMA takes FULL WINDOW (60 days) to catch up
            │                  Ghost z-score signal for 60 days!
            │
            └────────────── time

Kalman solution:

Price ratio │     ┌──────── 20% jump
            │     │
            │  ───┘         ← Kalman absorbs 60-90% of jump in 5-10 bars
            │  ·····            K ≈ 0.10-0.25 steady state
            │                   z-score returns to neutral within days
            └────────────── time
```

### Kalman Filter Equations

```python
# backtesting.py — _kalman_filter_loop()
# 1-D local-level model on log(price_a / price_b)

# State equation:   x[t] = x[t-1] + w,    w ~ N(0, Q)
# Observation eq:   y[t] = x[t]   + v,    v ~ N(0, R)

for t in range(n):
    # 1. PREDICT (Time Update)
    x_pred = x_filt              # state propagation
    p_pred = p_filt + Q          # uncertainty grows

    S = p_pred + R               # innovation variance
    pred_mean[t] = x_pred
    pred_std[t]  = sqrt(S)

    # 2. Z-SCORE (only after burn-in window)
    if t >= window:
        z[t] = (y[t] - x_pred) / sqrt(S)    # normalized innovation

    # 3. UPDATE (Measurement Update)
    K = p_pred / S               # Kalman gain ∈ [0, 1]
    x_filt = x_pred + K * (y[t] - x_pred)
    p_filt = (1 - K) * p_pred
```

### Auto-Tuning (No Lookahead)

Q and R are estimated from the **first `window` bars only** (burn-in):

```python
R = Var(y[:window])      # observation noise from burn-in data
Q = R / window           # process noise (yields steady-state gain ≈ 1/√window)
```

This gives a Kalman gain that roughly mirrors a `window`-bar EMA in steady state
but adapts **much faster** after large innovations.

### Z-Score to Trading Signal

```python
# _evaluate_pair_surface() in rolling_phase2.py

# Entry:
#   z > +threshold  → short the spread (expect reversion down)
#   z < -threshold  → long the spread (expect reversion up)
#   |z| > 5.0       → structural break, DO NOT TRADE

# Exit:
#   z crosses exit_threshold (default 0.0 = mean)
#   OR stop-loss hit

position = where(
    (z > threshold) & (z < max_zscore),  -1.0,   # short spread
    where(
        (z < -threshold) & (z > -max_zscore), 1.0,   # long spread
        0.0                                            # neutral
    )
)
```

---

## 7. Pair Scoring & Robustness

For each candidate pair, `_evaluate_pair_surface()` runs a full train/validation
analysis across the parameter grid. This is the **quality gate** — most pairs fail here.

### Grid Search Surface

```
              zscore threshold →
            1.0   1.2   1.4   1.6   1.8   2.0   2.2   2.4
  window  ┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐
    10    │  ·  │  ·  │  ·  │  ·  │  ·  │  ·  │  ·  │  ·  │
    14    │  ·  │  ·  │  ·  │  ✓  │  ✓  │  ✓  │  ·  │  ·  │
    18    │  ·  │  ·  │  ✓  │  ✓  │  ★  │  ✓  │  ·  │  ·  │  ← "stable region"
    22    │  ·  │  ·  │  ·  │  ✓  │  ✓  │  ✓  │  ·  │  ·  │
    26    │  ·  │  ·  │  ·  │  ·  │  ·  │  ·  │  ·  │  ·  │
    30    │  ·  │  ·  │  ·  │  ·  │  ·  │  ·  │  ·  │  ·  │
          └─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘
    ·  = failed consistency gates
    ✓  = profitable in both train and validation
    ★  = best_window, best_z (median of top-k region)
```

### The 6-Gate Funnel

```python
# _evaluate_pair_surface() — each gate can reject the pair entirely

# Gate 1: Training profitability
if train_entries == 0 or train_strat <= 0.0:
    continue  # Must have made money historically

# Gate 2: Validation profitability
if val_entries > 0 and val_strat <= 0.0:
    continue  # Recent trades lost money → pattern broken

# Gate 3: Anti-luck filter
if val_per_trade > train_per_trade * 5.0:
    continue  # Abnormally high recent returns → structural break / luck

# Gate 4: Best parameter selection (median of top-k, not the single best!)
best_window = median(top_k["window"])   # robust center of stable region
best_z = median(top_k["zscore"])

# Gate 5: Zero-Cost Stress Test (Neighborhood Cliff Check)
for each neighbor of (best_window, best_z):
    if neighbor.profit < best.profit * (1 - max_drop_pct):
        return None  # Performance cliff → overfit peak

# Gate 6: Z-score volatility consistency
z_vol_ratio = val_z_std / train_z_std
if z_vol_ratio > 2.5 or z_vol_ratio < 0.4:
    return None  # Spread dynamics changed structurally
```

### Final Score Computation

```python
stability = 1.0 / (1.0 + dist_window + dist_zscore + diff_ratio)
profit = tanh(val_margin_daily * 252 * 30)   # saturating profit score
final_score = 0.6 * profit + 0.4 * stability
```

- `dist_window`: coefficient of variation of window across top-k (lower = more stable)
- `dist_zscore`: coefficient of variation of zscore across top-k
- `diff_ratio`: |train - validation| daily margin divergence

---

## 8. Daily Portfolio Simulation

### Position Lifecycle

```
Day 1: Entry
    z-score crosses threshold → calculate quantities:
        slot_notional = equity × leverage / max_slots
        half = slot_notional / 2
        qty_a = half / price_a × (1 + slippage)   # buy with slippage
        qty_b = half / price_b × (1 - slippage)   # sell with slippage
        → deduct entry commission

Day 2..N: Holding
    Each day: mark-to-market (unrealized P&L)
    Check stop-loss: unrealized <= -(stop_loss_pct × slot_notional)?
    Check mean reversion: z-score crossed exit_threshold?
        → only if holding_days >= min_holding_days

Day N+1: Exit
    Apply slippage on closing prices
    realized_equity += P&L - exit_commission
    Record trade to trades list
```

### Slot-Based Position Sizing

```
                    Total Equity: $100,000
                    Leverage: 3.0×
                    Max Slots: 10
                           │
                           ▼
           Gross Exposure: $300,000
           Per-Slot Notional: $30,000
                           │
          ┌────────────────┼────────────────┐
          ▼                ▼                ▼
       Slot 1          Slot 2           Slot 3
    ┌──────────┐    ┌──────────┐    ┌──────────┐
    │ Long  $15k│   │ Long  $15k│   │ Long  $15k│
    │ Short $15k│   │ Short $15k│   │ Short $15k│
    │ Pair: A/B │   │ Pair: C/D │   │ Pair: E/F │   ... up to 10 slots
    └──────────┘    └──────────┘    └──────────┘
```

### Multi-Layer Risk Control

```
                   ┌──────────────────────────────────┐
                   │         PORTFOLIO LEVEL           │
                   │  Circuit Breaker: equity DD > 12% │
                   │  → close ALL, 5-day cooldown      │
                   └──────────────┬───────────────────┘
                                  │
              ┌───────────────────┼───────────────────┐
              ▼                   ▼                    ▼
   ┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
   │  SECTOR LEVEL    │ │ ENTRY LEVEL      │ │ PER-TRADE LEVEL  │
   │  max 3 slots per │ │ max 3 new/day    │ │ stop-loss: 8%    │
   │  sector          │ │ min_score: 0.3   │ │ min hold: 3 days │
   └──────────────────┘ │ max_zscore: 5.0  │ │ exit z: 0.0      │
                        │ min_spread: 3%   │ └──────────────────┘
                        └──────────────────┘
```

---

## 9. Hybrid Strategy — Macro Regime Switching

Pairs trading is a bear-market hedge, not a growth engine. The hybrid strategy
holds the S&P 500 in normal markets and switches to Pairs Trading only during
structural corrections.

### Bear Market Detection

**Entry is fast (protect capital), exit is slow (confirm recovery):**

```
  ENTRY (2 conditions, both must be true):
    ① S&P 500 drawdown from peak hits entry_dd (-10%)
    ② 100-day MA slope averaged over 15 days is negative
    ③ Cooldown period (40 days) since last exit has passed

  EXIT (2 conditions):
    ① 100-day MA slope averaged over 15 days turns positive
    ② At least 60 trading days have passed in bear mode
```

Why asymmetric? Being late to *enter* pairs costs real money (unhedged crash).
Being late to *exit* pairs costs only opportunity (pairs still earn, just less than S&P 500).

```
S&P 500 price
│         ╱╲
│        ╱  ╲               ← DD hits -10% AND slope < 0: ENTER pairs
│       ╱    ╲
│      ╱      ╲  ___
│     ╱        ╲╱   ╲___    ← slope still negative (stay in pairs)
│    ╱                   ╲___╱── ← slope avg > 0 for 15d: EXIT
│   ╱                              (after min 60d in bear)
└──────────────────────────────── time
    │←──── S&P 500 ────→│←── Pairs ──→│←── S&P 500 ──→│
```

### On-Demand WFA Architecture

The WFA engine is expensive (~45 min for full-period). Since pairs trading is
only active ~10-15% of the time, the hybrid strategy runs WFA **on-demand**:

```python
# run_hybrid_backtest() — simplified flow

# 1. Pre-compute all bear episodes using S&P 500 regime signal
episodes = detect_bear_episodes(sp500, entry_dd=-0.10, ...)

# 2. For each trading day:
for day in sim_dates:
    if entering_bear:
        # Run WFA ONLY for this bear episode's date range
        ep_result = run_phase2_rolling(
            prices[episode_start - lookback : episode_end]
        )
        bear_wfa_daily_ret = ep_result.daily_return
        #                    ↑ cached & reused for entire episode

    if in_bear:
        equity *= (1 + wfa_ret - pairs_daily_carry)
    else:
        equity *= (1 + sp500_ret - fx_daily_carry)
```

This cuts WFA computation by **~85-90%** compared to running it full-period.

### Carry Costs (Realistic Friction)

| Regime | Cost | Default | Rationale |
|--------|------|---------|-----------|
| Pairs Trading | `pairs_carry_bps` | 200 bps/yr | IBKR margin rate - short rebate spread |
| S&P 500 (FX hedged) | `fx_hedge_carry_bps` | 350 bps/yr | Realistic IBKR margin spread |

**Why 350 bps for FX hedge, not the theoretical interest rate differential?**

In academic theory (Covered Interest Rate Parity), hedging cost = local rate − USD rate.
In practice, IBKR applies a **Brokerage Interest Spread (Haircut)**:

```
Theoretical (CIP):  KRW 3.0% - USD 4.5% = -1.5% (earn 1.5%)
Actual (IBKR):      Earn 0~2.5% on KRW - Pay 6.0% on USD = -3.5% (pay 3.5%)
```

**Pro alternative: MES Futures (Micro E-mini S&P 500)**
- Traded on CME at institutional wholesale rates
- Bypasses IBKR's retail interest spread
- Hedging cost converges to true CIP → set `fx_hedge_carry_bps ≈ 0–50`

---

## 10. Point-in-Time (PIT) Survivorship Bias Fix

### The Problem

Using **today's** S&P 500 list (503 tickers) to backtest historical periods creates
survivorship bias: we only trade companies that survived until 2026, excluding
bankrupt firms (Enron, Lehman, Washington Mutual) that would have hurt returns.

### Three-Stage Discovery (Lecture Storyline)

```
Stage 1: Biased Backtest
    Universe = 2026 S&P 500 (503 tickers, all survivors)
    Result: Very high returns (5,642%)
    Problem: Future information leak — "don't use tomorrow's newspaper"

Stage 2: PIT with Free Data (the trap!)
    Universe = hanshof point-in-time membership (469-503 per window)
    Data = Yahoo Finance only → bankrupt companies return NO DATA
    Result: Even HIGHER returns (10,680%)
    Why: The PIT universe correctly excludes future additions,
         but also accidentally excludes the WORST performers
         (bankrupt companies) whose price data Yahoo doesn't have.
         → A more subtle form of survivorship bias

Stage 3: PIT with Complete Data (honest)
    Universe = hanshof point-in-time membership
    Data = Yahoo Finance + EODHD (delisted stocks, $19.99)
    Coverage: 1,108 / 1,126 tickers = 98.4%
    Result: [running now — expected to show lower returns]
    Key: Now includes Enron going from $90→$0.04, Lehman $67→$0.13, etc.
```

### PIT Implementation

```python
# sp500_history.py — Sp500History class

class Sp500History:
    """3,482 daily snapshots of S&P 500 membership (1996–2025)."""

    def members_as_of(self, date: pd.Timestamp) -> frozenset[str]:
        """O(log n) binary search for membership at any historical date."""
        idx = bisect.bisect_right(self._dates, date) - 1
        return self._members[idx]

    def universe_fn(self) -> Callable[[pd.Timestamp], frozenset[str]]:
        """Plug directly into RollingPhase2Input.universe_fn."""
        return self.members_as_of
```

```python
# rolling_phase2.py — PIT filtering hook in rebalance loop

for window in schedule:
    phase1 = prices.loc[window.phase1_start : window.phase1_end]

    if inp.universe_fn is not None:
        pit_members = inp.universe_fn(window.rebalance_date)
        valid_cols = [c for c in phase1.columns if c in pit_members]
        phase1 = phase1[valid_cols]
        # ↑ Now only trades tickers that were ACTUALLY in S&P 500 on this date

    scored, coint_stats = compute_robust_pair_scores(phase1, cfg, ...)
```

### EODHD Integration

For the 341 tickers missing from Yahoo Finance (mostly delisted/bankrupt):

```python
# eodhd_download.py — symbol mapping strategy

def _candidate_symbols(hanshof_ticker: str) -> list[str]:
    """Hanshof → EODHD symbol candidates."""
    candidates = [f"{ticker}.US"]
    if ticker.endswith("Q"):          # bankruptcy suffix
        candidates.append(f"{ticker[:-1]}.US")
    if "-" in ticker:                 # class shares (e.g. BRK-B)
        candidates.append(f"{ticker.replace('-', '.')}.US")
    return candidates

# Result: 323 found via EODHD, 18 truly unfindable (pre-1996 companies)
```

---

## 11. Runners & Dashboards

### Architecture

```
             ┌──────────────────────────────────────────────┐
             │              tmux session                     │
             │                                              │
             │  Window 0: Streamlit Dashboard               │
             │    ┌──────────────────────────────────┐      │
             │    │  Auto-refreshes every 3 seconds   │      │
             │    │  Reads from docs/*-progress.json  │      │
             │    │  Shows equity curves + metrics    │      │
             │    └──────────────────────────────────┘      │
             │                                              │
             │  Window 1: Python Runner                     │
             │    ┌──────────────────────────────────┐      │
             │    │  Loads data, runs WFA             │      │
             │    │  Writes progress JSON every 10    │      │
             │    │  trading days via on_step callback│      │
             │    └──────────────────────────────────┘      │
             └──────────────────────────────────────────────┘
```

### Three Independent Pipelines

| Pipeline | Runner | Dashboard | Port | tmux Session | Progress File |
|----------|--------|-----------|------|-------------|---------------|
| Hybrid | `run_wfa.py` | `wfa_dashboard.py` | 8501 | `wfa` | `wfa-progress.json` |
| Full Pairs | `run_pairs_only.py` | `pairs_dashboard.py` | 8502 | `pairs` | `pairs-progress.json` |
| PIT Pairs | `run_pairs_pit.py` | `pairs_pit_dashboard.py` | 8503 | `pairs-pit` | `pairs-pit-progress.json` |

### Launching

```bash
# Start any pipeline (tmux handles background + persistence):
bash reference/python_pairstrading/run_all.sh         # hybrid
bash reference/python_pairstrading/run_pairs_only.sh  # full pairs
bash reference/python_pairstrading/run_pairs_pit.sh   # PIT pairs

# Monitor:
tmux attach -t wfa         # or pairs, pairs-pit
open http://localhost:8501  # or 8502, 8503

# Stop:
tmux kill-session -t wfa
```

### Progress Callback Pattern

All runners use the same callback pattern for live dashboard updates:

```python
# Common pattern across all runners
_dates, _equity_y, _dd_y = [], [], []
_peak = 0.0

def on_step(day, equity, step_idx, total):
    global _peak
    _dates.append(day.strftime("%Y-%m-%d"))
    _equity_y.append(equity)
    _peak = max(_peak, equity)
    _dd_y.append(equity / _peak - 1.0)

    if step_idx % 10 == 0:
        PROGRESS_FILE.write_text(json.dumps({
            "dates": _dates,
            "equity": _equity_y,
            "dd": _dd_y,
            "pct": f"{step_idx/total*100:.0f}%",
        }))

result = run_phase2_rolling(inp, on_step=on_step, step_interval=1)
```

---

## 12. Configuration Reference

### WFA Parameters (current active — `run_pairs_pit.py`)

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `training_months` | 36 | Rolling window size (3 years) |
| `validation_days` | 180 | Phase 2a consistency check (~6 months) |
| `rebalance_frequency` | "MS" | Monthly rebalance |
| `coint_significance` | 0.10 | p-value threshold for Engle-Granger test |
| `coint_retest_margin` | 0.02 | Borderline margin for cache retest |
| `min_correlation` | 0.40 | Minimum returns correlation for candidacy |
| `max_correlation` | 0.85 | Maximum returns correlation (avoid same-stock) |
| `min_overlap_pct` | 0.80 | Minimum shared trading day coverage |
| `top_n_candidates` | 200 | Max candidate pairs per rebalance |
| `windows` | 10, 12, ..., 30 (11) | Kalman burn-in windows to test |
| `zscore_thresholds` | 1.0, 1.1, ..., 2.5 (16) | Entry thresholds to test |
| `stress_test_window_step` | 2 | Neighbor step for cliff check |
| `stress_test_zscore_step` | 0.1 | Neighbor step for cliff check |
| `watchlist_size` | 200 | Scored pairs per rebalance |
| `max_slots` | 10 | Max concurrent positions |
| `max_new_entries_per_day` | 3 | Daily entry cap |
| `leverage` | 3.0 | Position sizing multiplier |
| `max_drop_quantile` | 0.90 | Drop worst-10% volatile tickers |
| `entry_zscore_default` | 2.0 | Default entry threshold |
| `exit_zscore` | 0.0 | Exit when z-score reverts to mean |
| `stop_loss_pct` | 0.08 | Per-trade loss limit (8%) |
| `min_holding_days` | 3 | Prevents ultra-fast churn |
| `circuit_breaker_pct` | 0.12 | Portfolio-level tail risk defense |
| `min_entry_score` | 0.3 | Quality gate for new entries |
| `max_sector_slots` | 3 | Sector concentration limit |
| `min_spread_range_pct` | 0.03 | Reject illiquid pairs |
| `commission_per_leg_bps` | 0.5 | Commission cost per leg |
| `slippage_per_leg_bps` | 0.5 | Slippage assumption per leg |

### Grid Size & Performance

| Config | Grid | Combos | Approx Runtime |
|--------|------|--------|----------------|
| Current (full) | 11 × 16 | 176 | ~45 min |
| Narrowed (planned) | 5 × 8 | 40 | ~10 min |
| Adaptive 2-pass | ~15 + ~9 | 24 | ~7 min |
| Bayesian (Optuna) | — | 20-30 | ~5 min |

### Hybrid Parameters

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `entry_dd` | -10% | Drawdown threshold to enter pairs mode |
| `entry_slope_confirm_days` | 15 | Avg MA slope must be negative over this window |
| `exit_ma_window` | 100 days | MA window for exit slope calculation |
| `exit_slope_window` | 20 days | Days over which to measure MA slope |
| `exit_slope_confirm_days` | 15 days | Avg slope must be positive over this window |
| `min_bear_days` | 60 days | Minimum stay in bear mode (~3 months) |
| `cooldown_days` | 40 days | No re-entry for ~2 months after exit |
| `pairs_carry_bps` | 200 bps/yr | Pairs trading margin cost |
| `fx_hedge_carry_bps` | 350 bps/yr | FX hedge cost (SPY margin) |

---

## 13. Code Strengths

### Architecture
- **Clean separation**: library (`pairs_eda/`) vs. runners vs. dashboards vs. docs
- **Pydantic models** for all config/input/output — type-safe, self-documenting, serializable
- **42 public symbols** cleanly exported through `__init__.py`
- **85 unit tests** covering backtesting, correlation, WFA, and data download
- **Cointegration caching** — reduces redundant computation by ~80-90%
- **Progress callbacks** — enables live Streamlit monitoring of multi-hour runs
- **PIT-ready architecture** — `universe_fn` hook makes bias correction a config change

### Strategy Design
- **Walk-Forward Analysis** eliminates look-ahead bias (train → validate → simulate)
- **Returns correlation** (not price correlation) — statistically correct
- **Kalman filter** for z-score — adapts to structural breaks in 1-3 bars
- **Robust parameter selection** — median of top-k region, not the single best point
- **Zero-cost stress test** — rejects overfit peaks by checking performance neighbors
- **Multi-layer risk control**: per-trade stop-loss, min holding days, circuit breaker, sector limits
- **Adj Close** used consistently — accounts for splits/dividends
- **Capital compounding** — slot notional recalculated each rebalance based on current equity

### Development Workflow
- **10 Cursor rules** enforce consistency (commit style, refactor rules, bilingual response, etc.)
- **Quant agent pipeline** (`/quant-coding`, `/quant-brainstorm`, `/quant-refactor`) with specialized AI agents
- **Lecture-driven development** — `docs/lecture-ideas.md` captures teaching insights alongside code
- **Three independent tmux pipelines** — hybrid, full pairs, PIT — run in parallel
- **Auto-commit/push** on meaningful changes (rule-enforced)

---

## 14. Code Weaknesses & Technical Debt

### Architecture
- **Notebook is too large** (88 cells, ~6,500 lines with outputs). Cells 32-70 are dead code.
- **`rolling_phase2.py` is monolithic** (1,644 lines). The daily simulation loop alone is ~300 lines. Should be split into: config, timeline, scoring, simulation, output.
- **`vectorized_backtest.py` vs `backtesting.py`** — two parallel backtest implementations.
- **Helper functions defined in notebook** (`append_wfa_run_log`, etc.) should be in library.
- **No type stubs for pandas** — many Pyright workarounds.

### Performance
- **Grid search is brute-force** (176 combos per pair per rebalance). Adaptive/Bayesian alternatives could cut runtime by 80%.
- **Kalman filter is pure Python** (PERF-001). Numba/Cython JIT would give ~50× speedup.
- **Feature cache rebuilt per pair** — `_create_feature_cache` is called repeatedly for the same pair across days.
- **Sequential processing** — no `joblib` or multiprocessing for pair scoring.

### Testing
- **No integration test** for full WFA pipeline end-to-end on synthetic data.
- **No test for hybrid backtest** or PIT functionality.
- **No regression tests** against known historical results.

### Operational
- **No real-time execution engine** — backtest-only.
- **No IBKR/TWS integration** yet.
- **No position reconciliation** — no way to verify simulated vs actual positions.
- **No earnings blackout** — trades may open around earnings dates.

---

## 15. Investment Strategy Strengths

- **Market-neutral by design** — long/short pairs hedge out market beta
- **Grounded in statistics** — cointegration + z-score mean-reversion is well-established
- **Adaptive regime handling** — Kalman filter quickly resets after structural breaks
- **Walk-forward validation** — avoids the classic trap of in-sample overfitting
- **Hybrid adds upside** — captures S&P 500 bull market returns + pairs hedge in corrections
- **Three-stage PIT validation** — progressively eliminates survivorship bias (a teaching tool AND scientific rigor)

---

## 16. Investment Strategy Weaknesses

### Structural Limitations
- **Survivorship bias** — partially addressed with PIT + EODHD, but 18 tickers still missing
- **Daily granularity** — stop-loss checked once per day; intraday gaps can exceed 8% limit
- **Equal-weight S&P 500 proxy** — not a true cap-weighted S&P 500 benchmark

### Risk Concerns
- **Max drawdown still high** (~22-35% depending on config)
- **Leverage amplifies tail risk** — 3x leverage means 12% portfolio DD = 36% gross exposure shock
- **Correlation breakdown in crises** — COVID 2020 showed all pairs can move together
- **Low recent returns** (2020-2024 underperforms S&P 500 buy-and-hold in strong bull market)

### Overfitting Risk
- **Many tunable parameters** (35+ in config) — high-dimensional optimization surface
- **Iterative manual tuning** — each run influenced the next parameter choice
- **Exceptional historical performance** should be treated with skepticism until confirmed live

---

## 17. Recommended Next Steps (Priority Order)

1. **Complete PIT backtest** — currently running with EODHD data (1,108 tickers). Compare Stage 1/2/3.
2. **Narrow parameter grid** — analyze winning `(window, zscore)` distribution → reduce 176 → ~40 combos (77% speedup)
3. **Clean up notebook** — remove 40 dead cells (32-70), move helper functions to library
4. **Split `rolling_phase2.py`** — config, timeline, scoring, simulation as separate modules
5. **Numba-JIT the Kalman loop** — ~50× speedup for z-score computation (PERF-001)
6. **Bayesian optimization** — replace grid search with Optuna for further speed gains
7. **Live trading bridge** — IBKR TWS API integration (separate module)
8. **MES futures integration** — reduce FX hedge carry from 350 bps to ~0-50 bps
