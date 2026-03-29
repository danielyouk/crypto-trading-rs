# WFA Tuning Log

This document records walk-forward tuning decisions, rationale, and outcomes.
Goal: improve risk-adjusted return while controlling drawdowns, especially in 2020-2022 and recent years.

## Current Active Configuration (latest)

- `training_months = 48`
- `validation_days = 84`
- `min_overlap_years = 2.5`
- `circuit_breaker_pct = 0.12`
- `rebalance_frequency = "MS"`
- `leverage = 3.0`
- `max_slots = 7`
- `min_entry_score = 0.5`
- `max_sector_slots = 2`
- `min_spread_range_pct = 0.05`

## Why This Midpoint Configuration

- **3-year window was too reactive**: better adaptation but higher churn and weaker recent profitability.
- **5-year window was too stale**: stronger stability but slower adaptation in post-COVID regime.
- **4-year window (48m)** is a compromise to preserve adaptation without overfitting to short-term noise.
- **Circuit breaker 10% was too aggressive**: frequent early liquidation can reduce participation in rebounds.
- **Circuit breaker 12%** keeps tail-risk control but gives trades more room to recover.
- **Validation 84 days** tightens consistency enough to avoid fragile pairs while not being as strict as 6 months.

## Key Observations from Prior Iterations

- Drawdown control improved after tighter risk settings, but recent 5-year returns degraded.
- Recent underperformance appears linked to:
  - elevated stop-loss frequency during regime transitions
  - overly conservative global risk controls reducing exposure in recoveries
  - parameter instability when training window is too short

## Experiment Log Template

Use this table after each run:

| Date | Config Hash | 2012-2016 | 2016-2020 | 2020-2024 | Full Cumulative | Full Annualized | Max DD | Sharpe | Notes |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| YYYY-MM-DD | tm48-v84-mo2.5-cb12 | - | - | - | - | - | - | - | baseline midpoint |

Suggested config hash format:
- `tm{training_months}-v{validation_days}-mo{min_overlap_years}-cb{circuit_breaker_pct*100}`

## Next Step: Automated Parameter Search

This can be turned into an optimization problem over a bounded grid:

- Search dimensions:
  - `training_months`: `[36, 48, 60]`
  - `validation_days`: `[63, 84, 126]`
  - `circuit_breaker_pct`: `[0.10, 0.12, 0.15]`
  - `min_entry_score`: `[0.45, 0.50, 0.55]`
  - `max_sector_slots`: `[1, 2, 3]`

- Objective (example):
  - maximize: `annualized_return - 0.5 * abs(max_drawdown) + 0.2 * sharpe`
  - constraint: `max_drawdown >= -0.45`
  - constraint: `2020-2024 annualized >= threshold`

- Procedure:
  1. run WFA for each candidate config
  2. compute period metrics and full-period metrics
  3. rank by objective under constraints
  4. keep top-N configs for robustness check (not only top-1)

This keeps the process systematic and reproducible instead of ad-hoc tuning.
