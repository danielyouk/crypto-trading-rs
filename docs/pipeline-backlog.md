# Pairs Trading Pipeline — Backlog

> Project tracking for the pairs trading pipeline.
> Record completed milestones, in-progress work, and pending tasks here.
> For **design decisions and agent constraints**, see `.cursor/rules/pairs-trading-pipeline.mdc`.

---

## Completed

- [x] Data acquisition: S&P 500 fetch (Wikipedia + Gemini fallback)
- [x] yfinance download with retry (only failed tickers re-downloaded)
- [x] Pair-level overlap filter: min_overlap_years=5 (replaces per-ticker MIN_HISTORY_START cutoff)
- [x] Correlation screening: returns correlation (pct_change), band [0.40, 0.85], dual condition (full + 3yr recency)
- [x] Phase design: P1 = daily (1990→p1_end), P2a = 1h (730d), P2b = 5m (60d)
- [x] find_candidate_pairs returns dict {pair: corr_value}
- [x] Histogram: full returns correlation distribution with candidate cutoff line

## In Progress

- [ ] **Kalman Filter for Z-score calculation**: Replace Simple Moving Average (SMA) with a Kalman Filter to mathematically handle structural breaks (jumps) in the spread. This allows us to treat post-jump spreads as the "new normal" and reduces the need for aggressive volatility filtering.
- [ ] Parameter optimization: z-score thresholds and rolling window per pair (notebook cells 17-31)

## Pending

Items below should be implemented when reaching the corresponding stage.

### After Optimization
- [ ] **Non-overlapping pair selection**: each ticker in at most 1 (or K) active pair(s). Use graph maximum matching weighted by expected return. Prevents concentration risk. Ref: Gatev et al. (2006).

### Phase 2
- [ ] **Phase 2a reassurance on 1h data** (730 days) — validate strategy across recent market events.
- [ ] **Phase 2b reassurance on 5m data** (60 days) — execution-level validation with the selected non-overlapping pairs.

### Operational Risk Scenarios
- [ ] **Active position becomes ineligible**: ticker delisted, removed from S&P 500, or dropped by data filter while positions are open. Options: (a) immediate close at market, (b) set stop-loss and wait for orderly exit, (c) switch to manual management. Needs policy decision + implementation in the execution layer (outside notebook scope). **Critical for live trading automation.**
- [ ] **Scenario testing infrastructure**: Adapt reference/.cursor/skills/scenario-testing pattern. YAML-defined scenarios, separate `scenario_tests/` folder, brainstorm→implement workflow. Cover: delisting mid-trade, data gap during position, spread blowout beyond stop-loss, exchange API outage during rebalance.

### Course 2: FX Risk Management
- [ ] **FX-adjusted returns module**: Show historical EUR (and KRW) vs. USD returns for pairs trading strategies. Help students decide if USD alpha justifies FX exposure.
- [ ] **Automated FX hedge**: Python script using `ib_insync` — event-driven or daily threshold-based rebalancing of EUR/USD position to match USD cash balance. Leverage 1x on hedge side.
- [ ] **Margin safety monitor**: Alert or pause when total margin usage exceeds 60-70% of equity, to prevent IB from liquidating FX hedge during pairs drawdown.

### IB Account Setup (Pre-Launch)
- [ ] **Live account upgrade**: Cash Account → Margin Account → Portfolio Margin. Currently EUR 20 cash account (U12848664). Paper account (DU7788413) has Portfolio Margin with USD 187K.
- [ ] **Verify margin parity**: Confirm live and paper accounts have identical margin type before deploying real capital.
- [ ] **Data subscription**: Real-time Level 1 US equity data (~USD 10/month, waived with sufficient commissions).

### Future Enhancements
- [ ] **LLM-assisted strategy design**: feed pair statistics (spread distribution, half-life, volatility regime) to LLM. Build as Cursor skill or agent command.
- [ ] **Regime-adaptive parameters**: different z-score / window per volatility regime (low vol vs high vol).
- [ ] **Cointegration test (ADF on spread)**: the real pair selection criterion. Returns correlation is only a coarse pre-filter; cointegration validates that the spread is mean-reverting. Expected to reduce ~4K candidates to ~200-500.
