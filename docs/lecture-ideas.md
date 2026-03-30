# Lecture Ideas & Discussion Highlights

> Insights, visual examples, and teaching points collected during development.
> Each entry serves as raw material for lecture slides, live session topics, or notebook annotations.

## Table of Contents
1. [Course Structure & Strategy](#course-structure--strategy)
2. [Data & Correlation](#data--correlation)
3. [Pipeline Design & Architecture](#pipeline-design--architecture)
4. [Risk Management & Execution](#risk-management--execution)
5. [FX Risk Management for Non-USD Traders](#fx-risk-management-for-non-usd-traders-course-2)
6. [Performance Measurement & Metrics](#performance-measurement--metrics)
7. [Open Source & Community](#open-source--community)

---

## Course Structure & Strategy

### Revenue Model: Online + Live Workshop

**Online Course (Course 1): ~15 hours, self-paced**

| Part | Hours | Content |
|------|-------|---------|
| Part 1: Strategy | 6h | Data → Correlation → Cointegration → Backtest → Grid Search |
| Part 2: Validation | 4h | WFA → Results Analysis → Leverage/Margin → "Does this make money?" |
| Part 3: Go Live | 5h | IBKR setup → TWS API → Production code → Oracle VM → Monitoring |

**Language**: English (primary). Korean version update planned separately
(existing Korean course to be refreshed with WFA + automation content).

**Live Workshop (Premium): Weekend, ₩300K–₩1M per person**
- Target: Online course graduates who want deep, hands-on guidance.
- Content (not included in online course):
  - Iterative strategy refinement — loss-period forensics with LLM agent
  - Earnings blackout implementation
  - Sector concentration constraints
  - Volatility regime detection
  - Advanced position sizing (Kelly criterion)
  - Live debugging of student's own IBKR setup
- Format: Small group (5–15 people), screen-sharing, real portfolio review.
- Value proposition: "The online course teaches you the system.
  The workshop teaches you to evolve it."

### Course Lineup (Independent, Self-Contained)
- **Course 1**: Python stock pairs trading (S&P 500) — includes a dedicated module on FX risk management (Natural Hedge via Margin Loan) for non-USD investors.
- **Course 2**: Python FX & Commodities Trading (Macro Quant) — Traditional quant methods applied to currencies and commodities (Carry trades, Cross-asset statistical arbitrage like CAD vs. Crude Oil).
- **Course 3**: Python AI Quant Assistant (RAG & Multimodal) — Post-market reverse engineering. Building a RAG system with quant textbooks, codebase, and charts to automatically analyze daily trading anomalies and propose code improvements.
- **Course 4**: Rust grid trading bot (crypto) — Rust entry point. Ownership, types, async through a simple strategy. One exchange, one asset, clear state machine.
- **Course 5**: Rust funding rate arbitrage (crypto) — the serious Rust project. Multi-exchange delta-neutral hedging (spot + futures), websocket streaming, state persistence, error recovery.
- **Course 6**: Rust cross-exchange arbitrage (crypto) — where Rust's speed genuinely matters. Concurrent connections, latency optimization.
- Platform: Udemy — each course ~10 hours max, independently complete.

### Rust Course Philosophy (Courses 3-5) — Honest Framing
**What the Rust courses are NOT:**
- NOT "Rust is better than Python" — Python runs 24/7 just fine (Instagram, Spotify, countless trading bots prove this). Arguments like GIL limitations or memory leaks are weak and misleading for this use case.
- NOT "master Rust" — the goal is not language mastery but practical comfort.

**What the Rust courses ARE:**
- **Goal 1: Rust proficiency through real projects.** Students become comfortable enough with Rust to confidently use it in any future project.
- **Goal 2: Actually build working crypto trading systems.** These are not toy projects. The strategy design, risk management, and execution logic are real and practical.
- **Python + Rust together, not Rust replacing Python.** Strategy research stays in Python/Jupyter (where it belongs). Execution and deployment use Rust (where its strengths — single binary, no runtime dependencies, deployment simplicity — genuinely shine).

**Honest Rust advantages:**
- Single static binary: `scp bot server && ./bot` — no virtualenv, no pip, no dependency issues on the server
- ~5-20 MB memory vs. ~50-200 MB Python — meaningful only when running multiple services on a free VM
- If it compiles, certain classes of bugs (null references, data races) cannot exist
- Learning Rust is career-valuable for engineers, independent of this specific project

**What we explicitly tell students:**
- "Python can do everything in these courses. We use Rust because learning it through a real project is the most effective way to gain confidence with the language."
- "By the end, you won't be a Rust expert — but you'll be able to read Rust code, write Rust code, and decide for yourself when Rust is the right tool."

### Why Each Course Teaches Rust Differently
- **Course 3 (Grid)**: ownership, basic async, state machines — "your first Rust project that makes real trades"
- **Course 4 (Funding rate)**: multi-exchange concurrency, complex state, error recovery — "the engineering problem that keeps your money safe while you sleep"
- **Course 5 (Arbitrage)**: performance optimization, low-latency networking — "the one strategy where 'Rust is fast' actually means more profit"
- No repetition — each course deepens Rust through a different engineering challenge.

### Development vs. Deployment Architecture (Courses 3-5)
- **Development (laptop)**: Python + PyO3 → Jupyter for visualization, Rust crate for core logic
- **Deployment (VM)**: Same Rust crate compiled as pure binary → no Python runtime needed on server

---

## Data & Correlation

### Price Correlation vs. Returns Correlation — Why They Differ
- **Visual**: S&P 500 26-year price correlation (median ~0.90) vs. returns correlation (median ~0.31)
- **Key message**: Two stocks both trending up over decades will have high price correlation automatically (shared trend). Returns correlation measures genuine daily co-movement.
- **Anticipated student question**: "Why is 0.5 considered high?" → R²=0.25. Even the entire market (S&P 500 index) only explains 30-40% of individual stock variance. Two individual stocks sharing 25% is remarkably strong.

### FRT/KIM 2008 Case — The Limits of Correlation
- **Visual**: FRT/KIM price chart (r=0.84). During 2008 GFC, KIM dropped ~85% vs. FRT ~70%.
- **Key message**: Same sector (retail REIT) but quality gap surfaces during crises. High correlation does not prevent extreme spread divergence.
- **Lecture storyline**:
  1. "High correlation, so they move together, right?" → show the chart
  2. "But look at 2008" → one side crashed far more severely
  3. "That's why we need cointegration" → verify spread mean-reversion
  4. "And hedge ratio adjustment" → weight by β, not 1:1
  5. "And still, stop-loss on spread" → practical risk management
- **Student takeaway**: Why multiple pipeline stages are necessary — this single chart explains it all.

### Survivorship Bias — Known Limitation

**The problem**: Our backtest uses today's S&P 500 constituent list applied
to the entire 1990–2024 period. In reality, the S&P 500 changes ~20-30
stocks per year — companies are added (TSLA in 2020) and removed (Enron
in 2001, Lehman in 2008). Removed stocks often had poor performance before
removal, and we never see them in our current list.

**Impact**:
- We are backtesting on *survivors* — stocks that were successful enough
  to remain in (or be added to) the S&P 500 by 2024.
- Stocks that were delisted, bankrupted, or removed are excluded from our
  universe, which slightly inflates historical returns.

**Why it's acceptable for this course**:
- Our goal is to validate the *mechanism* of pairs trading (cointegration,
  z-score mean-reversion, stop-loss behavior) — not to claim precise
  historical returns.
- In live trading, we will always use the *current* S&P 500 list, which
  is updated quarterly. So the survivorship bias only affects the backtest,
  not the live system.
- The WFA rolling framework naturally handles constituent changes: new
  stocks appear when they have enough data; removed stocks disappear from
  the price panel.

**Lecture note**: Present this as an honest disclosure. Students respect
transparency about limitations. "This backtest shows the strategy *works*,
but the exact return numbers have a small upward bias due to survivorship.
In live trading, this bias does not exist."

---

## Pipeline Design & Architecture

### Why Coarse Filter → Cointegration → Half-life
- **Key message**: Testing all 60,726 pairs for cointegration is too expensive. Returns correlation is a "coarse sieve" — confirms minimum co-movement, then cointegration does the real validation.
- **Numbers**: 60,726 → ~12,000 (corr filter) → ~200-500 (cointegration) → final pairs

### The Computational Bottleneck (Backtesting 26 Years of Data)
- **The Problem**: When moving from a 2-year backtest to a 26-year backtest (1990s to present), the grid search for optimal parameters (window size, z-score threshold) across hundreds of pairs becomes computationally explosive.
- **The Solution (Vectorized Backtesting)**: We replace the slow Python loops (e.g., `iterrows`, `apply`) with pure NumPy array operations and Pandas' C-optimized `.rolling()` methods.
- **The "Aha!" Moment on Scale (Why Pipeline Ordering Matters)**:
  - Even after vectorizing (reducing a single 26-year backtest to ~4 milliseconds), running a grid search (567 parameter combinations) across 14,000 "highly correlated" pairs means running **~8 million backtests**. This still takes ~10 hours.
  - *If we didn't vectorize, 8 million backtests at 1 second each would take 92 days!*
  - **The Real Fix (Architecture)**: Never run grid search on the coarse filter output. The pipeline MUST be: 
    1. Coarse Filter (Correlation) -> 14,000 pairs
    2. **Cointegration Test** -> Reduces to ~500 structurally sound pairs
    3. Grid Search Optimization -> 500 pairs * 2.5 seconds = **20 minutes**.
- **Python/NumPy vs. Rust (The Truth About Speed)**:
  - **Student Question**: "If we rewrite this backtest in Rust, will it be 100x faster?"
  - **Answer**: No. NumPy is already executing C code under the hood. The vectorized operations (`np.divide`, `np.where`) are running at near hardware limits. Rewriting this specific *vectorized* logic in Rust might yield a 2-3x speedup (due to better cache locality and avoiding Python object overhead between operations), but not a 100x speedup.
  - **When Rust actually wins**: Rust destroys Python/NumPy when you *cannot* vectorize the logic. For example, complex path-dependent state machines (like a trailing stop-loss that resets based on a dynamic condition) require `for` loops. A Python `for` loop over 100,000 rows is dead slow; a Rust `for` loop over 100,000 rows is instant.
- **Lecture Storyline**:
  1. Show the `joblib.Parallel` code block and explain why it's a bottleneck (Python loop overhead).
  2. Mark it as `[DEPRECATED]` in the notebook to show the evolution of the codebase.
  3. Introduce the `vectorized_backtest.py` module.
  4. Show the performance difference: what took hours now takes seconds per pair.
  5. **The Twist**: Run it on all 14,000 pairs and show the progress bar saying "10 hours left". Ask the students: "Wait, our code is blazing fast (4ms per backtest), why is it still so slow?"
  6. **The "Aha!" Moment**: Show the **[BAD APPROACH]** code block right next to the **[GOOD APPROACH]** code block in the notebook. Explain that the problem isn't the speed of the code, but the *number of times* we are running it (8 million vs 280,000).
  7. **Key Takeaway**: "Brute force doesn't scale, even in C." Teach the importance of pipeline ordering. Cointegration must act as a strict filter *before* parameter optimization.
  8. **The Rust Reality Check**: Explain that NumPy is already C. Use this to set the stage for *why* we use Rust later in the course (for complex state machines and live execution, not just array math).

### Screener vs. State Machine: Watchlist vs. Triggered Entry
- **The Problem**: If you run the Jupyter notebook daily and trade the top 5 pairs, your portfolio will churn constantly. A pair that was #1 yesterday might be #5 today just because its spread narrowed slightly. High turnover destroys accounts via transaction costs and slippage.
- **The Solution (Stateful Portfolio Management)**: Separate the system into Discovery, Monitoring, and Execution.
  1. **Phase 1: Discovery (The Notebook)**: Runs daily to find structurally sound pairs (Cointegrated). This creates a **"Watchlist"**, NOT a buy signal.
  2. **Phase 2: Monitoring (The Bot)**: Tracks the Watchlist in real-time. Waits for the **Entry Trigger** (e.g., Z-score > 2.0).
  3. **Phase 3: Execution & Allocation**: When a Watchlist pair hits the trigger, check if there is an empty slot (e.g., max 6 pairs). If yes, allocate a fixed amount (e.g., 1000 EUR) and enter.
- **Key Takeaway**: Cointegration means "this pair is worth watching." The Z-score spike is the actual "buy signal." Once entered, the pair is held until its specific exit condition (mean reversion or stop-loss) is met, regardless of its rank in tomorrow's notebook run. This prevents path dependence and eliminates unnecessary churn.

### Validation Design: Why 60 Days of 5m Data Is Not Enough
- **Visual**: Price chart showing sudden crashes (GFC 2008, COVID 2020) — if 60 days fall in a calm period, you never test crisis behavior at intraday resolution.
- **Key message**: Phase 1 (daily, 26yr) covers regime risk, but that's at daily granularity. You need intraday stress-testing too.
- **Solution**: Split Phase 2 into 2a (1h, 730 days — captures recent market events) and 2b (5m, 60 days — execution mechanics). This is driven by yfinance's interval-dependent history limits.

### Adaptive but Stable: Rolling Selection With Sticky Watchlist
- **Visual**: Two equity curves from the same universe:
  1) "Re-pick top pairs every month with immediate replacement"
  2) "Monthly rolling discovery + sticky watchlist + trigger-based entry"
- **Key message**: Adaptiveness is necessary, but without stickiness it becomes churn. A robust pipeline updates slowly at the watchlist layer and quickly at the trigger layer.
- **Lecture storyline**:
  1. Show why "global best pair over 30 years" is unstable in live deployment.
  2. Introduce monthly rolling Phase 1 (trailing 24-36 months) instead of full-history optimization.
  3. Separate Discovery (eligibility) from Monitoring/Execution (actual trade trigger).
  4. Add retention buffer/persistence bonus to avoid pair flip-flopping every rebalance.
  5. Conclude with slot-constrained portfolio simulation (e.g., max 7 positions) as the real validation target.
- **Anticipated student questions**:
  - "If adaptiveness is good, why not replace all pairs immediately each month?"
  - "How do we choose retention length without overfitting?"
  - "Is this still market-neutral if watchlist members persist across rebalances?"

---

## Risk Management & Execution

### Spread Blowout During Extreme Events
- **Case study**: FRT/KIM 2008 — same sector, but leverage/credit differences cause asymmetric crashes
- **Lesson**: Pairs trading bets on "spread will revert" → managing losses when it doesn't is the core challenge
- **Mitigations**: Cointegration verification, hedge ratio (β) adjustment, spread stop-loss

### Execution Risk: Slippage & Legging Risk
- **The Problem**: In backtests, orders execute instantly at the mid-price. In live trading, you must cross the Bid-Ask spread (Slippage). Worse, in pairs trading, one leg might execute while the other doesn't, leaving you with unhedged directional risk (**Legging Risk**).
- **Solution 1: IBKR Native Combo Orders (Best)**: Interactive Brokers has a specific order type called a "Spread" or "Combo" order (API `Bag` contract). You specify the *price difference* you want, and IBKR's engine guarantees both legs execute simultaneously or not at all. This completely eliminates Legging Risk.
- **Solution 2: Liquidity Filtering**: Only trade highly liquid stocks (e.g., S&P 500 constituents with >$10M daily volume). Wide bid-ask spreads mathematically destroy the edge of pairs trading.
- **Solution 3: Slippage Buffer in Signal**: If your entry trigger is Z-score > 2.0, calculate the expected slippage (Bid-Ask spread of both stocks). Only enter if the expected *post-slippage* Z-score is still > 1.8. (Note: The Phase 2b 5-minute data validation step in the pipeline specifically tests for this intraday slippage impact).

### Earnings Blackout Window — Avoid Known, Scheduled Risk

**Core idea**: Pairs trading is built on the assumption that the spread is stationary. Earnings announcements
break this assumption *temporarily but predictably*. Unlike the stop-loss (which reacts after the fact),
an earnings blackout avoids the loss proactively.

**Why the pair breaks around earnings**:
- Ticker A and B report on different days (typically days apart)
- On A's report day: +15% gap → z-score spikes → looks like a strong entry signal
- But this is NOT mean-reversion — A has genuinely re-rated at a new fundamental level
- The spread does not revert; the stop-loss fires after a 5% loss anyway
- **Lesson**: If you can predict the risk, avoid it — don't just react to it

**Proposed rule**:
```
blackout_start = min(earnings_A, earnings_B) - 2 days
blackout_end   = max(earnings_A, earnings_B) + 1 day
→ close any open position for this pair before blackout_start
→ skip all new entry signals within the window
→ apply cooldown = window bars after blackout_end (same logic as post-stop-loss)
```
Why 2 days before: implied volatility (IV) spikes in the lead-up; market makers widen spreads; slippage increases.

**How far apart are earnings dates within a pair?**
Same-sector pairs (which is what our correlation filter selects) report almost simultaneously:

| Pair | Sector | Closest quarterly gap | Example |
|---|---|---|---|
| MSFT / GOOGL | Tech (same Dec FY) | 0 days (same day) | MSFT 2026-01-28, GOOGL 2026-02-04 |
| JPM / BAC | Banks | 1 day | JPM 2026-01-13, BAC 2026-01-14 |
| MSFT / AAPL | Tech (AAPL Sep FY) | 1 day | MSFT 2026-01-28, AAPL 2026-01-29 |
| XOM / CVX | Energy | 0 days | Both 2026-01-30 |
| WMT / COST | Retail (diff FY) | 14 days | WMT 2026-02-19, COST 2026-03-05 |
| NKE / PVH | Apparel (NKE May FY) | 11 days | NKE 2025-12-18, PVH 2025-12-03 |

**Implication**: For high-correlation pairs (same sector, similar business), the typical quarterly gap is
0–2 days → blackout window is only about 5 days per quarter → **~20 trading days per year (~8%)**.
Edge cases with unusual fiscal year ends (WMT/COST, NKE/PVH, 11–14 day gaps) are likely already
filtered out by the correlation filter — different reporting rhythms produce weaker return correlations.

**Data sources**:

| Source | Cost | Coverage | Use case |
|---|---|---|---|
| `yfinance.Ticker.calendar` | Free | Next date only | Live trading |
| `yfinance.Ticker.earnings_dates` | Free | ~8 quarters back | Phase 2a/2b backtesting |
| Polygon.io / Nasdaq Data Link | ~$30/mo | Full history | Production-grade backtesting |
| SEC EDGAR (10-Q/10-K) | Free | Full history | Requires parsing |

**Relationship with stop-loss**:
- Earnings blackout = proactive (close before the event)
- Stop-loss = reactive (close after loss threshold is breached)
- Priority in execution: check blackout first → check stop-loss daily → after blackout ends, apply cooldown

**Implementation plan (TODO — next session)**:

Fetch earnings dates **once per pair** via `yfinance` REST (or LLM-assisted lookup as fallback),
cache the result so the live engine doesn't call the API on every bar.

```
Proposed flow:

  pair_defined (ticker_a, ticker_b)
       │
       ▼  fetch ONCE at pair registration time
  earnings_cache[(a, b)] = {
      "fetched_at": today,
      "dates_a": [...],   # next 4 quarters for ticker_a
      "dates_b": [...],   # next 4 quarters for ticker_b
  }
       │
       ▼  query cheaply on every bar (no API call)
  is_blacked_out(a, b, today) → bool
       │
       ▼  refresh only when stale
  if today > fetched_at + 90 days:
      re-fetch  (one quarter has passed, new dates available)
```

Why cache:
- yfinance `earnings_dates` is a network call; calling it on every signal check adds latency and may be rate-limited
- Earnings dates don't change — once fetched they're valid for ~90 days (one quarter)
- A simple JSON file or SQLite table per pair is sufficient for persistence across restarts

LLM fallback (if yfinance returns no data):
- Prompt: "What are the next 4 quarterly earnings dates for {ticker}? Return as JSON list of YYYY-MM-DD."
- Use for tickers with incomplete yfinance coverage (small-cap additions, recent IPOs)

Persistence across bot restarts:
- Store cache in `data/earnings_cache.json`
- Load on startup; refresh any entry older than 90 days before the trading session begins

**Status**: Pending implementation — design locked, implement next session.

---

### Walk-Forward Analysis (WFA) — The Real Backtest

**Core problem**: A static backtest (full-history grid search) sees the future. It optimises
parameters *after* the 2008 crisis, COVID crash, and every other event. This inflates returns
and hides overfitting.

**Solution**: Walk-Forward Analysis repeats the full Train → Validate → Execute cycle
rolling forward through time, so each turn never sees its own future.

```
Turn 1:  [── Phase 1 (2yr train) ──][P2b (1mo execute)]
Turn 2:       shift 3mo →  [── Phase 1 ──][P2b]
Turn 3:            shift 3mo →  [── Phase 1 ──][P2b]
...1998 → 2024...   (margin accumulates across turns)
```

**Design decisions** (locked):

| Decision | Choice | Rationale |
|---|---|---|
| Phase 1 re-run frequency | Quarterly | Cointegration relationships change slowly |
| Parameter selection | Stable region median (`df_sel` style) | Robust across market regimes |
| Slot capital allocation | Fixed `margin/n` at Phase 2b start | Prevents intra-month loss compounding |
| Max concurrent slots | 7 | Diversification without over-dilution |
| Cointegration cache | p-value margin (±0.02 from significance) | ~80% cache hit → ~4× speedup |

**Cointegration cache — why it works**:
When shifting Phase 1 by 3 months, 87% of the 2-year window overlaps with the prior turn.
Most pairs that were firmly cointegrated (p << 0.05) or firmly not (p >> 0.05) won't flip.
Only borderline pairs (0.03 < p < 0.07) need retesting. This cuts cointegration computation
by ~80% across all turns.

**Lecture flow**:
1. Show the static backtest result (e.g. "27% annual return — amazing!")
2. Run WFA on the exact same strategy → e.g. "12% annual return"
3. The 15%p gap = **the price of overfitting** (answer key vs. real exam)
4. "But 12% still beats the S&P 500 (~10%)!" → strategy is real, just less magical
5. Show the equity curve with drawdowns through 2008 and COVID

**Anticipated student questions**:
- "If 12% is good enough, why bother with static first?" → Static is fast screening;
  WFA is expensive but gives trustworthy numbers
- "Can we optimise WFA itself?" → Meta-overfitting trap — use fixed, sensible defaults

**Implementation**: `python/pairs_eda/rolling_phase2.py` (`run_phase2_rolling`)

---

### Portfolio-Level Circuit Breaker — Limiting Catastrophic Drawdowns

**Core problem**: Individual stop-losses limit per-trade risk, but multiple
sequential losses across different slots can accumulate into a severe portfolio
drawdown.  With 7 concurrent slots and 3× leverage, even a moderate per-trade
stop (5%) can compound into a -36% portfolio drawdown if several trades fail
within the same rebalance period.

**Solution**: A portfolio-level circuit breaker that monitors total equity
relative to the peak within each rebalance period.  When the drawdown exceeds
a threshold, ALL positions are liquidated and no new entries are allowed until
the next quarterly rebalance.

```
Equity peak tracking (per rebalance period):

  peak = max(peak, realized_equity)   ← updated daily
  dd   = (equity - peak) / peak

  if dd ≤ -circuit_breaker_pct:      ← e.g. -15%
      ┌─────────────────────────────────┐
      │  CIRCUIT BREAKER TRIGGERED      │
      │  • Close ALL open positions     │
      │  • Block new entries            │
      │  • Wait until next rebalance    │
      │  • Reset peak at rebalance      │
      └─────────────────────────────────┘
```

**Design decisions**:

| Decision | Choice | Rationale |
|---|---|---|
| Scope | Per rebalance period | Fresh start each quarter; avoids permanent shutdown |
| Threshold | 15% from peak | Aggressive enough to protect, loose enough to allow normal drawdowns |
| Action on trigger | Close all + block new entries | Decisive risk-off; half-measures leave residual exposure |
| Peak reset | At each rebalance boundary | Matches capital reallocation cycle |

**Relationship with other risk layers**:

```
Risk management stack (from narrowest to broadest):

  ┌─────────────────────────────────────────┐
  │  Per-trade stop-loss (5% + 2% slip)     │  ← single position
  ├─────────────────────────────────────────┤
  │  Post-stop cooldown (window bars)       │  ← single pair
  ├─────────────────────────────────────────┤
  │  Earnings blackout                      │  ← single pair, proactive
  ├─────────────────────────────────────────┤
  │  Volatility pre-filter (quantile)       │  ← universe level
  ├─────────────────────────────────────────┤
  │  ★ Portfolio circuit breaker (15% DD)   │  ← entire portfolio
  └─────────────────────────────────────────┘
```

**Lecture flow**:
1. Show the "before" equity curve with -46% max DD (baseline config, no circuit breaker)
2. Add `circuit_breaker_pct=0.15` → show reduced max DD
3. Discuss the trade-off: circuit breaker caps losses but also caps recovery
   (forced out of positions that might have reverted)
4. Show how many times the breaker fired over 30+ years — if it fires every
   quarter, the threshold is too tight; if it never fires, it's too loose
5. Ask students: "What threshold would you use for your personal capital?"

**Anticipated student questions**:
- "If we close everything at -15%, don't we lock in losses?" → Yes, but the
  alternative is risking -30% or worse.  The next rebalance gets a fresh start.
- "Why not resume trading in the same period after equity recovers?" → With all
  positions closed, there is no mechanism for recovery within the period.  Waiting
  for the next rebalance with fresh pair selection is cleaner.
- "Can we combine this with a trailing stop on the portfolio level?" → Yes, but
  adds complexity.  Start simple; the quarterly reset already provides a natural
  recovery mechanism.

**Implementation**: `RollingPhase2Config.circuit_breaker_pct` in
`python/pairs_eda/rolling_phase2.py`

---

### Entry Quality Gate — "No Good Pairs? Don't Trade."

**Core problem (discovered via WFA period analysis)**:

The strategy forces trades to fill all 7 slots whenever the z-score triggers,
regardless of the pair's quality score. During low-volatility bull markets
(2004-2008, 2016-2020), this leads to trading thin-spread pairs that produce
tiny wins but occasional stop-loss hits that wipe out months of small profits.

**Evidence from WFA period breakdown**:

| Period    | Annualized | Avg PnL/trade | Win Rate | Stop % | Diagnosis |
|-----------|-----------|---------------|----------|--------|-----------|
| 2004-2008 | +1.6%     | $15           | 66%      | 6%     | Wins too small |
| 2016-2020 | -1.6%     | $-7           | 63%      | 6%     | Net negative |

Both periods have LOW stop-loss rates (6%) — the problem is not large losses
but insufficient profit per winning trade. The strategy is churning through
mediocre pairs.

**Solution**: Add `min_entry_score` threshold. Pairs below this score are
skipped even when slots are available. The key insight:

> "Not trading is itself a position. When the opportunity set is poor,
> cash preservation beats forced deployment."

```
Entry logic (before):
  slot_available AND z_score_triggered → ENTER

Entry logic (after):
  slot_available AND z_score_triggered AND final_score >= min_entry_score → ENTER
  slot_available AND z_score_triggered AND final_score <  min_entry_score → SKIP
```

**Calibration**: `final_score` is the sum of `base_score` (surface evaluation
margin) + `persistence_bonus` (up to +0.15) - `turnover_penalty` (up to -0.10).
A threshold of 0.3-0.5 filters out the bottom ~30-50% of watchlist pairs.

**Lecture flow**:
1. Show the period breakdown table (2004-2008 at 1.6% annual, 2016-2020 at -1.6%)
2. Ask: "Win rate is 66%, so why are we losing money?"
3. Answer: "Because the average win ($15) is so small that one stop-loss ($2,000) erases 130 winning trades"
4. Show: "These periods have LOW volatility — spreads are tight, profit per trade is thin"
5. Solution: "Don't force all 7 slots to be filled. Trade only when the opportunity is genuinely good."
6. Compare before/after with `min_entry_score=0.4`

**Anticipated student questions**:
- "Won't we miss opportunities by sitting in cash?" → In low-vol regimes,
  those 'opportunities' were destroying value. Missing them is a feature.
- "How do we pick the threshold?" → Start with the median `final_score`
  from the watchlist. Adjust via WFA comparison.

**Implementation**: `RollingPhase2Config.min_entry_score` in
`python/pairs_eda/rolling_phase2.py`

---

### Sector Diversification Constraint — Preventing REIT Concentration

**Core problem (discovered via WFA trade analysis)**:

The worst-performing pairs in 2020-2024 were dominated by REIT tickers:
ARE, KIM, EXR, PLD, ESS, MAA, REG. These are all Real Estate sector stocks.
When REITs face a sector-wide shock (rising interest rates in 2022), ALL
pairs in the sector fail simultaneously — the "diversification" of 7 slots
provides zero protection because they are all correlated.

**Evidence from trade-level analysis (2022-2024)**:

| Pair    | Total PnL | Trades | Sector |
|---------|----------|--------|--------|
| ARE/DLR | -$27,092 | 14     | Real Estate |
| ARE/EQR | -$17,042 | 6      | Real Estate |
| AVB/EXR | -$16,815 | 10     | Real Estate |
| ARE/ESS | -$13,852 | 8      | Real Estate |
| DOC/O   | -$13,498 | 13     | Real Estate |

5 of the top 5 worst pairs are REITs. This is sector concentration, not
diversification.

**Solution**: Limit the number of open slots sharing the same GICS sector.
Each pair involves two tickers, both typically from the same sector (that is
why they are correlated). Count sector exposure per ticker across all open
positions and enforce a cap.

```
For each candidate pair (ticker_a, ticker_b):
  sector_a = GICS_sector(ticker_a)
  sector_b = GICS_sector(ticker_b)

  count sector exposure across all open positions:
    sector_counts[sector] = number of open tickers in that sector

  if sector_counts[sector_a] >= max_sector_slots * 2:  SKIP
  if sector_counts[sector_b] >= max_sector_slots * 2:  SKIP
  (×2 because each pair has 2 tickers)
```

**Data source**: GICS sector from Wikipedia S&P 500 table, fetched via
`fetch_sp500_sector_map()` at startup.

**Lecture flow**:
1. Show the top-10 worst pairs table for 2022-2024
2. Highlight: "5 of 5 are REITs — our 7 slots were a REIT portfolio"
3. Add `max_sector_slots=3` and re-run
4. Compare: sector-constrained vs unconstrained equity curve

**Implementation**: `RollingPhase2Config.max_sector_slots` +
`RollingPhase2Input.sector_map` in `python/pairs_eda/rolling_phase2.py`

---

### Minimum Spread Volatility — Skip Flat Spreads in Calm Markets

**Core problem (discovered via WFA period analysis)**:

During low-volatility bull markets (2004-2008, 2016-2020), the strategy
traded pairs whose spreads were too narrow to generate meaningful profit.
Win rate was decent (66%) but average PnL per winning trade was only $15.
A single stop-loss hit ($2,000) erased 130 winning trades.

**Root cause**: In calm markets, stock prices move in lockstep with very
small deviations. The z-score barely crosses the entry threshold, and when
it does, the profit from mean-reversion is minimal. The spread's volatility
is simply too low to support profitable trading at the given cost structure.

**Solution**: Before entering a position, compute the recent (60-day)
annualized volatility of the z-score series. If it is below a threshold,
skip the entry — the spread is too flat to trade profitably.

```
lookback = min(60, available_days)
recent_zscore = zscore[day - lookback : day]
ratio = price_a / price_b   (60-day window)
ratio_range = (max - min) / mean

if ratio_range < min_spread_range_pct:  SKIP
```

**Calibration**: A z-score with annualized vol < 3.0 means daily moves
of ~0.19σ — the spread barely fluctuates. At vol = 5.0, daily moves are
~0.31σ, enough for the entry/exit to capture a meaningful range.

**Lecture flow**:
1. Show the 2004-2008 period stats: 66% win rate, $15 avg profit
2. Ask: "If you win 2 out of 3 trades but only make $15 each, and you lose
   $2,000 on the third, are you profitable?" → No
3. Explain: the spread is too flat — z-score oscillates in a tiny band
4. Add `min_spread_range_pct=0.05` and show improvement

**Implementation**: `RollingPhase2Config.min_spread_range_pct` in
`python/pairs_eda/rolling_phase2.py`

**Note on prior approach**: An earlier version measured z-score volatility
(`std(zscore) × sqrt(252)`), but z-scores are standardized (std ≈ 1.0),
so annualized vol ≈ 15.87 — always above any reasonable threshold.
The price-ratio range is a direct, interpretable measure of spread tradability.

---

### Operational Risk: What Happens When a Ticker Gets Delisted Mid-Trade?
- **Scenario**: Bot is long A / short B. Today's pipeline run drops ticker A (delisted, no data, removed from S&P 500). Position is still open.
- **Real-world example**: SNDK (SanDisk) acquired by WDC in 2016 — ticker ceased to exist.
- **Options for the execution layer**:
  1. **Immediate market close**: Safest, but may realize a loss at worst possible moment
  2. **Stop-loss with grace period**: Set tight stop-loss, allow N days for orderly exit
  3. **Manual override**: Alert the operator, pause automation for this pair only
- **Key teaching point**: The notebook (strategy design) and the execution engine (live trading) have different responsibilities. The notebook finds pairs; the engine must handle events the notebook never anticipated.

---

## FX Risk Management for Non-USD Traders (Module in Course 1)

### The Core Problem
- Non-USD traders (EUR, KRW, etc.) face FX risk even with a profitable USD strategy
- EUR/USD annual swings of 8-10% are common. A +12% USD return with +10% EUR/USD appreciation → ~+2% EUR return.
- For non-USD traders, FX risk can be the dominant factor in whether the strategy is worth running.

### "Dollar-Neutral" — What It Actually Means
- **Positions (long + short) ARE FX-neutral**: If USD weakens, the long loses EUR value but the short obligation also decreases in EUR. They cancel perfectly.
- **Cash/margin is NOT FX-neutral**: Initial capital, unrealized P&L, and free margin sitting in USD are 100% exposed to FX.
- **Correct statement**: FX exposure = USD cash balance (margin + realized P&L), NOT total notional of positions.

### FX Hedging — Three Methods Compared

**Method A: Separate EUR/USD Forex Position (Spot Conversion)**
- Convert EUR to USD, or use API `hedgeType = "F"` to attach a child FX order.
- Cost: Two conversion costs (spread + commission) per round-trip trade. Eats into high-frequency pairs trading profits.

**Method B: IB Margin Loan — Natural Hedge (RECOMMENDED)**
- Do NOT convert EUR to USD. Keep EUR as base currency.
- Buy US stocks directly → IB auto-creates a negative USD balance (margin loan)
- Account state: [+EUR 1,000] + [-USD 1,000 loan] + [USD 1,000 stock]
- USD assets and USD liabilities cancel → principal has ZERO USD exposure
- **For pairs trading specifically:** Long creates -USD, Short creates +USD → they cancel → net USD cash ≈ 0 → almost zero margin loan interest.
- **Mathematical guarantee:** USD profit can NEVER become EUR loss. +USD 100 becomes +EUR 50 or +EUR 100 depending on rate, but never -EUR. Only the P&L amount is exposed to FX, not the principal.

**Method C: No Hedge**
- Convert EUR to USD and accept FX risk. Simplest, but 8-10% annual EUR/USD swings fully impact returns.

### Student Misconceptions to Address
1. **"I'll buy US stocks in EUR on Xetra — no FX risk, right?"** → Wrong. The EUR price on Xetra is simply (USD price × EUR/USD rate). You're 100% exposed. Only EUR-Hedged ETFs actually hedge; individual stocks never do.
2. **"I need to sell stocks to get USD to convert to EUR"** → No. Use Method B (margin loan) where you never convert in the first place.
3. **"If I profit in USD but EUR strengthens, I could lose money"** → With Method B (margin loan): mathematically impossible. Principal stays in EUR. Only the P&L is exposed, and a positive P&L in USD is always positive in EUR (just smaller).

---

## Performance Measurement & Metrics

### How to Calculate Returns with Continuous Deposits/Withdrawals
- **The Problem**: If you constantly add or withdraw money, calculating a simple return `(Current Balance - Total Deposits) / Total Deposits` creates massive distortions. (e.g., A bot makes +100% on $1k, you deposit $100k, bot loses 1%, your simple math says the bot is unprofitable).
- **The Solution: Time-Weighted Return (TWR)**: The institutional standard. It measures the pure performance of the trading algorithm, ignoring *when* or *how much* money the user deposited.
- **How it works**: Treat every day as a "sub-period". Calculate the daily percentage return *before* any cash flows are applied. Then, geometrically link (multiply) these daily returns together: `Cumulative Return = (1 + Day 1 Return) * (1 + Day 2 Return) * ... - 1`.

### Core Quantitative Metrics (The "Holy Trinity")
When evaluating a trading bot against an ETF, absolute return is not enough. You must measure the quality of the ride.

1. **Volatility (변동성)**
   - **What it is**: How wildly the portfolio's returns swing up and down. Mathematically, the standard deviation of daily returns, annualized.
   - **Why it matters**: High volatility means a roller coaster ride. Even if the final return is high, a highly volatile strategy is emotionally difficult to stick with and harder to leverage safely.
2. **Maximum Drawdown / MDD (최대 낙폭)**
   - **What it is**: The maximum observed loss from a historical peak to a trough before a new peak is attained.
   - **Why it matters**: It measures the worst-case scenario / pain tolerance. If a strategy has a 50% MDD, you must ask yourself: "Would I have panicked and turned off the bot when my account was cut in half?" Pairs trading aims for much lower MDDs than the S&P 500.
3. **Sharpe Ratio (샤프 지수)**
   - **What it is**: The ultimate measure of "Risk-Adjusted Return" (가성비). Formula: `(Strategy Return - Risk-Free Rate) / Volatility`.
   - **Why it matters**: It tells you how much excess return you are getting *per unit of risk* you take. A strategy that makes 20% with massive volatility might have a lower Sharpe Ratio than a strategy that makes 10% with almost no volatility. Institutional investors care more about a high Sharpe Ratio than high absolute returns, because a high Sharpe strategy can simply be leveraged up.

---

## Open Source & Community

### Phased Open Source Contribution Model
- **Phase 1 (launch)**: Code public, PRs not accepted yet. `CONTRIBUTING.md` says "coming soon".
- **Phase 2 (50+ students)**: Accept limited PRs via `good-first-issue` labels only (unit tests, docs, translations). No core logic changes.
- **Phase 3 (community formed)**: Promote 2-3 active students to reviewer role → share review burden.
- **Key selling point**: "Contribute to open source as part of the course" — real resume value for students.

### Course Positioning & Marketing Message
- **Target audience**: Engineers interested in investing (not traders learning to code)
- **Honest framing ("I'm not a professional quant, BUT...")**: 
  > "I am a software engineer, not a Wall Street quant. Professional quants teach complex stochastic calculus that retail investors can't execute due to infrastructure limits (fees, slippage, latency). I don't teach 'magic math formulas'. I teach you how to build a **robust, automated engineering system** that protects your capital. How do you handle API disconnections? How do you architect your system so FX risk doesn't eat your principal? How do you mathematically defend against slippage? This is an engineer's domain. This course isn't just about a strategy; it's about building an unbreakable trading infrastructure."
- **Differentiation**: Other courses say "make money with this strategy". This course says "build an engineering system for any strategy".
- **The Leverage Paradox (Aha! Moment)**: Retail investors think "Leverage = High Risk". Quants know that prime brokers only offer high leverage (e.g., 15% under Portfolio Margin) *because* the structural risk of a hedged pair is mathematically so low. The broker's willingness to lend you money is actual proof of the strategy's safety.
- **Core marketing message**:
  > Pairs trading typically involves leverage, which means significant risk.
  > Like any course, I cannot guarantee your returns — nor will I disclose my own.
  > What I CAN offer: I've deeply considered many ways to NOT lose money,
  > and automated them into the system.
- **Why this works**: Most individual investors fear large losses more than they desire gains (loss aversion). Each pipeline stage maps directly to a "don't lose money" safeguard.

## Course 3: AI-Driven Quant System Optimization (RAG & Multimodal)

### The "Post-Market Reverse Engineering" Concept
- **The Edge of Equity Markets (Time to Think)**: Unlike crypto which trades 24/7 in a chaotic, continuous loop, the US stock market closes. This daily "maintenance window" is a massive structural advantage for engineers. When a loss occurs today, you don't have to panic-fix a live system. You have 16 hours to analyze, reverse-engineer the failure, and deploy a fix before the market opens tomorrow.
- **Continuous Evolution vs. Set-and-Forget**: Most retail bots fail because the market regime changes, but the bot stays the same ("set-and-forget"). This course teaches a paradigm shift: **The bot is never finished.** Every single day, the RAG agent analyzes the day's trades, finds inefficiencies, and proposes code changes. Your system evolves and adapts daily, compounding its intelligence over time.
- **The RAG Application**: Instead of using AI to predict prices (which is highly prone to overfitting), we use AI to **debug and improve the trading system itself during this downtime**.
- **Workflow**:
  1. Market closes. Bot generates a daily report (PnL, slippage logs, failed pairs, spread charts).
  2. The AI Agent reads the report and identifies anomalies (e.g., "Pair A-B hit stop-loss due to massive slippage at the open").
  3. The Agent queries the **Quant RAG System**.

### Building the Multimodal Quant RAG
- **Text & Code Embedding**: The Vector DB is populated with classic quant textbooks, academic papers on pairs trading, and the project's own Python codebase.
- **The "Holy Trinity" of RAG Reference Books**: To make the AI agent truly intelligent, we embed these specific institutional textbooks:
  1. *Advances in Financial Machine Learning* (Marcos López de Prado): The bible for debugging why backtests fail in live trading (Overfitting, Lookahead Bias, Purged Cross-Validation).
  2. *Quantitative Equity Portfolio Management* (Chincarini & Kim): The ultimate reference for execution mechanics, FX margin hedging, and transaction cost modeling.
  3. *Statistical Arbitrage* (Andrew Pole): The mathematical foundation for pairs trading, cointegration, and half-life decay.
- **Image/Chart Embedding (Multimodal)**: Financial analysis is highly visual. We embed historical charts of "spread blowouts" or "successful mean reversions." When a trade fails today, the agent can search for visually similar historical failures in the DB to diagnose the structural issue.
- **Actionable Output**: The Agent doesn't just give advice. It finds the relevant textbook theory, locates the exact Python function in our codebase, and generates a Pull Request (e.g., "Based on Chapter 13 regarding liquidity buffering, I propose increasing the `min_volume_threshold` in `filters.py`. Here is the code update.").

### Why This is a Killer Course
- It bridges the hottest tech (Multimodal RAG, LLM Agents) with practical Quantitative Finance.
- It solves the "Black-box AI" problem. The AI isn't trading blindly; it is acting as a Junior Quant Researcher reading textbooks and suggesting logical code updates to the Senior Engineer (the student) for approval.

---

## Iterative Strategy Refinement via Loss-Period Analysis

### Core Concept

After running the full WFA simulation, identify months (or quarters) with the
largest drawdowns.  For each loss period:

1. **Diagnose** — What caused the loss?  Was it a specific pair blowing up, a
   regime shift (e.g. 2008 GFC, 2020 COVID), a sector rotation, or a failure of
   the z-score/window parameters?
2. **Decide** — Was the loss *unavoidable* (systemic event affecting all pairs)
   or *addressable* (poor pair selection, missing stop-loss, earnings event)?
3. **Update** — If addressable, formulate a new rule or filter (e.g. earnings
   blackout, tighter stop-loss, sector diversification constraint, volatility
   regime detector).
4. **Re-simulate** — Re-run the WFA from that month onwards with the updated
   strategy, keeping everything before that date untouched.
5. **Repeat** — Find the next large-loss period in the updated simulation and
   iterate.

This process is **not look-ahead bias** because:
- Each strategy update uses only information available up to the loss month.
- The updated strategy is tested forward, never backward.
- It mirrors how a real portfolio manager would evolve their system over decades.

### LLM Agent Role

This is where an LLM agent adds massive value:
- **Trade-level forensics**: Given a loss month, the agent retrieves the
  specific trades, their entry/exit z-scores, holding periods, and the
  underlying price paths.  It identifies whether the loss was from spread
  divergence, stop-loss cascade, or parameter mismatch.
- **News/event overlay**: The agent cross-references loss dates with major
  market events (rate decisions, earnings surprises, geopolitical shocks) to
  distinguish systematic vs. idiosyncratic losses.
- **Strategy suggestion**: Based on the diagnosis, the agent proposes concrete
  code changes — a new filter, an adjusted parameter range, or a new exit
  condition — with expected impact analysis.
- **Automated re-simulation**: The agent re-runs the WFA from the identified
  date and compares before/after metrics (Sharpe, max drawdown, cumulative
  return).

### Lecture Demo: Reproducing the "Before" State (Loss Scenario)

To show students that pairs trading CAN produce significant losses before
strategy refinement, use this baseline config.  This produces the unrefined
WFA results with visible drawdown periods (especially 2004-2008).

```python
# ── BASELINE CONFIG (before iterative refinement) ──
# Save this to reproduce the "before" equity curve in the lecture.
# Key characteristics:
#   - No sector diversification constraint
#   - No earnings blackout
#   - Basic volatility filter only (quantile-based)
#   - No regime detection
#   - Result: ~11.95% annualized (3x), but max DD -46%, worst month -23%

wfa_config_baseline = RollingPhase2Config(
    training_months=24,
    expanding_window=True,
    validation_days=63,
    rebalance_frequency="QS",
    min_correlation=0.40,
    max_correlation=0.85,
    min_overlap_years=1.5,
    recent_years=1.0,
    top_n_candidates=200,
    windows=(10, 15, 20, 30),
    zscore_thresholds=(2.0, 2.5, 3.0),
    watchlist_size=20,
    max_slots=7,
    leverage=3.0,
    max_drop_quantile=0.90,
    entry_zscore_default=2.0,
    exit_zscore=0.0,
    stop_loss_pct=0.05,
    commission_per_leg_bps=0.5,
    slippage_per_leg_bps=0.5,
)

# Results snapshot (3x leverage, 1992-2024):
#   Cumulative      : ~3526%
#   Annualized      : ~11.95%
#   Sharpe          : 0.75
#   Max drawdown    : -46.41%
#   Worst month     : -23.48%

# Leverage comparison table for the lecture:
# | Leverage | Annual | Sharpe | Max DD  | Worst Month |
# |----------|--------|--------|---------|-------------|
# | 1x       | 4.9%   | 0.89   | -15.7%  | -5.6%       |
# | 3x       | 11.95% | 0.75   | -46.4%  | -23.5%      |
# | 5x       | 18.5%  | 0.72   | -70.5%  | -39.7%      |
```

After RCA and strategy refinement, run the same WFA with improved config
to show the "after" equity curve side-by-side.

First refinement step: Add `circuit_breaker_pct=0.15` to the baseline config
and re-run.  This alone should visibly reduce max drawdown from -46% to a
capped ~15% per rebalance period, at the cost of slightly lower cumulative
returns (forced exits during temporary drawdowns).  This is the "before vs.
after circuit breaker" comparison — a powerful visual for the lecture.

### What "Strategy" Means (Far Beyond Parameters)

"Strategy" is not just `window` and `z_score`.  It is the full stack of
decisions that determine trade outcomes:

| Layer                    | Example rules                                   |
|--------------------------|------------------------------------------------|
| Universe filter          | Volatility pre-filter, min liquidity, GICS sector |
| Pair selection           | Correlation band, cointegration, overlap years  |
| Parameter selection      | Window, z-score threshold, robustness scoring   |
| Entry/exit rules         | Z-score trigger, mean-reversion exit            |
| Risk management          | Stop-loss %, cooldown period, max sector exposure|
| Event avoidance          | Earnings blackout, index rebalance dates        |
| Position sizing          | Equal weight, volatility-scaled, Kelly criterion|
| Portfolio constraints    | Max slots, max pairs per sector, leverage cap   |
| **Not yet discovered**   | Rules that emerge from loss-period analysis     |

Each iteration of the refinement cycle may touch ANY of these layers.

### Lecture Flow — Branching Equity Curves

```
          Strategy v1 (baseline)
          ─────────────────────────────────────── (gray, dashed)
         /
────────●─── 2008-10: worst month detected
         \
          Strategy v2 (+sector constraint)
          ──────────────────────────────────────── (blue)
                        /
               ────────●─── 2011-08: next loss period
                        \
                         Strategy v3 (+earnings blackout)
                         ────────────────────────── (green)
                                    ...
```

At each branch point:
1. **Compare** old vs. new equity curves side-by-side from that date forward
2. **Quantify** improvement: Sharpe delta, max-DD delta, cumulative return delta
3. **Validate** the new rule doesn't degrade performance in other periods
4. **Decide** whether to keep the rule (parsimony check)

### Key Teaching Points

- **Strategy evolution is continuous** — A trading system is never "done."
  Windows and z-scores are just the starting layer.  The system grows as we
  discover new failure modes.
- **Not all losses are fixable** — Systemic crashes (2008, 2020) affect all
  pairs simultaneously.  The goal is to *survive* them, not avoid them.
  Recognizing "this was unavoidable" is itself a valuable conclusion.
- **Over-fitting risk** — Each new rule added to fix a past loss must be
  validated on out-of-sample periods.  Adding too many rules creates a
  brittle system.  Emphasize parsimony: fewer rules that each cover broad
  failure modes beat many narrow rules.
- **Human + AI collaboration** — The student makes the final judgment on
  whether a rule change is justified; the LLM does the heavy analytical
  lifting and code generation.  The LLM proposes, the human disposes.

---

## Future Course: FX & Commodities (Macro Quant)

### Do Quant Techniques Work in FX?
- **Yes, but differently than equities**: The FX market is the most liquid in the world ($7.5 trillion daily), trading 24/5 OTC (Over-The-Counter) with no central exchange. It is heavily dominated by institutional algorithms, central banks, and macro events.
- **Why Traditional Pairs Trading is Harder**: In equities, you have 500+ stocks to find cointegrated pairs (thousands of combinations). In FX, there are only about 20-30 highly liquid currency pairs. Finding purely statistical, mean-reverting pairs within just currencies is difficult.
- **Classic FX Quant Strategies**:
  1. **Carry Trade**: Going long on high-interest-rate currencies and short on low-interest-rate currencies, capturing the yield differential while managing drawdown risk.
  2. **Trend Following / Momentum**: FX pairs tend to trend longer and harder than equities due to prolonged central bank policy cycles.
  3. **Volatility Trading**: Using options to trade the implied vs. realized volatility of currency pairs.

### Mixing FX and Commodities (Cross-Asset Quant)
- **The Connection**: FX and Commodities are deeply intertwined. In fact, on platforms like MetaTrader or IBKR, Gold (XAU) and Silver (XAG) are traded exactly like currencies (e.g., XAU/USD).
- **Commodity Currencies**: Certain currencies are structurally tied to commodity exports.
  - **AUD (Australian Dollar)** & **Gold/Iron Ore**
  - **CAD (Canadian Dollar)** & **Crude Oil**
  - **NZD (New Zealand Dollar)** & **Dairy/Agriculture**
- **Cross-Asset Statistical Arbitrage**: This is where quant pairs trading shines in FX. Instead of pairing two currencies, you pair a currency with a commodity. For example, if Crude Oil spikes but the CAD/JPY pair hasn't moved yet, a quant bot can exploit this temporary divergence (Lead-Lag relationship).
- **The "Safe Haven" Trade**: Modeling the relationship between USD, JPY, CHF, and Gold during risk-off market events.

---

## Advanced Topic: Market Regime Overlay (Live Workshop Material)

### The Core Insight from WFA Results

Our WFA simulation (1995-2026, 3x leverage, 5-year rolling window, monthly rebalance)
demonstrates a clear pattern:

| Metric | Pairs Trading | S&P 500 |
|--------|--------------|---------|
| Annualized | ~12.5% | ~10-11% |
| Sharpe | **0.96** | ~0.5-0.7 |
| Max Drawdown | **-23%** | **-55%** |
| Worst Month | -9.6% | -16.8% |
| 2000-2010 (lost decade) | Strong gains | ~0% |
| 2015-2025 (bull run) | Modest | Strong |

**Key takeaway**: Pairs trading is not about beating S&P 500 in bull markets.
It's about producing **institutional-quality risk-adjusted returns** (Sharpe ~1.0)
with dramatically lower drawdowns. The drawdown chart is the most compelling
visual — S&P reaches -55% while pairs stays within -23%.

### Why This Matters (Lecture Positioning)

1. **Psychological advantage**: Individual investors lose money because they buy high
   (FOMO) and sell low (panic). Pairs trading is fully systematic — emotions removed.
2. **Pension/endowment fit**: Institutions need steady returns, not home runs. Sharpe
   0.96 with -23% max DD is exactly what pension funds and endowments target.
3. **The max drawdown occurred early** in the simulation when the strategy was
   still calibrating. As the rolling window accumulates market regime experience,
   drawdowns shrink — a sign of genuine adaptive learning.

### Dynamic Beta Exposure ("Bull Market Tilt")

**Student question**: "Can we capture some bull market upside while staying mostly neutral?"

**Concept**: Detect market regime and slightly tilt long/short balance.

**Simple implementation**:
- Trend filter: S&P 500 price vs. 200-day moving average
- Bullish (price > 200MA): long leg 1.2x / short leg 0.8x
- Bearish (price < 200MA): stay fully neutral 1.0x / 1.0x

**Danger**: If 200MA signals "bull" but a crash hits next day (e.g., COVID Feb 2020),
the long bias amplifies losses. This is why it's an advanced topic, not the default.

**Verdict**: Keep the base strategy market-neutral. The regime overlay is optional
and should be presented as a research direction, not a recommendation.

### Real-Time Adaptive Strategy (The True Edge)

The WFA simulation uses ONE fixed parameter set across all periods. In real trading,
the investor has a crucial advantage: **real-time adaptation**.

When a loss period occurs:
1. Examine which pairs caused the loss and WHY
2. Check if the loss was structural (pair relationship broke) or temporary (shock event)
3. Adjust strategy parameters (windows, z-score thresholds, sector limits)
4. Resume with updated parameters from that point forward

This iterative refinement process — aided by LLM analysis of market news and
trade logs — is where pairs trading truly excels vs. passive investing.
A human-in-the-loop quantitative system can adapt to regime changes that no
single backtest can anticipate.

**Lecture flow**: Show the WFA equity curve → identify a loss period →
analyze it with the student → update strategy → re-simulate from that point →
compare before/after. This teaches the *process* of quantitative investing,
not just the *strategy*.

---

## FAQ: Anticipated Student Questions

### Q: "Can't I just buy S&P 500 with a stop-loss instead of pairs trading?"

**Short answer**: In theory yes, in practice the stop-loss approach has a fatal flaw
called **whipsaw** that pairs trading avoids structurally.

**The whipsaw problem**:
1. S&P drops -5% in one day → you sell
2. Next day it bounces +4% → you see "momentum confirmed" → you buy back
3. Day after, it drops -3% again → you sell again
4. Each cycle: you lock in a loss + pay transaction costs + miss part of the recovery
5. During COVID March 2020, this would have triggered multiple times in a single week

**Why "momentum confirmation" is the hardest part**:
- Define "confirmed": 3 consecutive up days? 10-day MA cross? There is no reliable signal.
- Enter too early → another drop. Enter too late → miss the strongest recovery days.
- Famous stat: **Missing the best 10 days over 20 years cuts S&P returns by more than half.**
  Those best days almost always occur right after the worst days.

**S&P -5% single-day drops are extremely rare**:
- Most crashes unfold as -2% to -3% over many consecutive days
- By the time you see -5% in one day, cumulative drawdown is already -15% to -20%
- The stop-loss fires too late to protect and too early to capture recovery

**Comparison table (show in lecture)**:

| | S&P 500 + Stop-Loss | Pairs Trading |
|---|---|---|
| Entry/exit decision | Subjective judgment each time | Fully systematic (z-score) |
| Whipsaw risk | Severe in volatile markets | None (pair-level management) |
| Psychological burden | High ("when do I re-enter?") | Low (automated execution) |
| Drawdown protection | Theoretically possible, practically hard | **Structurally built-in** |
| Works in flat markets | No returns while waiting | Mean-reversion still works |

**Lecture delivery**: Present as a 3-minute FAQ slide. Acknowledge the student's
intuition is reasonable, then show the whipsaw diagram. End with: "Pairs trading
doesn't need you to predict market direction — that's its structural advantage."

---

### Training Window Length: 3yr vs 5yr Trade-offs

**Why this matters**: The training window determines how much historical data the strategy
uses to find pairs and optimize parameters. This is one of the most impactful config choices.

**5-year rolling window**:
- More data for cointegration tests (statistical power)
- Better at detecting long-term structural relationships
- Risk: carries stale regime data — patterns from 2015 don't apply to post-COVID 2021
- Observed: underperformance in 2012-2016 (9% cumulative) and 2020-2024 (19% cumulative)
  because parameters trained on distant regimes failed to adapt

**3-year rolling window**:
- Faster adaptation to current market regime
- Drops stale pairs sooner when correlations break down
- Risk: less data for cointegration → more false positives
- Mitigation: the Phase 2a consistency gate catches pairs that pass cointegration
  but fail to produce consistent profit patterns

**Teaching point**: Show students both configurations side-by-side. The 5yr window
produces a smoother equity curve in stable decades (1995-2005) but stagnates when
regimes shift. The 3yr window is choppier but recovers faster from regime changes.
This is a microcosm of the bias-variance trade-off in all quantitative strategies.

### Circuit Breaker Tuning: 15% vs 10%

**15% threshold (original)**:
- Allows more room for temporary drawdowns to recover
- Risk: by the time it triggers during a crash, the damage is already severe
- During COVID (March 2020), equity could drop 15% before any positions are closed

**10% threshold (current)**:
- Triggers earlier, limiting tail risk
- Trade-off: may trigger during "normal" volatile periods, causing unnecessary exits
- The 5-day cooldown after trigger prevents immediate re-entry into a crashing market

**Teaching point**: Circuit breakers are a last-resort defense. The primary defense
should be good pair selection (consistency gate, cointegration), position sizing,
and per-trade stop losses. A tighter circuit breaker compensates for the reduced
statistical power of a shorter training window.

---

### Lecture Angle for the Next Course
- **Theme**: "Macro Quant: Trading the Global Machine"
- **Pedagogical Philosophy (Traditional First)**: Strictly focus on traditional, interpretable quant methods (Linear regression, Cointegration, Z-scores). Avoid deep learning or "black-box" AI. Students must first master market mechanics, execution infrastructure, and risk management using transparent "white-box" models where every trade's rationale can be mathematically proven and debugged.
- **Differentiation**: While the stock course teaches *Micro* relationships (Coca-Cola vs. Pepsi), the FX/Commodity course teaches *Macro* relationships (A nation's currency vs. its primary export). It introduces concepts like interest rate differentials (Carry) and cross-asset correlation.

---

### Sector De-meaning (섹터 평균 차감)

**Visual**: Show a chart of an Energy stock dropping 10% on a day the whole Energy sector drops 10%, compared to a Tech stock dropping 10% on a day the Tech sector is flat.
**Key message**: Not all volatility is created equal; we must distinguish between systemic (sector-wide) shocks and idiosyncratic (company-specific) shocks.
**Lecture storyline**:
1. Start with the basic volatility filter: `max(abs(return))`. Explain why we need to filter out extreme shocks (fraud, M&A).
2. Introduce the flaw: What if the whole market or sector crashes? We might accidentally filter out perfectly normal stocks just because their sector had a wild day.
3. Introduce the institutional solution: **Sector De-meaning**. We subtract the sector's daily average return from the stock's return: `abs(return - sector_mean)`.
4. Result: A stock that drops 10% alongside its sector has an idiosyncratic shock of 0%. It survives the filter. A stock that drops 10% alone has a shock of 10%. It gets filtered.
5. **The "Empty Cell" (NaN) Problem**: Explain that real market data is messy. What if a stock is halted on Tuesday? What if it's a holiday?
   - *Bad approach (For-loops)*: The code crashes because it tries to subtract a number from an empty cell.
   - *Good approach (Numpy Broadcasting)*: We use `np.nanmean` and `np.nanmax`. If a cell is empty, Numpy just ignores it and calculates the average of the remaining stocks. If the whole sector is empty (holiday), the shock is simply `NaN` (empty) for that day. It's blazing fast and mathematically safe.
### The Dual Nature of "Jumps" in Pairs Trading

**Visual**: A whiteboard split into two columns: "Training Phase (The Past)" vs. "Execution Phase (The Future)".
**Key message**: We must completely separate how we handle historical jumps (data pollution) from how we handle future jumps (portfolio risk).

**Lecture storyline**:
1. **The Execution Risk (The Future)**:
   - When we are holding a live position, a sudden 20% jump or drop in one leg is our worst nightmare. It blows out the spread and hits our stop-loss instantly.
   - *How do we protect against this?* Institutions spend millions on alternative data (real-time news sentiment, Twitter traffic, options IV) to predict jumps. (We will cover NLP and Twitter sentiment APIs in the Advanced AI course).
   - *Our Solution for this course*: The **Earnings Blackout Window**. It is a "naive" but incredibly powerful defense. By simply closing positions 2 days before earnings and ignoring signals until 1 day after, we eliminate the vast majority of predictable jump risk without needing complex alternative data.

2. **The Training Risk (The Past)**:
   - What if a stock had a 20% jump *a year ago*? Is it dangerous today?
   - *Counter-intuitive thought*: A stock that had a massive earnings surprise a year ago is actually LESS likely to have another one today. Lightning rarely strikes twice. So why do we filter them out?
   - *The real reason*: We filter them out because that historical 20% jump *permanently breaks our mathematical spread* in the training data. If we use a Simple Moving Average (SMA), a 20% jump creates a 60-day "ghost signal" period where the Z-score is artificially high, generating fake entry signals today even though the stock is peaceful.

3. **The Ultimate Institutional Solution (Kalman Filters)**:
   - *Student question*: "But aren't jumps a fundamental nature of stocks? If we just filter them out, aren't we ignoring reality?"
   - *Answer*: Yes! Filtering out the top 10% volatile stocks is a blunt heuristic. The true institutional approach is to fix the math, not throw away the stock.
   - Instead of SMA, we use a **Kalman Filter**. When a 20% jump occurs in the training data, the Kalman Filter mathematically detects a "regime change" and instantly resets the moving average to the new price level. It treats the post-jump spread as the **"New Normal"** rather than an anomaly. This completely eliminates the data pollution without having to discard perfectly good stocks. **We will implement this Kalman Filter directly in our Z-score calculation for this course.**

**Real-world Proof (COVID-19 Crash, Jan-Jun 2020)**:
We ran a simulation comparing the old "Raw" filter vs the new "Sector-Adjusted" filter on the top 100 S&P 500 stocks during the 2020 COVID crash. The results perfectly demonstrate the power of this institutional technique:
1. **The Threshold Dropped:** The 90th percentile cutoff for "extreme volatility" dropped from 23.8% (Raw) to 16.2% (Sector-Adjusted). By removing the sector's baseline panic, our definition of an "idiosyncratic shock" became much sharper.
2. **Saved from False Penalties (e.g., APA, COF):** APA (Energy) and COF (Financials) were dropped by the old filter because their prices collapsed. But the new filter saw that *their entire sectors* collapsed. It realized these stocks were just behaving normally for their sector and **saved them**.
3. **Caught Hidden Dangers (e.g., AMZN):** Amazon (Consumer Discretionary) survived the old filter because its absolute move didn't breach the massive 23.8% threshold. But the new filter caught it! Why? Because while the rest of the Consumer Discretionary sector was stagnant or dropping, Amazon surged as a "stay-at-home" winner. It moved *against* its sector, creating a massive idiosyncratic shock (16.2%+) that would have destroyed any pair it was part of. The new filter successfully dropped it.
