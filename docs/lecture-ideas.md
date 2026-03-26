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

### Course Lineup (Independent, Self-Contained)
- **Course 1**: Python stock pairs trading (S&P 500) — nearly ready to launch
- **Course 2**: Python FX risk management for non-USD traders (2-3 hours) — FX-adjusted returns, IB multi-currency API, automated event-driven/daily hedge rebalancing. Prerequisite: Course 1.
- **Course 3**: Rust grid trading bot (crypto) — Rust entry point. Ownership, types, async through a simple strategy. One exchange, one asset, clear state machine.
- **Course 4**: Rust funding rate arbitrage (crypto) — the serious Rust project. Multi-exchange delta-neutral hedging (spot + futures), websocket streaming, state persistence, error recovery. Assumes Rust basics from Course 3.
- **Course 5**: Rust cross-exchange arbitrage (crypto) — where Rust's speed genuinely matters. Concurrent connections, latency optimization.
- **Course 6 (Future)**: Python/Rust FX & Commodities (Macro Quant) — Cross-asset statistical arbitrage (e.g., CAD vs. Crude Oil), carry trades, and macro trend following.
- Platform: Udemy — each course ~10 hours max, independently complete (except Course 2 which extends Course 1)

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

---

## Pipeline Design & Architecture

### Why Coarse Filter → Cointegration → Half-life
- **Key message**: Testing all 60,726 pairs for cointegration is too expensive. Returns correlation is a "coarse sieve" — confirms minimum co-movement, then cointegration does the real validation.
- **Numbers**: 60,726 → ~12,000 (corr filter) → ~200-500 (cointegration) → final pairs

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

### Operational Risk: What Happens When a Ticker Gets Delisted Mid-Trade?
- **Scenario**: Bot is long A / short B. Today's pipeline run drops ticker A (delisted, no data, removed from S&P 500). Position is still open.
- **Real-world example**: SNDK (SanDisk) acquired by WDC in 2016 — ticker ceased to exist.
- **Options for the execution layer**:
  1. **Immediate market close**: Safest, but may realize a loss at worst possible moment
  2. **Stop-loss with grace period**: Set tight stop-loss, allow N days for orderly exit
  3. **Manual override**: Alert the operator, pause automation for this pair only
- **Key teaching point**: The notebook (strategy design) and the execution engine (live trading) have different responsibilities. The notebook finds pairs; the engine must handle events the notebook never anticipated.

---

## FX Risk Management for Non-USD Traders (Course 2)

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

### Lecture Angle for the Next Course
- **Theme**: "Macro Quant: Trading the Global Machine"
- **Differentiation**: While the stock course teaches *Micro* relationships (Coca-Cola vs. Pepsi), the FX/Commodity course teaches *Macro* relationships (A nation's currency vs. its primary export). It introduces concepts like interest rate differentials (Carry) and cross-asset correlation.
