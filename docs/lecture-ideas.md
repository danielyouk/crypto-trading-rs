# Lecture Ideas & Discussion Highlights

> Insights, visual examples, and teaching points collected during development.
> Each entry serves as raw material for lecture slides, live session topics, or notebook annotations.

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

## Pipeline Design

### Why Coarse Filter → Cointegration → Half-life
- **Key message**: Testing all 60,726 pairs for cointegration is too expensive. Returns correlation is a "coarse sieve" — confirms minimum co-movement, then cointegration does the real validation.
- **Numbers**: 60,726 → ~12,000 (corr filter) → ~200-500 (cointegration) → final pairs

---

## Risk Management

### Spread Blowout During Extreme Events
- **Case study**: FRT/KIM 2008 — same sector, but leverage/credit differences cause asymmetric crashes
- **Lesson**: Pairs trading bets on "spread will revert" → managing losses when it doesn't is the core challenge
- **Mitigations**: Cointegration verification, hedge ratio (β) adjustment, spread stop-loss

---

## Validation Design

### Why 60 Days of 5m Data Is Not Enough for Reassurance
- **Visual**: Price chart showing sudden crashes (GFC 2008, COVID 2020) — if 60 days fall in a calm period, you never test crisis behavior at intraday resolution.
- **Key message**: Phase 1 (daily, 26yr) covers regime risk, but that's at daily granularity. You need intraday stress-testing too.
- **Solution**: Split Phase 2 into 2a (1h, 730 days — captures recent market events) and 2b (5m, 60 days — execution mechanics). This is driven by yfinance's interval-dependent history limits.
- **Anticipated student question**: "Why not just use 5m for 2 years?" → yfinance only provides ~60 days of 5m data; 1h allows up to 730 days.

---

## Course Structure & Strategy

### Two-Course Plan (Independent, Self-Contained)
- **Course 1**: Python-based stock pairs trading (nearly ready to launch)
- **Course 2**: Rust-based crypto trading (Python+Rust blend via PyO3 for research, pure Rust for deployment)
- Platform: Udemy — each course must be independently complete (no prerequisite dependency)

### Development vs. Deployment Architecture (Course 2)
- **Development (laptop)**: Python + PyO3 → Jupyter for visualization, Rust crate for core logic
- **Deployment (VM)**: Same Rust crate compiled as pure binary → no Python runtime → ~10-30 MB vs. ~200-400 MB Python overhead
- **Key selling point**: "Rust isn't about speed for individuals — it's about safety and running 24/7 on a free VM"

### Course 2 Section Outline (~10 hours max)
1. Why Rust for Crypto Trading (2-3 lectures)
2. Strategy Design in Jupyter via PyO3 (5-6 lectures)
3. Live Trading Bot — pure Rust binary (5-6 lectures)
4. Oracle Free Tier VM Deployment (2-3 lectures)

### One Strategy Per Course vs. Mega-Course Trade-off
- **Option A: One strategy per short course (~10h)**
  - Pros: focused, maintainable flow, students finish the course
  - Cons: shared Rust/infra setup repeated across courses, risk of "splitting for profit" perception
- **Option B: Multiple strategies in one long course (30-60h)**
  - Pros: comprehensive, single purchase, better Udemy marketing (longer = higher perceived value)
  - Cons: hard to maintain narrative flow, high drop-off rate, harder to update individual sections
- **Mitigation for Option A**: Make the shared infrastructure portion a genuinely different angle each time (e.g., pairs trading focuses on websocket streaming for 2 assets; momentum trading focuses on order book depth for many assets). The infra isn't repeated — it's adapted to each strategy's unique requirements.

---

## Open Source & Community

### Phased Open Source Contribution Model
- **Phase 1 (launch)**: Code public, PRs not accepted yet. `CONTRIBUTING.md` says "coming soon".
- **Phase 2 (50+ students)**: Accept limited PRs via `good-first-issue` labels only (unit tests, docs, translations). No core logic changes.
- **Phase 3 (community formed)**: Promote 2-3 active students to reviewer role → share review burden.
- **Key selling point**: "Contribute to open source as part of the course" — real resume value for students.
- **Risk mitigation**: Control scope via labels; never accept PRs to core trading logic without your review.

### Course Positioning & Marketing Message
- **Target audience**: Engineers interested in investing (not traders learning to code)
- **Honest framing**: "I'm not a professional quant — I build systems that keep your strategy running 24/7"
- **Differentiation**: Other courses say "make money with this strategy". This course says "build an engineering system for any strategy".
- **Core marketing message**:
  > Pairs trading typically involves leverage, which means significant risk.
  > Like any course, I cannot guarantee your returns — nor will I disclose my own.
  > What I CAN offer: I've deeply considered many ways to NOT lose money,
  > and automated them into the system.
- **Why this works**: Most individual investors fear large losses more than they desire gains (loss aversion). Each pipeline stage (cointegration → hedge ratio → stop-loss) maps directly to a "don't lose money" safeguard — the lecture storyline (lines above) IS the marketing pitch.
- **Contrast with competitors**: Other courses hide behind disclaimers at the end. This course makes risk management the HEADLINE.

---

<!-- Add new ideas under the appropriate section above, or create a new section below -->
