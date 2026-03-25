---
name: pairs-trading
description: Concrete tools, templates, and recipes for the quant trading pipeline. Agents read this for usable code patterns and project context.
---

# Pairs Trading — Skill (Tools & Templates)

## Project Structure

```
crypto-trading-rs/
├── python/pairs_eda/            # Reusable modules
│   ├── correlation.py           # find_candidate_pairs()
│   └── yfinance_tools.py        # download_with_retry()
├── python/tests/                # pytest
├── reference/python_pairstrading/
│   └── stock-trading-eda-scheduled_eng.ipynb
├── docs/
│   ├── lecture-ideas.md
│   └── pipeline-backlog.md
└── .cursor/
    ├── commands/   → user entry points
    ├── agents/     → task instructions
    ├── skills/     → THIS: tools & templates
    └── rules/      → hard constraints
```

---

## Tool 1: Pydantic I/O Template

Use this pattern for every new function. Agent C (Synthesizer) enforces this.

```python
from pydantic import BaseModel, Field, field_validator


class MyFunctionInput(BaseModel):
    """Validated input for my_function()."""
    param_a: float = Field(ge=0.0, le=1.0, description="Correlation threshold")
    param_b: int = Field(gt=0, default=252, description="Trading days per year")

    @field_validator("param_a")
    @classmethod
    def check_param_a(cls, v: float) -> float:
        if v == 0.0:
            raise ValueError("param_a must be non-zero for meaningful results")
        return v


class MyFunctionOutput(BaseModel):
    """Structured output from my_function()."""
    result: dict[str, float]
    metadata: dict[str, int]

    model_config = {"arbitrary_types_allowed": True}


def my_function(inp: MyFunctionInput) -> MyFunctionOutput:
    """One-line summary.

    Args:
        inp: Validated input parameters.

    Returns:
        Structured output with result and metadata.

    Complexity:
        Time: O(n^2) for pairwise correlation.
        Space: O(n^2) correlation matrix.

    Failure modes:
        - Empty DataFrame → returns empty result.
        - All NaN column → excluded from correlation.
    """
    # ... vectorized implementation ...
    return MyFunctionOutput(result={}, metadata={})
```

---

## Tool 2: Vectorized Correlation Pattern

Current production implementation lives in `python/pairs_eda/correlation.py`.
See `references/correlation-api.md` for full parameter reference.

```python
# Fast pairwise correlation (no Python loops)
returns = prices.pct_change()                   # NaN on row 0
full_corr = returns.corr(min_periods=min_pd)    # pairwise deletion

# Upper triangle extraction (vectorized)
n = full_corr.shape[0]
mask = np.triu(np.ones((n, n), dtype=bool), k=1)
rows, cols = np.where(mask)

# Filter in bulk using numpy arrays
fc_values = full_corr.values[rows, cols]
valid = ~np.isnan(fc_values) & (fc_values >= min_corr) & (fc_values <= max_corr)
filtered_rows, filtered_cols = rows[valid], cols[valid]
```

---

## Tool 3: Download with Retry Pattern

Current production implementation lives in `python/pairs_eda/yfinance_tools.py`.

```python
import yfinance as yf
yf.set_tz_cache_location("/tmp/yf_cache")  # avoid sqlite3.OperationalError

def download_with_retry(
    tickers: list[str],
    max_retries: int = 2,
    retry_delay: float = 5.0,
    **yf_kwargs,
) -> pd.DataFrame:
    # First attempt: all tickers
    data = yf.download(tickers, **yf_kwargs)
    failed = [t for t in tickers if t not in data.columns.get_level_values(1)]

    for attempt in range(max_retries):
        if not failed:
            break
        time.sleep(retry_delay)
        retry_data = yf.download(failed, **yf_kwargs)
        # merge retry_data into data ...
        failed = [t for t in failed if t not in retry_data.columns.get_level_values(1)]

    return data
```

---

## Tool 4: Time-Series Safety Checklist

Apply before any rolling/iloc/loc operation:

```python
# 1. Sort index
df = df.sort_index()

# 2. Verify no duplicates
assert not df.index.duplicated().any(), "Duplicate timestamps"

# 3. Check minimum length
assert len(df) >= min_periods, f"Need {min_periods} rows, got {len(df)}"

# 4. NaN report
nan_pct = df.isna().mean()
if (nan_pct > 0.5).any():
    warn(f"Columns with >50% NaN: {nan_pct[nan_pct > 0.5].index.tolist()}")
```

---

## Tool 5: Test Template

```python
import pandas as pd
import numpy as np
import pytest

_NO_OVERLAP = {"min_overlap_years": 0, "recent_years": 0}


def _make_correlated_df(
    n_days: int = 500,
    corr: float = 0.8,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic price data with known correlation."""
    rng = np.random.default_rng(seed)
    base = rng.standard_normal(n_days).cumsum() + 100
    noise = rng.standard_normal(n_days) * (1 - corr)
    return pd.DataFrame(
        {"A": base, "B": base * corr + noise * base.std()},
        index=pd.bdate_range("2020-01-01", periods=n_days),
    )


class TestMyFunction:
    def test_happy_path(self):
        df = _make_correlated_df()
        result = my_function(MyFunctionInput(param_a=0.5))
        assert len(result.result) > 0

    def test_empty_input(self):
        df = pd.DataFrame()
        with pytest.raises(ValueError):
            my_function(MyFunctionInput(param_a=0.5))
```

---

## Tool 6: Optimization Recipes

### Recipe A: Replace loop with vectorized mask
```python
# SLOW
for i, row in df.iterrows():
    if row["zscore"] > 2.0:
        signals.append("SELL")

# FAST
signals = np.where(df["zscore"] > 2.0, "SELL", "HOLD")
```

### Recipe B: Rolling calculation without apply
```python
# SLOW
df["rolling_zscore"] = df["spread"].rolling(60).apply(lambda x: (x[-1] - x.mean()) / x.std())

# FAST
rolling_mean = df["spread"].rolling(60).mean()
rolling_std = df["spread"].rolling(60).std()
df["rolling_zscore"] = (df["spread"] - rolling_mean) / rolling_std
```

### Recipe C: Efficient pairwise operations
```python
# SLOW: double loop
for i in range(n):
    for j in range(i+1, n):
        corr_val = series.iloc[:, i].corr(series.iloc[:, j])

# FAST: matrix operation
corr_matrix = series.corr(min_periods=min_pd)
# extract upper triangle with np.triu
```

---

## Domain Quick Reference

| Concept | Formula / Detail |
|---------|-----------------|
| Returns correlation | `.pct_change().corr()` |
| Z-score | `(spread - rolling_mean) / rolling_std` |
| Min overlap | `int(years * 252)` trading days |
| Dual condition | Full-period AND recent-window both in [0.40, 0.85] |
| Cointegration | ADF test on spread residuals (pending implementation) |
| FX hedge size | Match cash/margin exposure, not notional. 1x leverage. |

---

## References

- `references/correlation-api.md` — `find_candidate_pairs()` full API docs
- `.cursor/rules/quant-coding-rules.mdc` — hard constraints
- `.cursor/rules/pairs-trading-pipeline.mdc` — design decisions
- `docs/pipeline-backlog.md` — current project state
