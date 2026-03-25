# `find_candidate_pairs()` — Parameter Reference

Source: `python/pairs_eda/correlation.py`

## Signature

```python
def find_candidate_pairs(
    data: pd.DataFrame,
    *,
    start: Optional[datetime | pd.Timestamp] = None,
    end: Optional[datetime | pd.Timestamp] = None,
    top_n: Optional[int] = None,
    min_correlation: float = 0.40,
    max_correlation: float = 0.85,
    use_returns: bool = True,
    min_overlap_years: float = 5.0,
    recent_years: float = 3.0,
) -> dict[tuple[str, str], float]:
```

## Parameters

| Parameter | Default | Validation | Purpose |
|-----------|---------|------------|---------|
| `data` | — | shape[1] >= 2 | DataFrame: dates × tickers, Adj Close prices |
| `start` | None | — | Slice start (inclusive). None = data.index.min() |
| `end` | None | — | Slice end (inclusive). None = data.index.max() |
| `top_n` | None | >= 0 or None | Cap on returned pairs. None = no cap |
| `min_correlation` | 0.40 | [-1, 1] | Lower bound (inclusive) |
| `max_correlation` | 0.85 | [-1, 1] | Upper bound — excludes structurally identical pairs |
| `use_returns` | True | — | True = .pct_change().corr(); False = price-level corr |
| `min_overlap_years` | 5.0 | >= 0 | Minimum overlapping trading days = years × 252 |
| `recent_years` | 3.0 | >= 0 | Recency window; 0 = disable dual condition |

## Key Implementation Details

1. **Index sorted** before any operation: `sliced.sort_index()`
2. **`min_periods`** = `int(min_overlap_years * 252)` — pairs with fewer overlap days → NaN → excluded
3. **Dual condition**: both full-period and recent-window correlations must fall in `[min, max]`
4. **Recent min_periods**: `min(min_periods, len(recent_series))` — avoids excluding everything in short windows
5. **Return format**: `dict[tuple[str, str], float]` ordered by full-period correlation descending

## Test Patterns

```python
_NO_OVERLAP = {"min_overlap_years": 0, "recent_years": 0}

# Disable time filters for unit tests:
result = find_candidate_pairs(df, min_correlation=0.5, **_NO_OVERLAP)

# Test overlap filter:
result = find_candidate_pairs(df, min_overlap_years=10)  # should exclude short-history pairs

# Test dual condition:
result = find_candidate_pairs(df, recent_years=1.0)  # pair must pass in last 252 days too
```
