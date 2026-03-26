"""
Vectorized Pairs Trading Backtest Utility

This module provides a highly optimized, vectorized implementation of a pairs trading backtest.
It uses NumPy and Pandas to efficiently calculate rolling statistics, Z-scores, and strategy returns
without relying on slow Python loops over time series data.
"""

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, field_validator
from typing import Literal, Tuple, Dict, Any, Optional

class PairsBacktestInput(BaseModel):
    """
    Validated input parameters for the vectorized pairs backtest.

    Attributes:
        price_a (pd.Series): Time series of prices for the first asset (Leg A). Must have a DatetimeIndex.
        price_b (pd.Series): Time series of prices for the second asset (Leg B). Must have a DatetimeIndex.
        window (int): The lookback window size for calculating rolling mean and standard deviation.
        zscore_threshold (float): The absolute Z-score value at which to enter a spread position.
            (e.g., if 2.0, enter short spread when Z > 2.0, enter long spread when Z < -2.0).
        exit_threshold (float): The Z-score value at which to exit an open position. Defaults to 0.0 (mean reversion).
        min_periods (Optional[int]): Minimum number of observations in window required to have a value. Defaults to `window`.
        std_epsilon (float): A small value to prevent division by zero. If rolling std is below this, Z-score is undefined.
        allow_flip (bool): If True, allows the strategy to directly flip from long to short (or vice versa) 
            if the opposite threshold is breached without crossing the exit threshold first.
    """
    model_config = {"arbitrary_types_allowed": True}
    
    price_a: pd.Series = Field(description="Series of prices for leg A (e.g. close), DatetimeIndex")
    price_b: pd.Series = Field(description="Series of prices for leg B, aligned index with A")
    window: int = Field(gt=1, description="Rolling window length (observations)")
    zscore_threshold: float = Field(gt=0.0, description="|z| above this enters a spread")
    exit_threshold: float = Field(default=0.0, description="Long spread exits when z >= this; short spread when z <= this (often 0)")
    min_periods: Optional[int] = Field(default=None, description="Rolling min_periods; default = window")
    std_epsilon: float = Field(default=1e-12, ge=0.0, description="Treat rolling std <= epsilon as undefined z (no trade)")
    allow_flip: bool = Field(default=True, description="If True, flip directly from long spread to short and vice versa on opposite signal")

    @field_validator("price_a", "price_b")
    @classmethod
    def _must_be_series(cls, v: pd.Series) -> pd.Series:
        if not isinstance(v, pd.Series):
            raise TypeError("price_a and price_b must be pandas Series")
        return v

    def model_post_init(self, __context: object) -> None:
        if self.min_periods is None:
            object.__setattr__(self, "min_periods", self.window)
        if self.min_periods > self.window:
            raise ValueError("min_periods cannot exceed window")
        if self.min_periods < 1:
            raise ValueError("min_periods must be >= 1")

class PairsBacktestOutput(BaseModel):
    """
    Structured output containing the results of the vectorized backtest.
    All Series arrays are aligned with the common index.

    Attributes:
        index (pd.Index): The common DatetimeIndex for all output series.
        spread (pd.Series): The log price spread: log(price_a / price_b).
        rolling_mean (pd.Series): The rolling mean of the spread.
        rolling_std (pd.Series): The rolling standard deviation of the spread.
        z_score (pd.Series): The calculated Z-score of the spread.
        position (pd.Series): The target position at the end of the bar (+1 for long spread, -1 for short spread, 0 for flat).
        position_for_pnl (pd.Series): The lagged position used to calculate returns (prevents look-ahead bias).
        log_ret_a (pd.Series): Log returns of asset A.
        log_ret_b (pd.Series): Log returns of asset B.
        spread_log_return (pd.Series): The log return of the spread (log_ret_a - log_ret_b).
        strategy_return (pd.Series): The realized strategy return for the bar.
        valid_price_mask (pd.Series): Boolean mask indicating where both prices were valid (positive and finite).
        metadata (dict): Additional summary statistics about the backtest run.
    """
    model_config = {"arbitrary_types_allowed": True}
    index: pd.Index
    spread: pd.Series
    rolling_mean: pd.Series
    rolling_std: pd.Series
    z_score: pd.Series
    position: pd.Series
    position_for_pnl: pd.Series
    log_ret_a: pd.Series
    log_ret_b: pd.Series
    spread_log_return: pd.Series
    strategy_return: pd.Series
    valid_price_mask: pd.Series
    metadata: Dict[str, Any]

def _prepare_prices(price_a: pd.Series, price_b: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Index]:
    """
    Aligns and cleans the input price series.
    Removes duplicates, finds the common index intersection, and strips timezones for stable comparison.
    """
    if price_a.index.inferred_type == "empty" or len(price_a) == 0:
        raise ValueError("price_a is empty")
    if len(price_b) == 0:
        raise ValueError("price_b is empty")
    
    a, b = price_a.copy(), price_b.copy()
    a.index, b.index = pd.to_datetime(a.index), pd.to_datetime(b.index)
    a, b = a.sort_index(), b.sort_index()
    
    if a.index.duplicated().any(): 
        a = a[~a.index.duplicated(keep="last")]
    if b.index.duplicated().any(): 
        b = b[~b.index.duplicated(keep="last")]
        
    common = a.index.intersection(b.index).sort_values()
    if len(common) == 0:
        raise ValueError("No overlapping index between price_a and price_b")
        
    a, b = a.reindex(common), b.reindex(common)
    
    def _strip_tz(s: pd.Series) -> pd.Series:
        return s.copy() if getattr(s.index, "tz", None) is not None else s
        
    return _strip_tz(a), _strip_tz(b), common

def run_grid_search_optimization(
    pairs: list[Tuple[str, str]], 
    prices_df: pd.DataFrame, 
    windows: list[int], 
    zscore_thresholds: list[float], 
    num_cores: int = -1,
    desc: str = "Optimizing Pairs"
) -> pd.DataFrame:
    """
    Runs a parallel grid search optimization over a list of pairs.
    
    Pipeline:
        pairs ──→ joblib.Parallel ──→ optimize_pair() ──→ best params ──→ concat DataFrame
    
    Args:
        pairs: List of tuple pairs, e.g., [('AAPL', 'MSFT'), ...].
        prices_df: DataFrame containing daily prices for all tickers.
        windows: List of rolling window sizes to test.
        zscore_thresholds: List of Z-score thresholds to test.
        num_cores: Number of CPU cores to use for parallel processing (-1 for all).
        desc: Description for the tqdm progress bar.
        
    Returns:
        A DataFrame containing the best parameters and total return for each pair.
        Returns an empty DataFrame if no pairs are processed successfully.
        
    Performance notes:
        PERF-001: Uses joblib.Parallel to distribute the grid search across CPU cores.
        The inner loop is fully vectorized via run_pairs_backtest_vectorized.
    """
    from joblib import Parallel, delayed
    import itertools
    from tqdm import tqdm
    
    def optimize_pair(pair):
        best_return = -np.inf
        best_params = {}
        
        # Extract and dropna
        price_a = pd.Series(prices_df[pair[0]].dropna())
        price_b = pd.Series(prices_df[pair[1]].dropna())
        
        if len(price_a) < max(windows) or len(price_b) < max(windows):
            return None
            
        for w, z in itertools.product(windows, zscore_thresholds):
            inp = PairsBacktestInput(
                price_a=price_a,
                price_b=price_b,
                window=w,
                zscore_threshold=z
            )
            out = run_pairs_backtest_vectorized(inp)
            total_ret = out.strategy_return.sum()
            
            if total_ret > best_return:
                best_return = total_ret
                best_params = {'window': w, 'zscore_threshold': z}
                
        if best_params:
            return pd.DataFrame([{
                'Asset1': pair[0],
                'Asset2': pair[1],
                'Best_Window': best_params['window'],
                'Best_Z_Threshold': best_params['zscore_threshold'],
                'Total_Return': best_return
            }])
        return None

    results = Parallel(n_jobs=num_cores)(
        delayed(optimize_pair)(pair) for pair in tqdm(pairs, desc=desc)
    )
    
    valid_results = [res for res in results if res is not None]
    if not valid_results:
        return pd.DataFrame()
        
    return pd.concat(valid_results, ignore_index=True)


def _safe_log_spread(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Calculates the log spread log(A/B) safely.
    Returns NaN where prices are invalid or non-positive.
    """
    out = np.full_like(a, np.nan, dtype=np.float64)
    valid = np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)
    out[valid] = np.log(a[valid] / b[valid])
    return out

def _positions_state_machine(z: np.ndarray, valid_z: np.ndarray, z_thr: float, exit_z: float, allow_flip: bool) -> np.ndarray:
    """
    O(n) state machine to determine positions based on Z-score thresholds.
    
    Rules:
    - If flat (0): Enter short (-1) if Z > z_thr. Enter long (+1) if Z < -z_thr.
    - If long (+1): Exit (0) if Z >= exit_z.
    - If short (-1): Exit (0) if Z <= exit_z.
    - Invalid Z-scores force a flat position (0) for safety.
    """
    n = len(z)
    pos = np.zeros(n, dtype=np.float64)
    for i in range(n):
        if not valid_z[i] or not np.isfinite(z[i]):
            pos[i] = 0.0
            continue
            
        zi = float(z[i])
        prev = float(pos[i - 1]) if i > 0 else 0.0
        
        if prev == 0.0:
            if zi > z_thr: pos[i] = -1.0
            elif zi < -z_thr: pos[i] = 1.0
            else: pos[i] = 0.0
        elif prev == 1.0:
            if zi >= exit_z: pos[i] = 0.0
            elif allow_flip and zi > z_thr: pos[i] = -1.0
            else: pos[i] = 1.0
        elif prev == -1.0:
            if zi <= exit_z: pos[i] = 0.0
            elif allow_flip and zi < -z_thr: pos[i] = 1.0
            else: pos[i] = -1.0
        else:
            pos[i] = 0.0
    return pos

def run_pairs_backtest_vectorized(inp: PairsBacktestInput) -> PairsBacktestOutput:
    """
    Executes a fully vectorized pairs trading backtest.
    
    This function avoids slow Python loops over time series data by utilizing
    NumPy array operations and Pandas rolling window functions. The only sequential
    part is the position state machine, which is optimized as a simple 1D array loop.
    
    Args:
        inp (PairsBacktestInput): Validated input parameters including prices and thresholds.
        
    Returns:
        PairsBacktestOutput: A structured output containing aligned series of spreads, 
                             Z-scores, positions, and strategy returns.
    """
    a, b, idx = _prepare_prices(inp.price_a, inp.price_b)
    pa, pb = a.to_numpy(dtype=np.float64, copy=False), b.to_numpy(dtype=np.float64, copy=False)
    
    valid_px = np.isfinite(pa) & np.isfinite(pb) & (pa > 0) & (pb > 0)
    spread_vals = _safe_log_spread(pa, pb)
    spread = pd.Series(spread_vals, index=idx, name="log_spread")
    
    # Vectorized rolling statistics
    rm = spread.rolling(window=inp.window, min_periods=inp.min_periods or inp.window).mean()
    rs = spread.rolling(window=inp.window, min_periods=inp.min_periods or inp.window).std(ddof=0)
    mu, sig = rm.to_numpy(dtype=np.float64, copy=False), rs.to_numpy(dtype=np.float64, copy=False)
    
    # Vectorized Z-score calculation
    z_vals = np.full_like(spread_vals, np.nan, dtype=np.float64)
    ok = np.isfinite(spread_vals) & np.isfinite(mu) & np.isfinite(sig) & (sig > inp.std_epsilon)
    np.divide(spread_vals - mu, sig, out=z_vals, where=ok)
    valid_z = valid_px & np.isfinite(z_vals)
    
    # Sequential position sizing (O(n) but fast)
    pos_arr = _positions_state_machine(z_vals, valid_z, inp.zscore_threshold, inp.exit_threshold, inp.allow_flip)
    position = pd.Series(pos_arr, index=idx, name="position")
    
    # Shift position to prevent look-ahead bias in returns
    position_for_pnl = position.shift(1)
    position_for_pnl.name = "position_for_pnl"
    
    # Vectorized log returns
    log_ra, log_rb = np.full_like(pa, np.nan), np.full_like(pb, np.nan)
    m_a = np.isfinite(pa[:-1]) & np.isfinite(pa[1:]) & (pa[:-1] > 0) & (pa[1:] > 0)
    log_ra[1:] = np.where(m_a, np.log(pa[1:] / pa[:-1]), np.nan)
    
    m_b = np.isfinite(pb[:-1]) & np.isfinite(pb[1:]) & (pb[:-1] > 0) & (pb[1:] > 0)
    log_rb[1:] = np.where(m_b, np.log(pb[1:] / pb[:-1]), np.nan)
    
    spread_ret = log_ra - log_rb
    
    # Vectorized strategy return
    strat = position_for_pnl.to_numpy(dtype=np.float64, copy=False) * spread_ret
    strat[~np.isfinite(spread_ret)] = np.nan
    
    meta = {
        "n_bars": len(idx), 
        "window": inp.window, 
        "bars_with_position": int(np.sum(pos_arr != 0))
    }
    
    return PairsBacktestOutput(
        index=idx, 
        spread=spread, 
        rolling_mean=rm, 
        rolling_std=rs, 
        z_score=pd.Series(z_vals, index=idx),
        position=position, 
        position_for_pnl=position_for_pnl, 
        log_ret_a=pd.Series(log_ra, index=idx),
        log_ret_b=pd.Series(log_rb, index=idx), 
        spread_log_return=pd.Series(spread_ret, index=idx),
        strategy_return=pd.Series(strat, index=idx), 
        valid_price_mask=pd.Series(valid_px, index=idx),
        metadata=meta
    )
