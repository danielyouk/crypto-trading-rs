#!/usr/bin/env python3
"""Download missing delisted stock data from EODHD and merge into parquet cache.

Usage:
    source .venv/bin/activate
    python scripts/download_eodhd_missing.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from pairs_eda.eodhd_download import download_missing_from_eodhd
from pairs_eda.sp500_history import Sp500History

API_KEY = "69d417f6cf0176.11169684"
CACHE_PATH = Path(__file__).resolve().parent.parent / "data" / "sp500_all_prices.parquet"
REPORT_PATH = Path(__file__).resolve().parent.parent / "data" / "eodhd_download_report.json"
# Note: paths already correct — runners/ is one level below repo root


def main():
    print("=" * 60)
    print("EODHD Missing Ticker Download")
    print("=" * 60)

    history = Sp500History.from_csv()
    all_tickers = history.all_unique_tickers()
    print(f"Total unique historical S&P 500 tickers: {len(all_tickers)}")

    existing = pd.read_parquet(CACHE_PATH)
    yahoo_tickers = set(existing.columns)
    print(f"Yahoo cache tickers: {len(yahoo_tickers)}")

    missing = sorted(all_tickers - yahoo_tickers)
    print(f"Missing tickers to download: {len(missing)}")
    print()

    if not missing:
        print("No missing tickers — cache is complete!")
        return

    print(f"Downloading {len(missing)} tickers from EODHD...")
    print(f"  API calls needed: ~{len(missing) * 2} (trying multiple symbol formats)")
    print(f"  Estimated time: ~{len(missing) * 0.3:.0f} seconds")
    print()

    eodhd_prices, ticker_map, still_missing = download_missing_from_eodhd(
        missing, api_key=API_KEY, verbose=True
    )

    report = {
        "total_missing_from_yahoo": len(missing),
        "found_on_eodhd": len(ticker_map),
        "still_missing": still_missing,
        "ticker_map": ticker_map,
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2))
    print(f"\nReport saved to {REPORT_PATH}")

    if eodhd_prices.empty:
        print("No data downloaded from EODHD — nothing to merge.")
        return

    print(f"\nMerging {eodhd_prices.shape[1]} EODHD tickers into cache...")

    combined = existing.join(eodhd_prices, how="outer")
    combined = combined.sort_index()

    all_nan_cols = combined.columns[combined.isna().all()]
    if len(all_nan_cols) > 0:
        combined = combined.drop(columns=all_nan_cols)
        print(f"  Dropped {len(all_nan_cols)} all-NaN columns")

    CACHE_PATH.rename(CACHE_PATH.with_suffix(".parquet.bak"))
    combined.to_parquet(CACHE_PATH)
    size_mb = CACHE_PATH.stat().st_size / 1024 / 1024

    print(f"\nFinal cache: {CACHE_PATH}")
    print(f"  Before: {existing.shape[1]} tickers, {existing.shape[0]} days")
    print(f"  After:  {combined.shape[1]} tickers, {combined.shape[0]} days")
    print(f"  Size:   {size_mb:.1f} MB")
    print(f"  New tickers added: {combined.shape[1] - existing.shape[1]}")

    if still_missing:
        print(f"\n⚠ {len(still_missing)} tickers not found anywhere:")
        for t in still_missing[:20]:
            print(f"    {t}")
        if len(still_missing) > 20:
            print(f"    ... and {len(still_missing) - 20} more")

    print("\nDone! You can now re-run the PIT backtest with complete data.")


if __name__ == "__main__":
    main()
