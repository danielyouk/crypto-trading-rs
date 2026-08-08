import yfinance as yf
import pandas as pd

spy = yf.download("SPY", start="2020-01-01", end="2020-01-10", progress=False)
print("Columns:", spy.columns)
print("Type of columns:", type(spy.columns))
