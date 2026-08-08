import json

nb_path = "reference/python_pairstrading/stock-trading-eda-scheduled_eng.ipynb"

with open(nb_path, "r") as f:
    nb = json.load(f)

last_cell = nb["cells"][-1]
new_source = []
for line in last_cell["source"]:
    if "spy = yf.download(\"SPY\"" in line:
        new_source.append('    # yfinance returns a MultiIndex when downloading even a single ticker in newer versions\n')
        new_source.append('    spy_df = yf.download("SPY", start=start_date, end=end_date, progress=False)\n')
        new_source.append('    # Fallback to Close if Adj Close is not available\n')
        new_source.append('    if "Adj Close" in spy_df.columns.get_level_values(0):\n')
        new_source.append('        spy = spy_df["Adj Close", "SPY"]\n')
        new_source.append('    else:\n')
        new_source.append('        spy = spy_df["Close", "SPY"]\n')
    else:
        new_source.append(line)

last_cell["source"] = new_source

with open(nb_path, "w") as f:
    json.dump(nb, f, indent=2)

