import json

nb_path = "reference/python_pairstrading/stock-trading-eda-scheduled_eng.ipynb"

with open(nb_path, "r") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        source = "".join(cell["source"])
        if "wfa_config = RollingPhase2Config(" in source:
            new_source = source.replace(
                "    initial_capital=10_000.0,       # $10K equity, 3x leverage \u2192 $30K buying power\n",
                "    initial_capital=10_000.0,       # $10K equity, 3x leverage \u2192 $30K buying power\n"
                "    macro_regime_entry_dd=-0.10,    # SP500 drawdown threshold to switch TO Pairs Trading\n"
                "    macro_regime_exit_dd=-0.05,     # SP500 drawdown threshold to switch BACK to SP500\n"
            )
            cell["source"] = [line + "\n" for line in new_source.split("\n")][:-1] # preserve newlines

with open(nb_path, "w") as f:
    json.dump(nb, f, indent=2)

