#!/bin/bash
# Run PIT (survivorship-bias-free) Pairs Trading WFA + comparison dashboard.
# Independent from the biased run (separate tmux session + port 8503).
#
# Usage:
#   bash reference/python_pairstrading/run_pairs_pit.sh    # start
#   tmux attach -t pairs-pit                                # reattach
#   tmux kill-session -t pairs-pit                          # stop

set -e
cd "$(dirname "$0")/../.."
source .venv/bin/activate

SESSION="pairs-pit"

# Kill existing session if any
tmux kill-session -t "$SESSION" 2>/dev/null || true

tmux new-session -d -s "$SESSION" -n dashboard \
    "source .venv/bin/activate && streamlit run reference/python_pairstrading/pairs_pit_dashboard.py --server.port 8503 --server.headless true; read"

tmux new-window -t "$SESSION" -n backtest \
    "source .venv/bin/activate && python reference/python_pairstrading/run_pairs_pit.py 2>&1 | tee docs/pairs-pit-stdout.log; echo '--- DONE (exit: \$?) ---'; read"

echo "Started tmux session '$SESSION' with 2 windows:"
echo "  [0] dashboard  — Streamlit on :8503 (PIT comparison)"
echo "  [1] backtest   — PIT pairs WFA (log: docs/pairs-pit-run.log)"
echo ""
echo "Commands:"
echo "  tmux attach -t $SESSION              # watch live output"
echo "  tmux attach -t $SESSION:backtest     # watch backtest only"
echo "  cat docs/pairs-pit-run.log           # read log file"
echo "  tmux kill-session -t $SESSION        # stop everything"
echo ""
echo "Dashboard: http://localhost:8503"
echo ""
echo "NOTE: First run downloads ~1,100 tickers from Yahoo Finance (~10 min)."
echo "      Subsequent runs load from parquet cache (< 2 seconds)."
