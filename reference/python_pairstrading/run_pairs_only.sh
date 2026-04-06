#!/bin/bash
# Run full pairs-only WFA + Streamlit dashboard in a tmux session.
# Does NOT interfere with the hybrid WFA (separate session + port 8502).
#
# Usage:
#   bash reference/python_pairstrading/run_pairs_only.sh    # start
#   tmux attach -t pairs                                     # reattach
#   tmux kill-session -t pairs                               # stop

set -e
cd "$(dirname "$0")/../.."
source .venv/bin/activate

SESSION="pairs"

# Kill existing session if any
tmux kill-session -t "$SESSION" 2>/dev/null || true

tmux new-session -d -s "$SESSION" -n dashboard \
    "source .venv/bin/activate && streamlit run reference/python_pairstrading/pairs_dashboard.py --server.port 8502 --server.headless true; read"

tmux new-window -t "$SESSION" -n backtest \
    "source .venv/bin/activate && python reference/python_pairstrading/run_pairs_only.py 2>&1 | tee docs/pairs-stdout.log; echo '--- DONE (exit: \$?) ---'; read"

echo "Started tmux session '$SESSION' with 2 windows:"
echo "  [0] dashboard  — Streamlit on :8502"
echo "  [1] backtest   — Full pairs WFA (log: docs/pairs-run.log)"
echo ""
echo "Commands:"
echo "  tmux attach -t $SESSION           # watch live output"
echo "  tmux attach -t $SESSION:backtest   # watch backtest only"
echo "  cat docs/pairs-run.log             # read log file"
echo "  tmux kill-session -t $SESSION      # stop everything"
echo ""
echo "Dashboard: http://localhost:8502"
