#!/bin/bash
# Run PIT (survivorship-bias-free) Pairs Trading WFA + comparison dashboard.
#
# Usage:
#   bash scripts/run_pairs_pit.sh
#   tmux attach -t pairs-pit
#   tmux kill-session -t pairs-pit

set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate

SESSION="pairs-pit"

tmux kill-session -t "$SESSION" 2>/dev/null || true

tmux new-session -d -s "$SESSION" -n dashboard \
    "source .venv/bin/activate && streamlit run dashboards/pairs_pit_dashboard.py --server.port 8503 --server.headless true; read"

tmux new-window -t "$SESSION" -n backtest \
    "source .venv/bin/activate && python runners/run_pairs_pit.py 2>&1 | tee docs/pairs-pit-stdout.log; echo '--- DONE (exit: \$?) ---'; read"

echo "Started tmux session '$SESSION' with 2 windows:"
echo "  [0] dashboard  — Streamlit on :8503 (PIT comparison)"
echo "  [1] backtest   — PIT pairs WFA"
echo ""
echo "  tmux attach -t $SESSION"
echo "  tmux kill-session -t $SESSION"
echo ""
echo "Dashboard: http://localhost:8503"
