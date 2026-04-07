#!/bin/bash
# Run full Pairs Trading WFA + dashboard.
#
# Usage:
#   bash scripts/run_pairs_only.sh
#   tmux attach -t pairs
#   tmux kill-session -t pairs

set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate

SESSION="pairs"

tmux kill-session -t "$SESSION" 2>/dev/null || true

tmux new-session -d -s "$SESSION" -n dashboard \
    "source .venv/bin/activate && streamlit run dashboards/pairs_dashboard.py --server.port 8502 --server.headless true; read"

tmux new-window -t "$SESSION" -n backtest \
    "source .venv/bin/activate && python runners/run_pairs_only.py 2>&1 | tee docs/pairs-stdout.log; echo '--- DONE (exit: \$?) ---'; read"

echo "Started tmux session '$SESSION' with 2 windows:"
echo "  [0] dashboard  — Streamlit on :8502"
echo "  [1] backtest   — Pairs WFA runner"
echo ""
echo "  tmux attach -t $SESSION"
echo "  tmux kill-session -t $SESSION"
