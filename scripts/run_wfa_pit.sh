#!/bin/bash
# Run PIT-aware Hybrid Backtest + dashboard.
#
# Usage:
#   bash scripts/run_wfa_pit.sh
#   tmux attach -t wfa-pit
#   tmux kill-session -t wfa-pit

set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate

SESSION="wfa-pit"

tmux kill-session -t "$SESSION" 2>/dev/null || true

tmux new-session -d -s "$SESSION" -n dashboard \
    "source .venv/bin/activate && streamlit run dashboards/wfa_dashboard.py --server.port 8504 --server.headless true; read"

tmux new-window -t "$SESSION" -n backtest \
    "source .venv/bin/activate && python runners/run_wfa_pit.py 2>&1 | tee docs/wfa-pit-stdout.log; echo '--- DONE (exit: \$?) ---'; read"

echo "Started tmux session '$SESSION' with 2 windows:"
echo "  [0] dashboard  — Streamlit on :8504 (PIT overlay)"
echo "  [1] backtest   — PIT hybrid runner"
echo ""
echo "  tmux attach -t $SESSION"
echo "  tmux kill-session -t $SESSION"
echo ""
echo "Dashboard: http://localhost:8504"
