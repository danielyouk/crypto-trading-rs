#!/bin/bash
# Run WFA backtest + Streamlit dashboard in a tmux session.
# The session survives terminal disconnects and SSH drops.
#
# Usage:
#   bash reference/python_pairstrading/run_all.sh        # start
#   tmux attach -t wfa                                    # reattach
#   tmux kill-session -t wfa                              # stop

set -e
cd "$(dirname "$0")/../.."
source .venv/bin/activate

SESSION="wfa"

# Kill existing session if any
tmux kill-session -t "$SESSION" 2>/dev/null || true

tmux new-session -d -s "$SESSION" -n dashboard \
    "source .venv/bin/activate && streamlit run reference/python_pairstrading/wfa_dashboard.py --server.port 8501 --server.headless true; read"

tmux new-window -t "$SESSION" -n backtest \
    "source .venv/bin/activate && python reference/python_pairstrading/run_wfa.py 2>&1 | tee docs/wfa-stdout.log; echo '--- DONE (exit: $?) ---'; read"

echo "Started tmux session '$SESSION' with 2 windows:"
echo "  [0] dashboard  — Streamlit on :8501"
echo "  [1] backtest   — WFA runner (log: docs/wfa-run.log)"
echo ""
echo "Commands:"
echo "  tmux attach -t $SESSION          # watch live output"
echo "  tmux attach -t $SESSION:backtest  # watch backtest only"
echo "  cat docs/wfa-run.log             # read log file"
echo "  tmux kill-session -t $SESSION    # stop everything"
