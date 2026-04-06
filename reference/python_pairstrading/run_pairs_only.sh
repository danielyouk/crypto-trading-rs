#!/bin/bash
# Run full pairs-only WFA in a tmux session.
# Does NOT interfere with the hybrid WFA (separate session + ports).
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

tmux new-session -d -s "$SESSION" -n backtest \
    "source .venv/bin/activate && python reference/python_pairstrading/run_pairs_only.py 2>&1 | tee docs/pairs-stdout.log; echo '--- DONE (exit: $?) ---'; read"

echo "Started tmux session '$SESSION':"
echo "  [0] backtest — Full pairs WFA (log: docs/pairs-run.log)"
echo ""
echo "Commands:"
echo "  tmux attach -t $SESSION    # watch live output"
echo "  cat docs/pairs-run.log     # read log file"
echo "  tmux kill-session -t $SESSION  # stop"
echo ""
echo "Progress: docs/pairs-progress.json"
