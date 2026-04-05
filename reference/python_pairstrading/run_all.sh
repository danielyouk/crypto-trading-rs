#!/bin/bash
# Run WFA backtest + Streamlit dashboard simultaneously.
# Usage: bash reference/python_pairstrading/run_all.sh

set -e
cd "$(dirname "$0")/../.."
source .venv/bin/activate

echo "Starting Streamlit dashboard on :8501..."
streamlit run reference/python_pairstrading/wfa_dashboard.py \
    --server.port 8501 \
    --server.headless true \
    &
STREAMLIT_PID=$!

echo "Starting WFA backtest..."
python reference/python_pairstrading/run_wfa.py
WFA_EXIT=$?

echo ""
echo "WFA finished (exit code: $WFA_EXIT)."
echo "Streamlit still running (PID: $STREAMLIT_PID) — press Ctrl+C to stop."
wait $STREAMLIT_PID
