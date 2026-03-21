# pairs_eda

Small utilities for the stock pairs reference notebook.

## Install (editable, from repo root)

```bash
.venv/bin/pip install -e ./python
```

## S&P 500 + Exa fallback

- **Primary:** Wikipedia HTML (with a proper `User-Agent`).
- **Fallback:** inject an object that implements `Sp500ExaBackend`, with **separate code paths** for `ExaRunMode.LIVE` (web / unrestricted search) vs `ExaRunMode.SIMULATION` (KB-only / `init.simulation` — **do not** use open web search there).

Wire your `google-agents-api-gen` `exa_answer` / `exa_search` inside `create_exa_backend(fn_live=..., fn_simulation=...)`.
