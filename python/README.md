# pairs_eda

Small utilities for the stock pairs reference notebook.

## What actually runs today

| Path | Needs API key? | Status |
|------|----------------|--------|
| **Wikipedia → `fetch_sp500_constituents_table(on_wiki_failure="raise")`** | No | **Works** after `pip install -e ./python`. |
| **Exa / LLM fallback** | Yes (whatever your `google-agents-api-gen` uses) | **Not implemented inside this repo.** Only **hooks** exist: you pass callables that call your other project. |

There is **no** `exa_search` / `exa_answer` source code in `crypto-trading-rs`, and **no** LLM HTTP calls in `pairs_eda`. That is intentional: this repo stays small and does not duplicate `google-agents-api-gen`.

So “how does Exa work?” → **It doesn’t, until you wire it.** Steps:

1. Keep `google-agents-api-gen` (or your Exa client) in a **separate checkout** or install path.
2. Put API keys in **`.env`** or your secret manager (see `.env.example` for placeholders — **`pairs_eda` does not load `.env` for you**).
3. In a notebook or a small local module, implement:
   - `fn_live()` → calls your **web-allowed** Exa entry point.
   - `fn_simulation()` → calls your **KB-only / `init.simulation`** entry point (no open web unless your KB policy allows it).
4. Pass them to `create_exa_backend(fn_live=..., fn_simulation=...)` and use `fetch_sp500_constituents_table(..., on_wiki_failure="exa", exa_backend=..., exa_mode=...)`.

If you want, add `python-dotenv` in the **notebook** `requirements.txt` and `load_dotenv()` at the top of the notebook before importing your Exa bridge.

## Install (editable, from repo root)

```bash
.venv/bin/pip install -e ./python
```

## S&P 500 + Exa fallback (design)

- **Primary:** Wikipedia HTML (with a proper `User-Agent`).
- **Fallback:** inject an object that implements `Sp500ExaBackend`, with **separate code paths** for `ExaRunMode.LIVE` vs `ExaRunMode.SIMULATION` (KB-only).

Wire your external `exa_answer` / `exa_search` inside `create_exa_backend(fn_live=..., fn_simulation=...)`.
