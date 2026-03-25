# Quant Refactor Command (`/quant-refactor`)

Cleans up, documents, and organizes raw quant code into a production-ready, **human-readable** state.

---

## Usage
```
/quant-refactor <target_file_or_code_block>
```

Examples:
```
/quant-refactor Clean up python/pairs_eda/correlation.py — add Pydantic models
/quant-refactor Extract reusable helpers from notebook cell 12 into pairs_eda module
/quant-refactor Refactor the Z-score function — vectorize and add type hints
```

---

## Pipeline: Single Agent

```
Existing Code
    ↓
┌──────────────────────────────────┐
│  Agent: quant-synthesizer        │  (GPT Codex latest)
│  Mode: refactor (no dual draft)  │
│  Focus: readability, Pydantic,   │
│         hover-friendly docstrings│
│  Output: cleaned production code │
└──────────────────────────────────┘
```

## Execution Steps

### Step 0: Read Rules and Context (parallel reads)

| What | Where |
|------|-------|
| **Refactor rules (PRIMARY)** | `.cursor/rules/quant-refactor-rules.mdc` |
| Coding rules | `.cursor/rules/quant-coding-rules.mdc` |
| Agent instructions | `.cursor/agents/quant-synthesizer.md` |
| Skill (tools & templates) | `.cursor/skills/pairs-trading/SKILL.md` |
| Design decisions | `.cursor/rules/pairs-trading-pipeline.mdc` |

**Priority when rules conflict:** `quant-refactor-rules.mdc` > `quant-coding-rules.mdc`.
(e.g., refactor rules allow intermediate variables for readability; coding rules prefer vectorized one-liners.)

### Step 1: Read Target Code Fully

Read the entire target file or code block. Understand the current structure before changing anything.

### Step 2: Extract Functions

- Multi-line inline logic (≥5 lines) → extract into named function in `python/pairs_eda/`
- Import the function at the call site
- Notebook cells should read like a high-level outline

### Step 3: Enrich Docstrings

For every extracted/modified function:
- First 3-5 lines: one-line summary + ASCII formula/diagram (hover preview)
- `Flow:` section with ASCII pipeline diagram
- `Args:` with units, ranges, and defaults explained
- `Performance note:` with `PERF-NNN` tags for any readability-vs-speed trade-offs
- `Example:` with a 1-2 line usage snippet

### Step 4: Add Pydantic Models (where appropriate)

- Input parameters → `BaseModel` with `Field` validators
- Return values → `BaseModel` with structured fields
- Replace manual `if/raise ValueError` with Pydantic validation

### Step 5: Performance Trade-off Documentation

- Tag every trade-off: `PERF-NNN` in comments AND in the function's docstring
- Include what the optimized version would look like
- User can grep `PERF-` to find all optimization opportunities later

### Step 6: Verify

- [ ] All functions have hover-friendly docstrings with ASCII diagrams
- [ ] No math logic changed (flag suspicious math in comments only)
- [ ] All `PERF-NNN` tags documented in docstrings
- [ ] Notebook cells are scannable (imports + 1-3 function calls, not inline logic)
- [ ] Suggest `docs/lecture-ideas.md` update if code has teaching value

---

## Rules

**Primary:** `.cursor/rules/quant-refactor-rules.mdc` (readability-first, hover-friendly, PERF tags)

**Secondary:** `.cursor/rules/quant-coding-rules.mdc` (general coding constraints)
