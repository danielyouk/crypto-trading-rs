# Quant Coding Command (`/quant-coding`)

Orchestrates a 3-agent pipeline to generate production-ready quant trading code.

---

## Usage
```
/quant-coding <task_description>
```

Examples:
```
/quant-coding Calculate the Z-score of a spread and generate a trading signal.
/quant-coding Implement cointegration test (ADF) for candidate pairs.
/quant-coding Build FX hedge rebalancer that monitors IB margin and adjusts EUR/USD position.
```

---

## Pipeline: Parallel Agents → Synthesizer

```
                    User Task
                        ↓
                 Context Gathering
                   (read skill, rules, source files)
                        ↓
          ┌─────────────┴─────────────┐
          ↓                           ↓
┌───────────────────────┐   ┌───────────────────────┐
│  Agent A (PARALLEL)   │   │  Agent B (PARALLEL)   │
│  quant-perf-optimizer │   │  quant-risk-manager   │
│  Claude latest        │   │  composer-2           │
│  → <draft_a>          │   │  → <draft_b>          │
└───────────────────────┘   └───────────────────────┘
          │                           │
          └─────────────┬─────────────┘
                        ↓
          ┌───────────────────────────┐
          │  Agent C (AFTER A+B)      │
          │  quant-synthesizer        │
          │  GPT Codex latest         │
          │  Merges <draft_a>+<draft_b>│
          │  + Pydantic I/O models    │
          │  + final optimization     │
          │  → production-ready code  │
          └───────────────────────────┘
                        ↓
                 Documentation update
```

**Key: Agent A and Agent B are independent. Launch them in parallel.**

---

## Execution Steps

### Step 1: Context Gathering

Before invoking any agent, read (all in parallel):

| What | Where |
|------|-------|
| Skill (tools & templates) | `.cursor/skills/pairs-trading/SKILL.md` |
| Rules (constraints) | `.cursor/rules/quant-coding-rules.mdc` |
| Design decisions | `.cursor/rules/pairs-trading-pipeline.mdc` |
| Backlog | `docs/pipeline-backlog.md` |
| Relevant source files | `python/pairs_eda/`, `python/tests/`, notebook |

### Step 2: Agent A + Agent B — PARALLEL

Launch both agents **simultaneously** in a single tool call batch:

- **Agent A** (`quant-perf-optimizer`): Read `.cursor/agents/quant-perf-optimizer.md`. Produce `<draft_a>`.
- **Agent B** (`quant-risk-manager`): Read `.cursor/agents/quant-risk-manager.md`. Produce `<draft_b>`.

Both receive the same task description and context. Neither depends on the other.

### Step 3: Agent C — Synthesis (AFTER both A and B complete)

Read `.cursor/agents/quant-synthesizer.md`. Feed both `<draft_a>` and `<draft_b>` as input.

Produce:
- Pydantic `BaseModel` for inputs and outputs
- Merged, optimized implementation
- Synthesis notes explaining merge decisions
- Suggested tests (≥ 2: happy path + edge case)

### Step 4: Documentation

- If the task completes a backlog item → update `docs/pipeline-backlog.md`
- If the code has teaching value → suggest `docs/lecture-ideas.md` update

---

## Rules

See `.cursor/rules/quant-coding-rules.mdc` for must-follow constraints.
