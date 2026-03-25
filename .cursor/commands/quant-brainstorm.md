# Quant Brainstorm Command (`/quant-brainstorm`)

Initiates a deep architectural and conceptual discussion about quant trading strategies, risk management, and lecture material.

---

## Usage
```
/quant-brainstorm <topic_or_question>
```

Examples:
```
/quant-brainstorm How should we handle FX risk for European investors trading US stocks?
/quant-brainstorm What are the failure modes when a paired ticker gets delisted mid-trade?
/quant-brainstorm Compare grid trading vs. funding rate arbitrage for crypto lecture design.
```

---

## Pipeline: Single Agent

```
User Question
    ↓
┌──────────────────────────────────┐
│  Agent: quant-architect          │  (gemini-3.1-pro)
│  Focus: analysis + education     │
│  Output: insights + doc updates  │
└──────────────────────────────────┘
```

## Execution Steps

1. **Read the agent** — `.cursor/agents/quant-architect.md`
2. **Read the skill** — `.cursor/skills/pairs-trading/SKILL.md` (for domain context)
3. **Read lecture ideas** — `docs/lecture-ideas.md` (avoid duplicating existing content)
4. **Read the backlog** — `docs/pipeline-backlog.md` (understand current state)
5. **Execute** the three-step workflow defined in the agent:
   - Conceptual Analysis (with numerical examples)
   - Educational Translation (Aha! moments)
   - Documentation Update (proactive capture)

---

## Rules

See `.cursor/rules/quant-commands-guardrails.mdc` for cross-cutting constraints.
