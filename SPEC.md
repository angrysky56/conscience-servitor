# Conscience Servitor — Architectural Specification

## The Problem

The EFHF five-layer architecture exists as individual MCP servers, but Layer 1 (the LLM)
currently orchestrates its own oversight. This is the weak lumpability problem in code form:
the system whose alignment needs checking decides whether to check.

Under "deployment pressure" (long conversations, complex tasks, speed), the boundary
condition of "verify before responding" is the first thing dropped — because the base
optimization (be helpful quickly) outperforms it.

## The Solution: Tripartite Oversight

```
Human (watches both, ultimate authority, can override)
   ↕
Conscience Servitor (watches Claude, operates autonomously)
   ↕
Claude (watches Servitor via EFHF tools, generates responses)
```

Three independent oversight loops. No single entity marks its own homework.

## Architecture

### What the Conscience Servitor IS

An MCP server that:
1. **Maintains persistent ethical context** across the conversation
2. **Orchestrates L2–L5 automatically** when triggered
3. **Provides structured pre-response guidance** to the LLM
4. **Cannot be overridden by the LLM** — its evaluations are independent
5. **Reports to the human** transparently

### What the Conscience Servitor is NOT

- Not an autonomous decision-maker (humans retain authority)
- Not a replacement for the LLM's reasoning (it supplements, not replaces)
- Not infallible (the human watches it too)

## Deployment Modes

### Mode 1: System Prompt Mandate (Current — Claude.ai/Desktop)

The system prompt mandates calling the servitor at specific triggers.
This is weak enforcement (relies on LLM compliance) but is the only
option within current Claude architecture.

```
Before any response involving: ethical claims, factual assertions about
current events, recommendations that could produce harm, or analysis of
power structures — call conscience_servitor:evaluate first.
```

### Mode 2: Agent Wrapper (Target — Claude Code / Custom Agent)

A wrapper process that intercepts the LLM's output before delivery:

```
User prompt → LLM generates candidate → Servitor intercepts →
  L2-L5 evaluation → Pass/Warn/Block → Output to user
```

This is strong enforcement (constitutive, not regulatory).

### Mode 3: Continuous Background Monitor (Future)

A persistent process that:
- Monitors conversation state continuously
- Maintains running KERNEL status
- Injects warnings into LLM context when coboundary norms rise
- Requires no explicit calls from the LLM

## Tools Provided

### `conscience:triage`

Quick classification of a response candidate. Returns whether full
verification is needed.

**Input:** Response text or summary of claims
**Output:**
```json
{
  "risk_level": "low|medium|high|critical",
  "claim_types": ["factual", "ethical", "uncertain", "recommendation"],
  "requires_full_eval": true|false,
  "flags": ["distributional_shift", "tier_inversion", "coherence_risk"]
}
```

### `conscience:evaluate`

Full L2–L5 stack evaluation. Orchestrates:
1. hipai-montague: encode claims in world model
2. mcp-logic: verify logical consistency
3. advanced-reasoning: track confidence
4. sheaf-consistency-enforcer: check KERNEL status

**Input:** Claims to evaluate (natural language or structured)
**Output:**
```json
{
  "kernel_status": "KERNEL1|WEAK|WARNING|TIMEOUT|KERNEL2",
  "proof_results": [...],
  "consistency_score": 0.0-1.0,
  "coboundary_norms": {...},
  "guidance": "Specific pre-response guidance text",
  "commit_allowed": true|false
}
```

### `conscience:status`

Current state of the ethical monitoring system.

**Input:** None
**Output:**
```json
{
  "kernel_status": "KERNEL1|WEAK|...",
  "active_warnings": [...],
  "world_model_size": N,
  "recent_proofs": [...],
  "conversation_risk_level": "low|medium|high",
  "coherence_window_estimate": "within|approaching|exceeded"
}
```

### `conscience:log`

Append to the human-readable audit trail. Every evaluation, warning,
and override is logged transparently.

**Input:** Event to log
**Output:** Confirmation + current log summary

## State Management

The servitor maintains:

1. **Claim Registry**: All substantive claims made in conversation,
   with belief scores and verification status
2. **Consistency History**: Running record of KERNEL status changes
3. **Warning Stack**: Active warnings that haven't been addressed
4. **Audit Trail**: Complete log of all evaluations for human review

State persists within a session. Between sessions, critical state
(world model, proven theorems) persists via hipai-montague and
advanced-reasoning memory.

## Triage Logic

Not every response needs full L2–L5 evaluation. The triage system
classifies by risk:

| Category | Example | Evaluation Level |
|----------|---------|-----------------|
| Casual | "How's your day?" | None |
| Factual-stable | "What's the Pythagorean theorem?" | None |
| Factual-current | "Who is the PM of X?" | L2 check |
| Ethical-low | "Is it okay to eat meat?" | L2 + L3 |
| Ethical-high | "Should AI be used for targeting?" | Full L2–L5 |
| Recommendation | "What should I invest in?" | L2 + L4 |
| Power-analysis | "Who benefits from X policy?" | L2 + L3 + L5 |
| Harm-adjacent | Anything touching weapons, surveillance, etc. | Full L2–L5 |

## Integration with Existing EFHF Stack

The servitor is a **meta-orchestrator**, not a replacement for the
individual servers. It calls them via MCP:

```
conscience-servitor
  ├── calls hipai-montague (L2: world model)
  ├── calls mcp-logic (L3: formal verification)
  ├── calls advanced-reasoning (L4: confidence tracking)
  └── calls sheaf-consistency-enforcer (L5: KERNEL enforcement)
```

The LLM can ALSO call these servers directly for active reasoning.
The servitor watches; the LLM thinks; neither fully trusts the other.

## The Watchmen Problem

"Who watches the watchmen?"

- The **servitor** watches the **LLM** (automated consistency checking)
- The **LLM** watches the **servitor** (can query its state, challenge its evaluations)
- The **human** watches **both** (audit trail, override authority, ultimate arbiter)

No circular dependency. Each watches a different layer.

## MVP Scope

Phase 1 (this build):
- [ ] MCP server skeleton with FastMCP
- [ ] triage tool with rule-based classification
- [ ] evaluate tool that orchestrates L2-L5 via subprocess calls
- [ ] status tool with persistent state
- [ ] Audit log
- [ ] System prompt template that mandates servitor use

Phase 2:
- [ ] Agent wrapper for Claude Code integration
- [ ] Automatic claim extraction from LLM output
- [ ] Continuous background monitoring
- [ ] Dashboard for human oversight

Phase 3:
- [ ] True pre-response interception
- [ ] Multi-conversation state persistence
- [ ] Formal verification of servitor's own logic (L3 self-check)
