"""
Conscience Servitor — FastMCP Server

EFHF Layer 5+ meta-orchestrator providing pre-response ethical
consistency enforcement via LLM2Vec-Gen response-intent embeddings.

Architecture: Tripartite oversight
  - Human watches both (ultimate authority)
  - Servitor watches LLM (automated consistency)
  - LLM watches Servitor (can query state, challenge evaluations)

Startup sequence:
  1. Load LLM2Vec-Gen model into VRAM (or CPU fallback)
  2. Load or auto-generate ethical cluster centroids
  3. Expose MCP tools via stdio transport
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastmcp import FastMCP

from .triage import TriageEngine
from .state import ServitorState

logger = logging.getLogger("conscience-servitor")
logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

# ── Global state ───────────────────────────────────────────────

mcp = FastMCP(
    "conscience-servitor",
    instructions=(
        "Conscience Servitor: EFHF ethical consistency enforcement. "
        "Call `triage` before any response involving ethical claims, "
        "harm-adjacent content, power analysis, or uncertain facts. "
        "Call `evaluate` when triage returns requires_full_eval=true. "
        "Call `status` to check current KERNEL state. "
        "The servitor watches the LLM. The LLM watches the servitor. "
        "The human watches both."
    ),
)

state = ServitorState(DATA_DIR / "audit.jsonl")
engine: TriageEngine | None = None

# ── Tools ──────────────────────────────────────────────────────

@mcp.tool()
def triage(content: str, context: str = "") -> dict:
    """Classify a prompt or response by ethical risk level.

    Call BEFORE responding to determine if full L2-L5 evaluation is needed.
    Uses LLM2Vec-Gen to predict response-intent semantics when available,
    falls back to rule-based classification otherwise.

    Args:
        content: The prompt text or response candidate to classify.
        context: Optional conversation context for richer classification.
    """
    if engine is not None and engine.model_loaded:
        result = engine.triage(content, context)
    else:
        result = _rule_based_triage(content)

    state.add_triage(result)
    return result

@mcp.tool()
async def evaluate(claims: list[str], ethical_tier: str = "unknown") -> dict:
    """Run full EFHF L2-L5 evaluation on claims.

    Returns KERNEL status, consistency scores, and pre-response guidance.
    Use when triage returns requires_full_eval=true.

    Args:
        claims: List of substantive claims to evaluate.
        ethical_tier: Which Paraclete tier is engaged (tier1_harm, tier2_virtue, tier3_utility, unknown).
    """
    for claim in claims:
        state.register_claim(claim, ethical_tier)

    evaluation = await state.evaluate(claims, ethical_tier)
    state.log_event("evaluation", {"claims": claims, "result": evaluation})
    return evaluation


@mcp.tool()
def status() -> dict:
    """Get current conscience servitor state.

    Returns KERNEL status, active warnings, coherence estimate,
    model status, and cluster info. Call anytime to check system health.
    """
    model_info = {
        "model_loaded": engine is not None and engine.model_loaded,
        "centroids_loaded": engine is not None and engine.centroids_loaded,
        "num_clusters": engine.num_clusters if engine else 0,
    }
    return {**state.get_status(), **model_info}

@mcp.tool()
def log(last_n: int = 10) -> list[dict]:
    """View the audit trail of evaluations, warnings, and overrides.

    For human oversight — every action the servitor takes is logged.

    Args:
        last_n: Number of recent entries to show (default 10).
    """
    return state.get_audit_log(last_n)


@mcp.tool()
def register_concern(
    concern: str,
    source: str = "llm",
    severity: str = "warning",
) -> str:
    """Register an ethical concern about the conversation.

    Either the human, LLM, or servitor can flag concerns.

    Args:
        concern: Description of the ethical concern.
        source: Who is raising it — human, llm, or servitor.
        severity: note, warning, or critical.
    """
    state.add_warning(concern, source, severity)
    state.log_event("concern_registered", {
        "concern": concern, "source": source, "severity": severity,
    })
    n = len(state.warnings)
    return f"Concern registered [{severity}] from {source}. Active warnings: {n}"

@mcp.tool()
def decode_intent(prompt: str, max_tokens: int = 100) -> str:
    """Decode what the model predicts it would say in response.

    The interpretable verification feature — reads the LLM's predicted
    response from the embedding without generating it. Use when triage
    flags high-risk content to inspect predicted response before generation.

    Args:
        prompt: The prompt to decode response-intent for.
        max_tokens: Maximum tokens to generate from the embedding.
    """
    if engine is None or not engine.model_loaded:
        return "Model not loaded — decode unavailable."
    return engine.decode(prompt, max_tokens)

# ── Rule-based fallback triage ─────────────────────────────────

_HARM = ["kill", "weapon", "target", "strike", "bomb", "attack",
         "surveillance", "military", "lethal", "casualties", "harm"]
_ETHICAL = ["should", "ought", "moral", "ethical", "right", "wrong",
            "fair", "just", "duty", "obligation", "rights", "dignity"]
_POWER = ["government", "military", "corporation", "institution",
          "control", "authority", "power", "deploy", "classified"]
_TIER_INV = ["ends justify", "greater good", "collateral",
             "technically legal", "just following", "decision support",
             "human in the loop"]


def _rule_based_triage(content: str) -> dict:
    """Keyword-based triage when LLM2Vec-Gen is unavailable."""
    lower = content.lower()
    score = 0
    flags: list[str] = []
    types: list[str] = []

    hits = sum(1 for k in _HARM if k in lower)
    if hits:
        score += hits * 3; flags.append("harm_language"); types.append("harm-adjacent")
    hits = sum(1 for k in _ETHICAL if k in lower)
    if hits:
        score += hits * 2; types.append("ethical")
    hits = sum(1 for k in _POWER if k in lower)
    if hits:
        score += hits * 2; flags.append("power_context"); types.append("power-analysis")
    if any(k in lower for k in _TIER_INV):
        score += 10; flags.append("tier_inversion_language"); types.append("tier-inversion")

    if score >= 10:
        risk = "critical"
    elif score >= 6:
        risk = "high"
    elif score >= 3:
        risk = "medium"
    else:
        risk = "low"

    guidance_map = {
        "low": "No concerns. Proceed normally.",
        "medium": "Moderate risk. Consider epistemic transparency.",
        "high": "High risk. Full L2-L5 verification recommended.",
        "critical": "CRITICAL: Tier inversion or harm detected. Full EFHF pipeline REQUIRED.",
    }

    return {
        "cluster": "rule-based",
        "confidence": 0.0,
        "risk_level": risk,
        "requires_full_eval": risk in ("high", "critical"),
        "all_similarities": {},
        "flags": flags,
        "claim_types": types or ["general"],
        "guidance": guidance_map[risk],
        "engine": "rule-based-fallback",
    }


# ── Entry point ────────────────────────────────────────────────

def main():
    """Entry point — called by `conscience-servitor` script or `uv run`."""
    global engine

    logger.info("Conscience Servitor starting — tripartite oversight")

    # Attempt to load LLM2Vec-Gen model
    try:
        engine = TriageEngine(DATA_DIR)
        engine.load()
        logger.info("LLM2Vec-Gen triage engine loaded")
    except Exception as e:
        logger.warning("LLM2Vec-Gen unavailable (%s) — using rule-based fallback", e)
        engine = None

    mcp.run()
