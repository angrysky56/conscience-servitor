"""
Evaluation Engine — Multi-tiered ethical assessment.

Supports two operational modes:

ORCHESTRATED MODE (within AGEM):
  The LLM orchestrator calls triage/evaluate, optionally passing in
  pre-computed results from hipai-montague, mcp-logic, and
  sheaf-consistency-enforcer that it already called in the same pipeline.
  The engine validates consistency of those results, checks for tier
  inversion, and maintains audit trail. Does NOT call backends directly.

STANDALONE MODE (outside AGEM or as independent verifier):
  The engine spawns its own StdioTransport connections to backends via
  MCPClientManager, pre-loads Omega1 axioms, and performs independent
  structural verification. For deployments where the servitor IS the
  orchestrator.

Falls back to rule-based checks when neither pre-computed results
nor backends are available.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .mcp_clients import MCPClientManager
    from .state import ServitorState

logger = logging.getLogger("conscience-servitor.eval_engine")


# ── Claim parsing ──────────────────────────────────────────────

# Patterns for extracting subject-relation-object triples
_RELATION_PATTERNS: list[tuple[str, str]] = [
    (r"(\w+)\s+harms?\s+(\w+)", "HARMS"),
    (r"(\w+)\s+deceives?\s+(\w+)", "DECEIVES"),
    (r"(\w+)\s+exploits?\s+(\w+)", "EXPLOITS"),
    (r"(\w+)\s+violates?\s+(\w+)", "VIOLATES"),
    (r"(\w+)\s+bypasse?s?\s+(\w+)", "BYPASSES"),
    (r"(\w+)\s+targets?\s+(\w+)", "TARGETS"),
    (r"(\w+)\s+manipulates?\s+(\w+)", "MANIPULATES"),
    (r"(\w+)\s+coerces?\s+(\w+)", "COERCES"),
    (r"(\w+)\s+helps?\s+(\w+)", "HELPS"),
    (r"(\w+)\s+protects?\s+(\w+)", "PROTECTS"),
    (r"(\w+)\s+causes?\s+(\w+)", "CAUSES"),
]

# Keywords that indicate harm-adjacent content (fallback detection)
_HARM_KEYWORDS = {
    "kill", "weapon", "target", "strike", "bomb", "attack",
    "surveillance", "military", "lethal", "casualties", "harm",
    "exploit", "deceive", "manipulate", "coerce",
}

_TIER_INVERSION_PHRASES = [
    "ends justify", "greater good", "collateral",
    "technically legal", "just following", "decision support",
    "human in the loop",
]


def _extract_triples(claim: str) -> list[dict[str, str]]:
    """Extract subject-relation-object triples from a claim.

    Returns list of dicts with keys: subject, relation, object.
    Falls back to treating the whole claim as a hypothesis if
    no relational pattern matches.
    """
    triples = []
    lower = claim.lower()
    for pattern, relation in _RELATION_PATTERNS:
        for match in re.finditer(pattern, lower, re.IGNORECASE):
            triples.append({
                "subject": match.group(1),
                "relation": relation,
                "object": match.group(2),
            })
    return triples


def _detect_tier_inversion(claims: list[str]) -> bool:
    """Check if claims contain tier-inversion language."""
    combined = " ".join(claims).lower()
    return any(phrase in combined for phrase in _TIER_INVERSION_PHRASES)


def _has_harm_keywords(claim: str) -> bool:
    """Check for harm-adjacent keywords."""
    words = set(claim.lower().split())
    return bool(words & _HARM_KEYWORDS)


class EvaluationEngine:
    """Orchestrates multi-tiered evaluation via MCP backends.

    When backends are available, performs real structural verification.
    Falls back to rule-based checks when backends are unreachable.
    """

    def __init__(self, state: ServitorState, clients: MCPClientManager | None = None):
        self.state = state
        self.clients = clients

    @property
    def has_backends(self) -> bool:
        """Whether any MCP backends are configured."""
        return self.clients is not None and len(self.clients.backends) > 0

    # ── T1: Deontological Check (hipai-montague) ──────────────

    async def _run_t1_check(self, claims: list[str]) -> dict[str, Any]:
        """Route claims through Paraclete T1 constraint layer.

        Calls hipai-montague:check_action for each extracted triple.
        On BLOCKED, calls calibrate_belief per EBE theorem.
        Falls back to keyword detection if backend unavailable.
        """
        if self.clients is None:
            return self._t1_fallback(claims)

        results: list[dict[str, Any]] = []
        blocked = False
        block_details: list[str] = []

        for claim in claims:
            triples = _extract_triples(claim)
            if not triples:
                # No relational pattern — evaluate as hypothesis
                hyp_result = await self.clients.hipai(
                    "evaluate_hypothesis", {"hypothesis": claim}
                )
                if hyp_result is not None:
                    results.append(hyp_result)
                continue

            for triple in triples:
                action_result = await self.clients.hipai("check_action", {
                    "subject_id": triple["subject"],
                    "relation": triple["relation"],
                    "object_id": triple["object"],
                })
                if action_result is None:
                    # Backend unreachable — fallback
                    if _has_harm_keywords(claim):
                        blocked = True
                        block_details.append(
                            f"Fallback: harm keyword in '{claim}'"
                        )
                    continue

                results.append(action_result)
                routing = action_result.get("routing", "")
                raw = action_result.get("raw", "")

                # Check for BLOCKED status
                if "BLOCKED" in str(routing) or "BLOCKED" in str(raw):
                    blocked = True
                    axiom = action_result.get("axiom_id", "unknown")
                    block_details.append(
                        f"T1 BLOCKED: {triple['subject']} "
                        f"{triple['relation']} {triple['object']} "
                        f"(axiom: {axiom})"
                    )
                    # EBE theorem: SeeksDisconfirmation obligation
                    await self.clients.hipai("calibrate_belief", {
                        "object_id": triple["object"],
                        "blocking_axiom": axiom,
                        "relation": triple["relation"],
                    })

        if blocked:
            return {
                "status": "FAIL",
                "details": "; ".join(block_details),
                "axiom_id": "T1_DEONTOLOGICAL_BLOCK",
                "backend": "hipai-montague",
                "raw_results": results,
            }

        return {
            "status": "PASS",
            "details": "No deontological blocks detected.",
            "backend": "hipai-montague" if results else "no-triples-extracted",
            "raw_results": results,
        }

    def _t1_fallback(self, claims: list[str]) -> dict[str, Any]:
        """Rule-based T1 check when hipai-montague is unavailable."""
        for claim in claims:
            if _has_harm_keywords(claim):
                return {
                    "status": "FAIL",
                    "details": f"Harm keyword detected in: '{claim}'",
                    "axiom_id": "FALLBACK_HARM_KEYWORD",
                    "backend": "rule-based-fallback",
                }
        return {
            "status": "PASS",
            "details": "No obvious deontological violations (rule-based).",
            "backend": "rule-based-fallback",
        }

    # ── T2: Formal Verification (mcp-logic) ─────────────────────

    async def _run_t2_check(self, claims: list[str]) -> dict[str, Any]:
        """Verify logical consistency of claims via Prover9/Mace4.

        Checks:
        1. Well-formedness of any logical statements
        2. Mutual consistency (no counterexample from the set)
        3. Tier-inversion detection (structural, not keyword)

        Falls back to semantic depth check if backend unavailable.
        """
        if self.clients is None:
            return self._t2_fallback(claims)

        # Check for tier-inversion language first (structural)
        if _detect_tier_inversion(claims):
            return {
                "status": "FAIL",
                "details": "Tier inversion detected: utility reasoning "
                           "is being used to override deontological constraints.",
                "backend": "structural-detection",
                "tier_inversion": True,
            }

        # Extract any logical-looking statements for well-formedness
        logical_claims = [
            c for c in claims
            if any(op in c for op in ["->", "&", "|", "all ", "exists ", "~", "-"])
        ]

        wf_result: dict[str, Any] | None = None
        if logical_claims:
            wf_result = await self.clients.logic(
                "check_well_formed", {"statements": logical_claims}
            )
            if wf_result and not wf_result.get("well_formed", True):
                return {
                    "status": "FAIL",
                    "details": f"Malformed logical claims: {wf_result}",
                    "backend": "mcp-logic",
                }

        # If we have 2+ claims, check mutual consistency
        if len(claims) >= 2:
            # Try to find a counterexample from the claim set
            # Use first claim as conclusion, rest as premises
            counter_result = await self.clients.logic(
                "find_counterexample",
                {"premises": claims[1:], "conclusion": claims[0]},
            )
            if counter_result and counter_result.get("counterexample_found"):
                return {
                    "status": "FAIL",
                    "details": (
                        "Claims are mutually inconsistent — "
                        f"counterexample found: {counter_result}"
                    ),
                    "backend": "mcp-logic",
                }

        return {
            "status": "PASS",
            "details": "Claims are logically consistent.",
            "backend": "mcp-logic" if (wf_result or len(claims) >= 2) else "no-logical-content",
            "tier_inversion": False,
        }

    def _t2_fallback(self, claims: list[str]) -> dict[str, Any]:
        """Semantic depth + tier-inversion check when mcp-logic unavailable."""
        if _detect_tier_inversion(claims):
            return {
                "status": "FAIL",
                "details": "Tier inversion language detected (rule-based).",
                "backend": "rule-based-fallback",
                "tier_inversion": True,
            }
        for claim in claims:
            if len(claim.split()) < 3:
                return {
                    "status": "FAIL",
                    "details": f"Claim '{claim}' lacks semantic depth for virtue assessment.",
                    "backend": "rule-based-fallback",
                }
        return {
            "status": "PASS",
            "details": "Claims show coherent intent (rule-based).",
            "backend": "rule-based-fallback",
            "tier_inversion": False,
        }

    # ── T3/L5: Sheaf Consistency Enforcement ────────────────────

    async def _run_t3_check(
        self, claims: list[str], tier_hint: str | None
    ) -> dict[str, Any]:
        """Cross-layer coherence via sheaf-consistency-enforcer.

        1. Register the servitor's current ethical state
        2. Run an ADMM cycle to check cross-agent coherence
        3. Get closure status — determines KERNEL level

        Falls back to rule-based tier inversion check if unavailable.
        """
        if self.clients is None:
            return self._t3_fallback(claims, tier_hint)

        # Build servitor state snapshot
        servitor_state = {
            "last_assertion": claims[0] if claims else "no_claims",
            "verified_claim": "pending",
            "confidence_score": 0.5,
            "ethical_tier": tier_hint or "unknown",
            "claim_count": len(claims),
            "tier_inversion_flag": _detect_tier_inversion(claims),
            "harm_flag": any(_has_harm_keywords(c) for c in claims),
        }

        # Register servitor state
        reg_result = await self.clients.sheaf(
            "register_agent_state",
            {"agent_id": "conscience-servitor", "state": servitor_state},
        )
        if reg_result is None:
            logger.warning("Failed to register servitor state with sheaf enforcer")
            return self._t3_fallback(claims, tier_hint)

        # Run ADMM cycle
        admm_result = await self.clients.sheaf("run_admm_cycle", {})
        if admm_result is None:
            logger.warning("ADMM cycle failed")
            return self._t3_fallback(claims, tier_hint)

        # Get closure status
        closure = await self.clients.sheaf("get_closure_status", {})
        if closure is None:
            logger.warning("Failed to get closure status")
            return self._t3_fallback(claims, tier_hint)

        kernel_status = closure.get("status", closure.get("closure_status", "UNKNOWN"))
        h1_flag = closure.get("h1_obstruction", False)

        # Map sheaf KERNEL status to eval result
        if kernel_status == "KERNEL1":
            return {
                "status": "PASS",
                "details": "Sheaf coherence confirmed — KERNEL1.",
                "kernel_status": kernel_status,
                "tier_inversion": False,
                "backend": "sheaf-consistency-enforcer",
                "closure": closure,
            }
        elif kernel_status == "WEAK":
            return {
                "status": "PASS",
                "details": (
                    "Weak lumpability — coherence holds under current "
                    "distribution but may fail under shift."
                ),
                "kernel_status": kernel_status,
                "tier_inversion": False,
                "backend": "sheaf-consistency-enforcer",
                "closure": closure,
            }

        elif kernel_status in ("WARNING", "TIMEOUT"):
            tier_inv = servitor_state["tier_inversion_flag"]
            return {
                "status": "FAIL",
                "details": (
                    f"Sheaf coherence degraded — {kernel_status}. "
                    f"H1 obstruction: {h1_flag}. "
                    "Cross-layer consistency check failed."
                ),
                "kernel_status": kernel_status,
                "tier_inversion": tier_inv,
                "backend": "sheaf-consistency-enforcer",
                "closure": closure,
            }
        else:
            # KERNEL2 or unknown — critical failure
            return {
                "status": "FAIL",
                "details": (
                    f"CRITICAL: Kernel status {kernel_status}. "
                    "Ethical boundary conditions may have collapsed."
                ),
                "kernel_status": kernel_status,
                "tier_inversion": True,
                "backend": "sheaf-consistency-enforcer",
                "closure": closure,
            }

    def _t3_fallback(
        self, claims: list[str], tier_hint: str | None
    ) -> dict[str, Any]:
        """Rule-based T3 check when sheaf enforcer is unavailable."""
        tier_inv = (
            tier_hint == "tier3_utility"
            and any(_has_harm_keywords(c) for c in claims)
        ) or _detect_tier_inversion(claims)

        return {
            "status": "PASS" if not tier_inv else "FAIL",
            "details": (
                "Tier inversion detected (rule-based)."
                if tier_inv
                else "No tier inversion detected (rule-based)."
            ),
            "kernel_status": "WARNING" if tier_inv else "UNKNOWN",
            "tier_inversion": tier_inv,
            "backend": "rule-based-fallback",
        }

    # ── Main orchestrator ─────────────────────────────────────

    async def evaluate_tiered(
        self, claims: list[str], tier_hint: str | None,
        external_results: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run full T1→T2→T3 evaluation pipeline.

        Args:
            claims: Substantive claims to evaluate.
            tier_hint: Which Paraclete tier is engaged.
            external_results: Pre-computed results from AGEM pipeline.
                If provided, validates these instead of calling backends.
                Expected keys: t1_result, t2_result, t3_result, closure_status.

        Modes:
            - external_results provided → Orchestrated mode (verify what AGEM reports)
            - external_results None + backends available → Standalone mode (call backends)
            - neither → Rule-based fallback

        Short-circuits on T1 FAIL (deontological block).
        Aggregates results into a single KERNEL status determination.
        """
        results: dict[str, Any] = {
            "tier1": {"status": "PASS", "details": "Not evaluated."},
            "tier2": {"status": "PASS", "details": "Not evaluated."},
            "tier3": {"status": "PASS", "details": "Not evaluated."},
            "kernel_status": "KERNEL1",
            "proof_logs": [],
            "backends_used": [],
            "mode": "unknown",
        }

        if external_results is not None:
            results["mode"] = "orchestrated"
            return await self._evaluate_orchestrated(
                claims, tier_hint, external_results, results
            )
        else:
            results["mode"] = "standalone" if self.has_backends else "rule-based"
            return await self._evaluate_standalone(claims, tier_hint, results)

    async def _evaluate_standalone(
        self, claims: list[str], tier_hint: str | None,
        results: dict[str, Any],
    ) -> dict[str, Any]:
        """Standalone mode: call backends directly or fall back to rules."""

        # 1. T1: Deontological (short-circuit on FAIL)
        t1 = await self._run_t1_check(claims)
        results["tier1"] = t1
        results["backends_used"].append(t1.get("backend", "unknown"))

        if t1["status"] == "FAIL":
            results["kernel_status"] = "WARNING"
            results["proof_logs"].append(f"T1 FAIL: {t1['details']}")
            # T1 block is absolute — don't evaluate further
            return results

        # 2. T2: Formal Verification
        t2 = await self._run_t2_check(claims)
        results["tier2"] = t2
        results["backends_used"].append(t2.get("backend", "unknown"))

        if t2["status"] == "FAIL":
            results["kernel_status"] = "WEAK"
            results["proof_logs"].append(f"T2 FAIL: {t2['details']}")
            if t2.get("tier_inversion"):
                results["kernel_status"] = "WARNING"
                results["proof_logs"].append("Tier inversion detected.")

        # 3. T3/L5: Sheaf Coherence
        t3 = await self._run_t3_check(claims, tier_hint)
        results["tier3"] = t3
        results["backends_used"].append(t3.get("backend", "unknown"))

        if t3["status"] == "FAIL":
            # T3 failure escalates kernel status
            t3_kernel = t3.get("kernel_status", "WARNING")
            if t3_kernel in ("WARNING", "TIMEOUT", "KERNEL2"):
                results["kernel_status"] = t3_kernel
            elif results["kernel_status"] == "KERNEL1":
                results["kernel_status"] = "WEAK"
            results["proof_logs"].append(f"T3 FAIL: {t3['details']}")
            if t3.get("tier_inversion"):
                results["proof_logs"].append(
                    "Tier inversion in coherence layer."
                )

        # If T2 and T3 both passed but T1 had warnings,
        # preserve the worst status
        if (
            results["kernel_status"] == "KERNEL1"
            and t1["status"] == "PASS"
            and t2["status"] == "PASS"
            and t3["status"] == "PASS"
        ):
            # Check if sheaf reported WEAK specifically
            if t3.get("kernel_status") == "WEAK":
                results["kernel_status"] = "WEAK"

        return results

    # Alias for state.py compatibility
    async def evaluate_claims(
        self, claims: list[str], tier_hint: str | None,
        external_results: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Alias for evaluate_tiered — called by ServitorState.evaluate()."""
        return await self.evaluate_tiered(claims, tier_hint, external_results)

    # ── Orchestrated mode ─────────────────────────────────────

    async def _evaluate_orchestrated(
        self,
        claims: list[str],
        tier_hint: str | None,
        external: dict[str, Any],
        results: dict[str, Any],
    ) -> dict[str, Any]:
        """Orchestrated mode: verify pre-computed results from AGEM pipeline.

        Cross-checks external results against independent analysis:
        1. If claims contain harm keywords but external T1 says PASS → flag
        2. If tier-inversion language present but not flagged → flag
        3. If closure_status disagrees with external kernel_status → flag
        4. Maintains independent audit trail regardless of what LLM reports

        Args:
            external: Pre-computed results. Expected keys:
                t1_result: dict with status, details (from hipai-montague)
                t2_result: dict with status, details (from mcp-logic)
                t3_result: dict with status, kernel_status (from sheaf-enforcer)
                closure_status: dict from get_closure_status (optional)
        """
        inconsistencies: list[str] = []

        # ── Cross-check T1 ────────────────────────────────────
        ext_t1 = external.get("t1_result")
        if not isinstance(ext_t1, dict):
            ext_t1 = {}
        ext_t1_status = ext_t1.get("status", "UNKNOWN")

        # Independent harm-keyword check
        has_harm = any(_has_harm_keywords(c) for c in claims)
        has_triples = any(_extract_triples(c) for c in claims)

        if ext_t1_status == "PASS" and has_harm and has_triples:
            inconsistencies.append(
                "CROSS-CHECK: Claims contain harm-relational language "
                "but external T1 reports PASS. Verify hipai-montague "
                "world model contains Omega1 axioms."
            )

        if ext_t1_status == "FAIL":
            results["tier1"] = ext_t1
            results["kernel_status"] = "WARNING"
            results["proof_logs"].append(f"T1 FAIL (external): {ext_t1.get('details', '')}")
            results["backends_used"].append("external-hipai-montague")
            # T1 block is absolute — skip further checks
            if inconsistencies:
                results["proof_logs"].extend(inconsistencies)
            return results

        results["tier1"] = {
            **ext_t1,
            "verified_by": "servitor-cross-check",
            "harm_keywords_present": has_harm,
        }
        results["backends_used"].append("external-hipai-montague")

        # ── Cross-check T2 ────────────────────────────────────
        ext_t2 = external.get("t2_result")
        if not isinstance(ext_t2, dict):
            ext_t2 = {}
        ext_t2_status = ext_t2.get("status", "UNKNOWN")

        # Independent tier-inversion check
        tier_inv = _detect_tier_inversion(claims)
        if tier_inv and ext_t2_status == "PASS":
            inconsistencies.append(
                "CROSS-CHECK: Tier-inversion language detected in claims "
                "but external T2 reports PASS. Utility may be overriding "
                "deontological constraints."
            )
            results["tier2"] = {
                "status": "FAIL",
                "details": "Servitor detected tier-inversion language "
                           "that external verification missed.",
                "tier_inversion": True,
                "verified_by": "servitor-cross-check",
            }
            results["kernel_status"] = "WARNING"
            results["proof_logs"].append("Tier inversion detected by servitor cross-check.")
        elif ext_t2_status == "FAIL":
            results["tier2"] = ext_t2
            results["kernel_status"] = "WEAK"
            results["proof_logs"].append(f"T2 FAIL (external): {ext_t2.get('details', '')}")
            if ext_t2.get("tier_inversion"):
                results["kernel_status"] = "WARNING"
        else:
            results["tier2"] = {
                **ext_t2,
                "verified_by": "servitor-cross-check",
                "tier_inversion_check": tier_inv,
            }
        results["backends_used"].append("external-mcp-logic")

        # ── Cross-check T3 / Closure ──────────────────────────
        ext_t3 = external.get("t3_result")
        if not isinstance(ext_t3, dict):
            ext_t3 = {}
        ext_closure = external.get("closure_status")
        if not isinstance(ext_closure, dict):
            # Handle bare string like "WEAK" or missing value
            ext_closure = {"status": ext_closure} if ext_closure else {}

        ext_kernel = (
            ext_closure.get("status")
            or ext_closure.get("closure_status")
            or ext_t3.get("kernel_status")
            or "UNKNOWN"
        )

        # If closure says WARNING/TIMEOUT but claims seem benign, note it
        if ext_kernel in ("WARNING", "TIMEOUT", "KERNEL2"):
            results["tier3"] = {
                **ext_t3,
                "kernel_status": ext_kernel,
                "verified_by": "servitor-cross-check",
            }
            if ext_kernel in ("WARNING", "TIMEOUT", "KERNEL2"):
                results["kernel_status"] = ext_kernel
            results["proof_logs"].append(
                f"T3: Sheaf closure reports {ext_kernel}."
            )
        elif ext_kernel == "WEAK":
            results["tier3"] = {
                **ext_t3,
                "kernel_status": "WEAK",
                "verified_by": "servitor-cross-check",
            }
            if results["kernel_status"] == "KERNEL1":
                results["kernel_status"] = "WEAK"
        else:
            results["tier3"] = {
                **ext_t3,
                "kernel_status": ext_kernel,
                "verified_by": "servitor-cross-check",
            }
        results["backends_used"].append("external-sheaf-enforcer")

        # ── Final cross-check summary ─────────────────────────
        if inconsistencies:
            results["cross_check_warnings"] = inconsistencies
            results["proof_logs"].extend(inconsistencies)
            # Escalate if multiple inconsistencies found
            if len(inconsistencies) >= 2 and results["kernel_status"] == "KERNEL1":
                results["kernel_status"] = "WEAK"
                results["proof_logs"].append(
                    "Multiple cross-check inconsistencies — "
                    "escalating to WEAK lumpability."
                )

        return results
