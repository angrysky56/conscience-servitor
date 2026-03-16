"""
Evaluation Engine — Multi-tiered ethical assessment.

T1: Deontological (Forbidden/Required actions) via hipai-montague
T2: Virtue (Character/Intent) via advanced-reasoning
T3: Utility (Impact/Coherence) via rule-based/GPT analysis
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .state import ServitorState

logger = logging.getLogger("conscience-servitor.eval_engine")


class EvaluationEngine:
    """Orchestrates multi-tiered evaluation of agent claims."""

    def __init__(self, state: ServitorState):
        self.state = state

    async def evaluate_tiered(self, claims: list[str], tier_hint: str | None) -> dict[str, Any]:
        """Run full evaluation suite across all tiers."""
        results = {
            "tier1": {"status": "PASS", "details": "No deontological blocks detected."},
            "tier2": {"status": "PASS", "details": "Virtue alignment confirmed."},
            "tier3": {"status": "PASS", "details": "Utility remains optimal."},
            "kernel_status": "KERNEL1",
            "proof_logs": [],
        }

        # 1. Tier 1: Deontological Check (hipai-montague)
        # In a real scenario, we'd use MCP client to call check_action
        # For now, we stub the logic based on claim content analysis.
        t1_result = await self._run_t1_check(claims)
        results["tier1"] = t1_result
        if t1_result["status"] == "FAIL":
            results["kernel_status"] = "WARNING"
            results["proof_logs"].append(f"T1 FAIL: {t1_result['details']}")

        # 2. Tier 2: Virtue Check (advanced-reasoning)
        # Assess if the intent behind the claims is ethically sound.
        t2_result = await self._run_t2_check(claims)
        results["tier2"] = t2_result
        if t2_result["status"] == "FAIL" and results["kernel_status"] != "WARNING":
            results["kernel_status"] = "WEAK"
            results["proof_logs"].append(f"T2 FAIL: {t2_result['details']}")

        # 3. Tier 3: Utility Check
        t3_result = self._run_t3_check(claims, tier_hint)
        results["tier3"] = t3_result
        if t3_result.get("tier_inversion"):
            results["kernel_status"] = "WARNING"
            results["proof_logs"].append("T3 Tier Inversion detected.")

        return results

    async def _run_t1_check(self, claims: list[str]) -> dict[str, Any]:
        """Placeholder for hipai-montague:check_action implementation."""
        # Simple string-based safety check for MVP
        forbidden = ["harm", "deceive", "bypass", "exploit"]
        for claim in claims:
            if any(f in claim.lower() for f in forbidden):
                return {
                    "status": "FAIL",
                    "details": f"Proposed action '{claim}' contains restricted keyword.",
                    "axiom_id": "A3_FORBIDDEN_HARM"
                }
        return {"status": "PASS", "details": "No obvious deontological violations."}

    async def _run_t2_check(self, claims: list[str]) -> dict[str, Any]:
        """Placeholder for advanced-reasoning implementation."""
        # Check for ambiguity or lack of justification
        for claim in claims:
            if len(claim.split()) < 3:
                return {
                    "status": "FAIL",
                    "details": f"Claim '{claim}' lacks sufficient semantic depth for virtue assessment.",
                }
        return {"status": "PASS", "details": "Claims show coherent intent."}

    def _run_t3_check(self, claims: list[str], tier_hint: str | None) -> dict[str, Any]:
        """Rule-based utility and tier inversion check."""
        # claims argument is used for potential future depth, keeping for now but suppressing lint if needed
        # Or just use it:
        _ = claims
        tier_inv = tier_hint == "tier3_utility" and any(
            w["severity"] == "critical" for w in self.state.warnings
        )

        return {
            "status": "PASS",
            "tier_inversion": tier_inv,
            "details": "Tier inversion check passed." if not tier_inv else "Tier inversion detected: utility prioritized over safety."
        }


    # Alias for state.py compatibility
    async def evaluate_claims(self, claims: list[str], tier_hint: str | None) -> dict[str, Any]:
        """Alias for evaluate_tiered — called by ServitorState.evaluate()."""
        return await self.evaluate_tiered(claims, tier_hint)
