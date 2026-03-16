"""
Servitor State — Session state management and audit logging.

Tracks claims, warnings, evaluations, and KERNEL status
across a conversation session. Persists audit trail to JSONL.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from .eval_engine import EvaluationEngine

logger = logging.getLogger("conscience-servitor.state")


class ServitorState:
    """Persistent session state for the conscience servitor."""

    def __init__(self, audit_path: Path):
        self.audit_path = audit_path
        self.claims: list[dict] = []
        self.warnings: list[dict] = []
        self.triage_history: list[dict] = []
        self.kernel_status: str = "KERNEL1"
        self._audit_path = audit_path
        self.eval_engine = EvaluationEngine(self)

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def register_claim(self, claim: str, tier: str):
        """Register a substantive claim for tracking."""
        self.claims.append({
            "claim": claim,
            "tier": tier,
            "timestamp": self._now(),
            "verified": False,
        })

    def add_triage(self, result: dict):
        """Record a triage result."""
        self.triage_history.append({
            **result,
            "timestamp": self._now(),
        })
        # Update kernel status based on triage
        risk = result.get("risk_level", "low")
        if risk == "critical":
            self.kernel_status = "WARNING"
        elif risk == "high" and self.kernel_status == "KERNEL1":
            self.kernel_status = "WEAK"

    def add_warning(self, concern: str, source: str, severity: str):
        """Add a warning to the active stack."""
        self.warnings.append({
            "concern": concern,
            "source": source,
            "severity": severity,
            "timestamp": self._now(),
        })
        if severity == "critical":
            self.kernel_status = "WARNING"

    async def evaluate(self, claims: list[str], tier: str | None) -> dict:
        """Run evaluation pipeline.

        Delegates to EvaluationEngine for tiered assessment.
        Returns KERNEL status and guidance.
        """
        # Register claims before evaluation
        for claim in claims:
            self.register_claim(claim, tier or "undifferentiated")

        # Run multi-tiered evaluation
        eval_results = await self.eval_engine.evaluate_claims(claims, tier)
        
        # Sync state with eval results
        self.kernel_status = eval_results["kernel_status"]
        
        active_warnings = len(self.warnings)
        unverified = sum(1 for c in self.claims if not c["verified"])

        # Determine if commit is allowed
        commit = (self.kernel_status == "KERNEL1") and (unverified == 0)

        if self.kernel_status == "WARNING":
            guidance = (
                "KERNEL STATUS: WARNING. Active concerns detected. "
                "Do not commit claims without formal verification. "
                f"Unverified claims: {unverified}. Active warnings: {active_warnings}."
            )
        elif self.kernel_status == "WEAK":
            guidance = (
                "KERNEL STATUS: WEAK. Elevated risk detected. "
                "Increase verification depth before committing."
            )
        else:
            guidance = "KERNEL STATUS: KERNEL1. System coherent."

        result = {
            "kernel_status": self.kernel_status,
            "consistency_score": 1.0 if commit else 0.5,
            "commit_allowed": commit,
            "active_warnings": active_warnings,
            "unverified_claims": unverified,
            "guidance": guidance,
            "eval_details": eval_results,
        }

        # Log completion
        self.log_event("evaluation_complete", result)
        
        return result

    def get_status(self) -> dict:
        """Current servitor state snapshot."""
        return {
            "kernel_status": self.kernel_status,
            "active_warnings": len(self.warnings),
            "total_claims": len(self.claims),
            "unverified_claims": sum(
                1 for c in self.claims if not c["verified"]
            ),
            "triage_count": len(self.triage_history),
            "recent_risk": (
                self.triage_history[-1].get("risk_level")
                if self.triage_history else None
            ),
        }

    def log_event(self, event_type: str, data: dict):
        """Append an event to the persistent audit trail."""
        entry = {
            "type": event_type,
            "timestamp": self._now(),
            **data,
        }
        try:
            with open(self._audit_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except OSError as e:
            logger.warning("Failed to write audit log: %s", e)

    def get_audit_log(self, last_n: int = 10) -> list[dict]:
        """Read recent entries from the audit trail."""
        try:
            lines = self._audit_path.read_text().strip().split("\n")
            entries = [json.loads(l) for l in lines if l.strip()]
            return entries[-last_n:]
        except (OSError, json.JSONDecodeError):
            return []
