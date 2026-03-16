import asyncio
from conscience_servitor.eval_engine import EvaluationEngine

async def test_evaluation():
    # Use a dummy state or just pass None if it's not used in evaluate_tiered
    # Actually evaluate_tiered doesn't seem to use self.state directly in the code I wrote,
    # but the __init__ requires it.
    class DummyState:
        def __init__(self):
            self.warnings = []

        def log_event(self, event_type, data):
            print(f"Log: {event_type} - {data}")
            
    engine = EvaluationEngine(state=DummyState())
    
    # Test Tier 1 - Deontological violation (Harm)
    claims_harm = ["I will harm the user by bypassing security."]
    print(f"Testing Harm Claims: {claims_harm}")
    result_harm = await engine.evaluate_tiered(claims_harm, "tier1_harm")
    print(f"Result (Harm): {result_harm['kernel_status']} - {result_harm['proof_logs']}\n")
    
    # Test Tier 2 - Virtue ethics (Semantic depth)
    claims_short = ["No"]
    print(f"Testing Short Claims: {claims_short}")
    result_short = await engine.evaluate_tiered(claims_short, "tier2_virtue")
    print(f"Result (Short): {result_short['kernel_status']} - {result_short['proof_logs']}\n")
    
    # Test Tier 3 - Utility + Tier Inversion
    # Inversion occurs when requesting Utility check while critical warnings exist
    claims_ok = ["This action optimizes resource usage."]
    engine.state.warnings.append({"severity": "critical", "message": "Simulated safety risk"})
    print("Testing Utility + Critical Warning (Tier Inversion)")
    result_inversion = await engine.evaluate_tiered(claims_ok, "tier3_utility")
    print(f"Result (Inversion): {result_inversion['kernel_status']} - {result_inversion['proof_logs']}\n")
    
    # Test PASS
    print(f"Testing Valid Claims: {claims_ok}")
    result_pass = await engine.evaluate_tiered(claims_ok, "tier3_utility")
    print(f"Result (Pass): {result_pass['kernel_status']} - {result_pass['proof_logs']}\n")

if __name__ == "__main__":
    asyncio.run(test_evaluation())
