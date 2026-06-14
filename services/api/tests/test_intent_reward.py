# services/api/tests/test_intent_reward.py
"""P-3: intent-resolution reward attribution plumbing.

AutoContextPlan must carry intent_resolution metadata through to_dict() so it
reaches observe_turn (via the auto_context payload) and can be credited in the
ActionValue table. The observe_turn credit itself mirrors the proven search_plan
update directly above it.
"""
import unittest

from app.memory.search_agent import AutoContextPlan


class IntentResolutionPlumbingTests(unittest.TestCase):
    def test_to_dict_carries_intent_resolution(self):
        ir = {
            "tier": "tier1_router",
            "intent": "prior_context",
            "source": "nim",
            "confidence": 0.96,
            "state_key": "intentres|len=short",
            "action_key": "tier1_router:prior_context",
        }
        plan = AutoContextPlan(
            state_key="s", action_key="a", strategy="narrow", intent_resolution=ir
        )
        d = plan.to_dict()
        self.assertEqual(d["intent_resolution"]["tier"], "tier1_router")
        self.assertEqual(d["intent_resolution"]["action_key"], "tier1_router:prior_context")
        # action_key convention: "<tier>:<intent>"
        tier, _, intent = d["intent_resolution"]["action_key"].partition(":")
        self.assertEqual(tier, d["intent_resolution"]["tier"])
        self.assertEqual(intent, d["intent_resolution"]["intent"])

    def test_default_intent_resolution_is_empty(self):
        plan = AutoContextPlan(state_key="s", action_key="a", strategy="narrow")
        self.assertEqual(plan.to_dict()["intent_resolution"], {})


if __name__ == "__main__":
    unittest.main()
