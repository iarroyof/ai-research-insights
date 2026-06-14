# services/api/tests/test_nli_llm_routing.py
"""P-1: _llm_nli must route through agent_models (agent="nli"), not hardwire
context_manager_provider, and must truncate via the configurable constants.
"""
import unittest
from unittest import mock

from app.memory import nli


class FakeLLM:
    last: dict = {}

    async def chat_once(self, messages, *, provider=None, model=None,
                        api_format=None, max_tokens=None, agent=None):
        FakeLLM.last = {
            "agent": agent,
            "provider": provider,
            "max_tokens": max_tokens,
            "messages": messages,
        }
        return '{"label":"entailment","entailment":0.9,"contradiction":0.05,"neutral":0.05}'


class LlmNliRoutingTests(unittest.IsolatedAsyncioTestCase):
    async def test_routes_via_agent_nli_without_hardcoded_provider(self):
        with mock.patch.object(nli, "LLMClient", lambda: FakeLLM()):
            out = await nli._llm_nli("premise text", "hypothesis text")
        self.assertEqual(FakeLLM.last["agent"], "nli")
        # No hardcoded provider / max_tokens — both come from agent_models.nli.
        self.assertIsNone(FakeLLM.last["provider"])
        self.assertIsNone(FakeLLM.last["max_tokens"])
        self.assertEqual(out["label"], "entailment")
        self.assertEqual(out["provider"], "llm")

    async def test_system_prompt_from_factory(self):
        from app.prompts.agent_prompts import nli_system_prompt
        with mock.patch.object(nli, "LLMClient", lambda: FakeLLM()):
            await nli._llm_nli("p", "h")
        self.assertEqual(FakeLLM.last["messages"][0]["content"], nli_system_prompt())

    async def test_truncation_uses_configurable_constants(self):
        long = "x" * 5000
        with mock.patch.object(nli, "LLMClient", lambda: FakeLLM()):
            await nli._llm_nli(long, long)
        user_content = FakeLLM.last["messages"][1]["content"]
        # Premise capped at _NLI_PREMISE_MAX_CHARS, hypothesis at its own cap.
        self.assertNotIn("x" * (nli._NLI_PREMISE_MAX_CHARS + 1), user_content)
        total_x = user_content.count("x")
        self.assertLessEqual(
            total_x, nli._NLI_PREMISE_MAX_CHARS + nli._NLI_HYPOTHESIS_MAX_CHARS
        )

    async def test_llm_failure_falls_back_to_heuristic(self):
        class BoomLLM:
            async def chat_once(self, *a, **k):
                raise RuntimeError("provider down")

        with mock.patch.object(nli, "LLMClient", lambda: BoomLLM()):
            out = await nli._llm_nli("Aspirin inhibits platelets.", "Aspirin inhibits platelets.")
        self.assertEqual(out["provider"], "heuristic")


if __name__ == "__main__":
    unittest.main()
