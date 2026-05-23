import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.config import settings
from app.main import app


class Capture:
    auto_called = False
    pinned = []
    options = {}
    allow_web_search = None
    observed_search_plan = {}
    observed_context = []
    llm_messages = []


async def fake_build_auto_context(**kwargs):
    Capture.auto_called = True
    return {
        "snippets": [
            {
                "paper_id": "paper-1",
                "sent_id": "s1",
                "text": "PD-L1 expression is associated with response.",
                "source": "auto_context",
                "auto_context": True,
            }
        ],
        "plan": {
            "state_key": "search:v1|len:short|intent:question|biomed:yes|selected:none",
            "action_key": "search:v1|queries:few|breadth:medium|synonyms:yes|llm:no|notes:no",
            "strategy": "medium",
            "variants": [{"label": "original", "query": "PD-L1 response", "strategy": "medium", "source": "deterministic"}],
            "result_count": 1,
            "query_labels": ["original"],
            "used_llm": False,
            "note": "Auto-context search found one snippet.",
        },
    }


async def fake_build_prompt_and_citations(tenant, message, pinned, options):
    Capture.pinned = pinned
    Capture.options = options
    return f"Context prompt for: {message}", {"snippets": pinned}, {"num_snippets": len(pinned)}


class FakeContextPolicy:
    def __init__(self, tenant):
        self.tenant = tenant

    async def plan(self, *, session_id, message, allow_web_search, confidence_min):
        Capture.allow_web_search = allow_web_search
        return SimpleNamespace(
            turn_index=0,
            context_prefix="",
            selected_context=[],
            retrieved_triplets=[],
            web_results=(
                [
                    {
                        "title": "DuckDuckGo abstract",
                        "snippet": "Web grounding snippet for lung cancer TME.",
                        "url": "https://example.org/lung-tme",
                    }
                ]
                if allow_web_search
                else []
            ),
            warnings=[],
            meta={"turn_index": 0},
        )

    async def observe_turn(self, *, selected_context, search_plan=None, **kwargs):
        Capture.observed_context = selected_context
        Capture.observed_search_plan = search_plan or {}
        return {
            "conflicts": [],
            "nli_evidence": [],
            "claim_support": [],
            "reward": {"score": 0.5},
            "evidence_table": {},
        }


class FakeLLMClient:
    async def chat_stream(self, messages):
        Capture.llm_messages = messages
        yield json.dumps({"choices": [{"delta": {"content": "grounded answer"}}]})
        yield "[DONE]"


class ChatAutoContextTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        self.headers = {"X-Tenant-Id": "default"}
        if settings.security.require_api_key:
            self.headers["X-API-Key"] = settings.security.api_key
        Capture.auto_called = False
        Capture.pinned = []
        Capture.options = {}
        Capture.allow_web_search = None
        Capture.observed_search_plan = {}
        Capture.observed_context = []
        Capture.llm_messages = []

    @staticmethod
    def _events(response):
        events = []
        for line in response.text.splitlines():
            if not line.startswith("data:"):
                continue
            raw = line[5:].strip()
            if raw and raw != "[DONE]":
                events.append(json.loads(raw))
        return events

    def test_chat_without_selected_context_uses_auto_context_and_records_search_plan(self):
        with patch("app.routers.chat.build_auto_context", side_effect=fake_build_auto_context), patch(
            "app.routers.chat.build_prompt_and_citations", side_effect=fake_build_prompt_and_citations
        ), patch("app.routers.chat.ContextPolicy", FakeContextPolicy), patch("app.routers.chat.LLMClient", FakeLLMClient):
            response = self.client.post(
                "/chat/",
                headers=self.headers,
                json={
                    "message": "Does PD-L1 predict response?",
                    "items": [],
                    "options": {"allow_memory": True, "allow_auto_context": True, "allow_extra_retrieval": True},
                },
            )

        self.assertEqual(response.status_code, 200)
        events = self._events(response)
        citations = next(item["data"] for item in events if item["type"] == "citations")
        self.assertTrue(Capture.auto_called)
        self.assertEqual(Capture.pinned[0]["source"], "auto_context")
        self.assertFalse(Capture.options["allow_extra_retrieval"])
        self.assertEqual(Capture.observed_search_plan["result_count"], 1)
        self.assertEqual(Capture.observed_context[0]["source"], "auto_context")
        self.assertEqual(citations["auto_context"]["result_count"], 1)
        self.assertIn("Do not add outside biomedical mechanisms", Capture.llm_messages[0]["content"])
        self.assertIn("plausible", Capture.llm_messages[0]["content"])

    def test_chat_discloses_enabled_web_context_in_citations(self):
        with patch(
            "app.routers.chat.build_prompt_and_citations", side_effect=fake_build_prompt_and_citations
        ), patch("app.routers.chat.ContextPolicy", FakeContextPolicy), patch("app.routers.chat.LLMClient", FakeLLMClient):
            response = self.client.post(
                "/chat/",
                headers=self.headers,
                json={
                    "message": "What external context exists for lung cancer TME?",
                    "items": [],
                    "options": {"allow_memory": True, "allow_auto_context": False, "allow_web_search": True},
                },
            )

        self.assertEqual(response.status_code, 200)
        events = self._events(response)
        citations = next(item["data"] for item in events if item["type"] == "citations")
        self.assertTrue(Capture.allow_web_search)
        self.assertEqual(citations["memory"]["web_result_count"], 1)
        self.assertEqual(citations["web_context"][0]["title"], "DuckDuckGo abstract")

    def test_ambiguous_evidence_assembly_clarification_is_plain_answer_prefix(self):
        from app.routers.chat import _hold_generation_for_clarification, _opening_clarification_prefix

        assembly = {
            "clarification_recommended": True,
            "candidate_frames": [
                {"label": "Literal user frame"},
                {"label": "Relation/evidence bridge frame"},
            ],
            "evidence_puzzle": {"edge_support_status": "missing"},
        }
        prefix = _opening_clarification_prefix(assembly)

        self.assertIn("supported evidence pieces", prefix)
        self.assertIn("which interpretation should lead", prefix.lower())
        self.assertIn("Literal user frame", prefix)
        self.assertTrue(_hold_generation_for_clarification(assembly))


if __name__ == "__main__":
    unittest.main()
