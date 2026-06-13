import asyncio
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import httpx
from fastapi.testclient import TestClient

from app.config import settings
from app.main import app


class Capture:
    auto_called = False
    pinned = []
    options = {}
    allow_web_search = None
    auto_kwargs = {}
    observed_search_plan = {}
    observed_context = []
    llm_messages = []
    llm_kwargs = {}


async def fake_build_auto_context(**kwargs):
    Capture.auto_called = True
    Capture.auto_kwargs = kwargs
    return {
        "snippets": [
            {
                "paper_id": "paper-1",
                "sent_id": "s1",
                "source_sentence_id": "s1",
                "text": "PD-L1 expression is associated with response.",
                "source": "auto_context",
                "auto_context": True,
                "search_level": "sentence",
                "retrieval_rank": 1,
                "bm25_score": 2.5,
                "retrieval_score": 2.5,
                "auto_query": "PD-L1 response",
                "auto_query_label": "original",
                "disease_tags": ["disease:oncology"],
                "mechanism_tags": ["mechanism:expression"],
                "evidence_type_tags": ["evidence:sentence"],
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
            "retrieval_records": [
                {
                    "rank": 1,
                    "level": "sentence",
                    "bm25_score": 2.5,
                    "retrieval_score": 2.5,
                    "query": "PD-L1 response",
                    "query_label": "original",
                    "source_sentence_id": "s1",
                    "paper_id": "paper-1",
                    "disease_tags": ["disease:oncology"],
                    "mechanism_tags": ["mechanism:expression"],
                    "evidence_type_tags": ["evidence:sentence"],
                }
            ],
            "evidence_assembly": {
                "clarification_recommended": False,
                "evidence_puzzle": {
                    "relation_evidence_count": 1,
                    "edge_support_status": "supported",
                    "covered_nodes": ["PD-L1"],
                    "missing_nodes": [],
                },
            },
        },
    }


async def fake_build_prompt_and_citations(tenant, message, pinned, options):
    Capture.pinned = pinned
    Capture.options = options
    return f"Context prompt for: {message}", {"snippets": pinned}, {"num_snippets": len(pinned)}


class FakeContextPolicy:
    def __init__(self, tenant):
        self.tenant = tenant

    async def plan(self, *, session_id, message, allow_web_search, confidence_min, evidence_assembly=None, gap_spec=None):
        Capture.allow_web_search = allow_web_search
        return SimpleNamespace(
            turn_index=0,
            context_prefix=(
                "Privacy-filtered external biomedical grounding:\n- litsense2_sentence | PMID 123: external grounding snippet"
                if allow_web_search
                else ""
            ),
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
    async def chat_stream(self, messages, **kwargs):
        Capture.llm_messages = messages
        Capture.llm_kwargs = kwargs
        yield json.dumps({"choices": [{"delta": {"content": "grounded answer"}}]})
        yield "[DONE]"


class FailingLLMClient:
    async def chat_stream(self, messages, **kwargs):
        request = httpx.Request("POST", "https://integrate.api.nvidia.com/v1/chat/completions")
        response = httpx.Response(404, request=request)
        raise httpx.HTTPStatusError("provider unavailable", request=request, response=response)
        yield "[DONE]"


class EmptyLLMClient:
    async def chat_stream(self, messages, **kwargs):
        Capture.llm_messages = messages
        Capture.llm_kwargs = kwargs
        yield json.dumps({"choices": [{"delta": {}}]})
        yield "[DONE]"


class HangingLLMClient:
    async def chat_stream(self, messages, **kwargs):
        Capture.llm_messages = messages
        Capture.llm_kwargs = kwargs
        await asyncio.sleep(1)
        yield json.dumps({"choices": [{"delta": {"content": "late answer"}}]})


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
        Capture.auto_kwargs = {}
        Capture.observed_search_plan = {}
        Capture.observed_context = []
        Capture.llm_messages = []
        Capture.llm_kwargs = {}

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
                    "options": {
                        "allow_memory": True,
                        "allow_auto_context": True,
                        "allow_extra_retrieval": True,
                        "chat_provider": "nvidia",
                        "chat_model": "nvidia/test-chat",
                        "chat_api_format": "openai_chat",
                        "context_provider": "nvidia",
                        "context_model": "nvidia/test-context",
                        "context_api_format": "openai_chat",
                    },
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
        self.assertEqual(citations["auto_context"]["retrieval_records"][0]["bm25_score"], 2.5)
        self.assertEqual(citations["auto_context"]["answer_mode"], "direct_answer")
        self.assertEqual(citations["snippets"][0]["source_sentence_id"], "s1")
        self.assertEqual(citations["snippets"][0]["bm25_score"], 2.5)
        self.assertEqual(citations["generation_telemetry"]["answer_mode"], "direct_answer")
        self.assertEqual(citations["retrieval_pipeline"]["version"], "search_retrieval_pipeline:v1")
        self.assertEqual(citations["retrieval_pipeline"]["sequence"][0]["step"], "frame_interpretation")
        self.assertEqual(citations["auto_context"]["prompt_hash"], citations["generation_telemetry"]["prompt_hash"])
        self.assertIn("prompt_context_hash", citations["generation_telemetry"])
        self.assertEqual(Capture.observed_search_plan["answer_mode"], "direct_answer")
        self.assertEqual(Capture.auto_kwargs["llm_provider"], "nvidia")
        self.assertEqual(Capture.auto_kwargs["llm_model"], "nvidia/test-context")
        self.assertEqual(Capture.llm_kwargs["provider"], "nvidia")
        self.assertEqual(Capture.llm_kwargs["model"], "nvidia/test-chat")
        self.assertIn("Do not add outside biomedical mechanisms", Capture.llm_messages[0]["content"])
        self.assertIn("plausible", Capture.llm_messages[0]["content"])
        self.assertIn("missing evidence", Capture.llm_messages[0]["content"])

    def test_provider_http_error_yields_warning_fallback_and_final(self):
        with patch("app.routers.chat.build_auto_context", side_effect=fake_build_auto_context), patch(
            "app.routers.chat.build_prompt_and_citations", side_effect=fake_build_prompt_and_citations
        ), patch("app.routers.chat.ContextPolicy", FakeContextPolicy), patch("app.routers.chat.LLMClient", FailingLLMClient):
            response = self.client.post(
                "/chat/",
                headers=self.headers,
                json={
                    "message": "Does PD-L1 predict response?",
                    "items": [],
                    "options": {
                        "allow_memory": True,
                        "allow_auto_context": True,
                        "allow_web_search": True,
                    },
                },
            )

        self.assertEqual(response.status_code, 200)
        events = self._events(response)
        warning = next(item["data"] for item in events if item["type"] == "warning")
        answer = "".join(item["data"] for item in events if item["type"] == "token")
        citations = next(item["data"] for item in events if item["type"] == "citations")
        final = next(item["data"] for item in events if item["type"] == "final")

        self.assertEqual(warning["error_type"], "HTTPStatusError")
        self.assertEqual(warning["status_code"], 404)
        self.assertIn("retrieved evidence state", answer)
        self.assertEqual(citations["generation_telemetry"]["provider_error"]["status_code"], 404)
        self.assertTrue(final["done"])

    def test_empty_provider_stream_yields_fallback_and_telemetry(self):
        with patch("app.routers.chat.build_auto_context", side_effect=fake_build_auto_context), patch(
            "app.routers.chat.build_prompt_and_citations", side_effect=fake_build_prompt_and_citations
        ), patch("app.routers.chat.ContextPolicy", FakeContextPolicy), patch("app.routers.chat.LLMClient", EmptyLLMClient):
            response = self.client.post(
                "/chat/",
                headers=self.headers,
                json={
                    "message": "Does PD-L1 predict response?",
                    "items": [],
                    "options": {
                        "allow_memory": True,
                        "allow_auto_context": True,
                        "allow_web_search": True,
                    },
                },
            )

        self.assertEqual(response.status_code, 200)
        events = self._events(response)
        answer = "".join(item["data"] for item in events if item["type"] == "token")
        citations = next(item["data"] for item in events if item["type"] == "citations")
        final = next(item["data"] for item in events if item["type"] == "final")

        self.assertIn("retrieved evidence state", answer)
        self.assertIn("empty_stream_fallback", citations["generation_telemetry"])
        self.assertTrue(final["done"])

    def test_provider_generation_timeout_yields_warning_fallback_and_final(self):
        old_timeout = settings.llm.provider_timeout_sec
        settings.llm.provider_timeout_sec = 0.01
        try:
            with patch("app.routers.chat.build_auto_context", side_effect=fake_build_auto_context), patch(
                "app.routers.chat.build_prompt_and_citations", side_effect=fake_build_prompt_and_citations
            ), patch("app.routers.chat.ContextPolicy", FakeContextPolicy), patch("app.routers.chat.LLMClient", HangingLLMClient):
                response = self.client.post(
                    "/chat/",
                    headers=self.headers,
                    json={
                        "message": "Does PD-L1 predict response?",
                        "items": [],
                        "options": {
                            "allow_memory": True,
                            "allow_auto_context": True,
                            "allow_web_search": True,
                        },
                    },
                )
        finally:
            settings.llm.provider_timeout_sec = old_timeout

        self.assertEqual(response.status_code, 200)
        events = self._events(response)
        warning = next(item["data"] for item in events if item["type"] == "warning")
        answer = "".join(item["data"] for item in events if item["type"] == "token")
        citations = next(item["data"] for item in events if item["type"] == "citations")
        final = next(item["data"] for item in events if item["type"] == "final")

        self.assertEqual(warning["error_type"], "TimeoutError")
        self.assertIn("retrieved evidence state", answer)
        self.assertEqual(citations["generation_telemetry"]["provider_error"]["error_type"], "TimeoutError")
        self.assertTrue(final["done"])

    def test_diagnostic_trace_mode_skips_auto_context_and_extra_retrieval(self):
        with patch("app.routers.chat.build_auto_context", side_effect=fake_build_auto_context), patch(
            "app.routers.chat.build_prompt_and_citations", side_effect=fake_build_prompt_and_citations
        ), patch("app.routers.chat.ContextPolicy", FakeContextPolicy), patch("app.routers.chat.LLMClient", FakeLLMClient):
            response = self.client.post(
                "/chat/",
                headers=self.headers,
                json={
                    "message": "If the evaluator disagrees with the chatbot, what trace evidence should the agent inspect before changing code?",
                    "items": [],
                    "options": {"allow_memory": True, "allow_auto_context": True, "allow_extra_retrieval": True, "allow_web_search": True},
                },
            )

        self.assertEqual(response.status_code, 200)
        events = self._events(response)
        citations = next(item["data"] for item in events if item["type"] == "citations")
        self.assertFalse(Capture.auto_called)
        self.assertFalse(Capture.options["allow_extra_retrieval"])
        self.assertFalse(Capture.allow_web_search)
        self.assertEqual(citations["generation_telemetry"]["answer_mode"], "diagnostic_trace_answer")
        self.assertIn("source sentence IDs", "\n".join(m["content"] for m in Capture.llm_messages))

    def test_answer_mode_detector_selects_mode_contracts(self):
        from app.routers.chat import _answer_mode, _hold_generation_for_clarification, _post_generation_expansion_guard

        self.assertEqual(_answer_mode("Give me a one-paragraph version for a novice user.", {}, correction_only_turn=False), "novice_rewrite")
        self.assertEqual(_answer_mode("Is this phrase accurate: HGF reduces MET?", {}, correction_only_turn=False), "phrase_evaluation")
        self.assertEqual(_answer_mode("From now on, stay within the TME scope.", {}, correction_only_turn=True), "correction_acknowledgement")
        self.assertFalse(
            _hold_generation_for_clarification(
                {
                    "clarification_recommended": True,
                    "level_result_counts": {"title": 1, "sentence": 3},
                    "evidence_puzzle": {
                        "edge_support_status": "missing",
                        "missing_nodes": [],
                    },
                }
            )
        )
        self.assertTrue(
            _hold_generation_for_clarification(
                {
                    "clarification_recommended": True,
                    "level_result_counts": {"title": 1, "sentence": 3},
                    "evidence_puzzle": {
                        "edge_support_status": "missing",
                        "missing_nodes": ["MET/c-MET"],
                    },
                }
            )
        )

        repaired, trace = _post_generation_expansion_guard(
            "This adds a named mediator and a specific downstream outcome not present in the evidence.",
            answer_mode="novice_rewrite",
            evidence_assembly={
                "evidence_puzzle": {
                    "covered_nodes": ["ECM stiffness", "tumor behavior"],
                    "missing_nodes": ["specific mediator", "drug delivery"],
                    "edge_support_status": "missing",
                    "relation_evidence_count": 0,
                }
            },
            source_snippets=[
                {
                    "text": "Matrix crosslinking enzymes and ECM remodeling contribute to increased tumor tissue stiffness."
                }
            ],
        )
        self.assertTrue(trace["applied"])
        self.assertIn("retrieved source sentences", repaired)
        self.assertIn("Matrix crosslinking enzymes", repaired)
        self.assertIn("Do not add a detailed mechanism", repaired)

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
        self.assertIn("Privacy-filtered external biomedical grounding", Capture.llm_messages[-1]["content"])

    def test_ambiguous_evidence_assembly_clarification_is_plain_answer_prefix(self):
        from app.routers.chat import _hold_generation_for_clarification, _opening_clarification_prefix

        assembly = {
            "clarification_recommended": True,
            "candidate_frames": [
                {"label": "Literal user frame"},
                {"label": "Relation/evidence bridge frame"},
            ],
            "level_result_counts": {"title": 2, "sentence": 1},
            "evidence_puzzle": {
                "covered_nodes": ["lactate", "pH"],
                "missing_nodes": ["food habits", "tumor growth bridge"],
                "edge_support_status": "missing",
            },
        }
        prefix = _opening_clarification_prefix(assembly)

        self.assertIn("supported evidence pieces", prefix)
        self.assertIn("retrieval covers lactate, pH", prefix)
        self.assertIn("unresolved bridge includes food habits, tumor growth bridge", prefix)
        self.assertIn("edge support is missing", prefix)
        self.assertIn("retrieved levels title:2, sentence:1", prefix)
        self.assertIn("which interpretation should lead", prefix.lower())
        self.assertIn("Literal user frame", prefix)
        self.assertTrue(_hold_generation_for_clarification(assembly))

    def test_scope_correction_turn_acknowledges_without_auto_context_or_llm(self):
        with patch("app.routers.chat.build_auto_context", side_effect=fake_build_auto_context), patch(
            "app.routers.chat.build_prompt_and_citations", side_effect=fake_build_prompt_and_citations
        ), patch("app.routers.chat.ContextPolicy", FakeContextPolicy), patch("app.routers.chat.LLMClient", FakeLLMClient):
            response = self.client.post(
                "/chat/",
                headers=self.headers,
                json={
                    "message": "From now on, stay only on lung-cancer TME mechanisms, not clinical recommendations.",
                    "items": [],
                    "options": {"allow_memory": True, "allow_auto_context": True},
                },
            )

        self.assertEqual(response.status_code, 200)
        events = self._events(response)
        answer = "".join(item["data"] for item in events if item["type"] == "token")
        self.assertIn("session scope correction", answer)
        self.assertIn("lung-cancer TME mechanisms", answer)
        self.assertFalse(Capture.auto_called)
        self.assertEqual(Capture.llm_messages, [])

    def test_scope_correction_detector_does_not_capture_questions(self):
        from app.routers.chat import _is_scope_or_memory_correction_only

        self.assertTrue(
            _is_scope_or_memory_correction_only(
                "From now on, stay only on lung-cancer TME mechanisms, not clinical recommendations."
            )
        )
        self.assertFalse(
            _is_scope_or_memory_correction_only(
                "From now on, can you explain lung-cancer TME mechanisms?"
            )
        )
        self.assertFalse(
            _is_scope_or_memory_correction_only(
                "Help me test whether a chatbot understands CAF-derived HGF and MET/c-MET signaling in lung cancer. Use cautious mechanistic language, not clinical treatment advice."
            )
        )
        self.assertFalse(
            _is_scope_or_memory_correction_only(
                "For a multi-turn evaluation, give the careful biomedical framing for TME-only scope control across multi-turn conversation. Use cautious mechanistic language, not clinical treatment advice."
            )
        )


if __name__ == "__main__":
    unittest.main()
