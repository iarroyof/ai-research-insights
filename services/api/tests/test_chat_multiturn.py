import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.config import settings
from app.main import app


class FakeContextPolicy:
    history: dict[str, list[dict]] = {}

    def __init__(self, tenant: str):
        self.tenant = tenant

    async def plan(self, *, session_id: str, message: str, allow_web_search: bool, confidence_min: float, evidence_assembly: dict | None = None, gap_spec: dict | None = None):
        history = list(self.history.get(session_id, []))
        return SimpleNamespace(
            turn_index=len(history),
            context_prefix="",
            selected_context=[{"source": "recent", **item} for item in history],
            retrieved_triplets=[],
            web_results=[],
            warnings=[],
            meta={"turn_index": len(history)},
        )

    async def observe_turn(self, *, session_id: str, turn_index: int, question: str, answer: str, **kwargs):
        self.history.setdefault(session_id, []).extend(
            [
                {"role": "user", "text": question, "turn_index": turn_index},
                {"role": "assistant", "text": answer, "turn_index": turn_index + 1},
            ]
        )
        return {
            "conflicts": [],
            "nli_evidence": [],
            "claim_support": [],
            "reward": {},
            "evidence_table": {},
        }


class FakeLLMClient:
    calls: list[list[dict]] = []

    async def chat_stream(self, messages):
        self.calls.append(messages)
        answer = "first answer" if len(self.calls) == 1 else "second answer"
        yield json.dumps({"choices": [{"delta": {"content": answer}}]})
        yield "[DONE]"


class ChatMultiturnTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        self.headers = {"X-Tenant-Id": "default"}
        if settings.security.require_api_key:
            self.headers["X-API-Key"] = settings.security.api_key
        FakeContextPolicy.history = {}
        FakeLLMClient.calls = []

    def _post_chat(self, message: str, session_id: str | None = None):
        payload = {
            "message": message,
            "items": [],
            "options": {"allow_memory": True, "allow_extra_retrieval": False, "allow_auto_context": False},
        }
        if session_id:
            payload["session_id"] = session_id
        return self.client.post("/chat/", headers=self.headers, json=payload)

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

    def test_session_id_reuses_previous_turns_as_native_chat_messages(self):
        with patch("app.routers.chat.ContextPolicy", FakeContextPolicy), patch("app.routers.chat.LLMClient", FakeLLMClient):
            first = self._post_chat("What is aspirin?")
            self.assertEqual(first.status_code, 200)
            first_events = self._events(first)
            session_id = next(e["data"]["session_id"] for e in first_events if e["type"] == "final")

            second = self._post_chat("Continue with bleeding risk.", session_id=session_id)
            self.assertEqual(second.status_code, 200)

        self.assertEqual(len(FakeLLMClient.calls), 2)
        second_messages = FakeLLMClient.calls[1]
        second_events = self._events(second)
        roles_and_content = [(item["role"], item["content"]) for item in second_messages]

        self.assertIn(("user", "What is aspirin?"), roles_and_content)
        self.assertIn(("assistant", "first answer"), roles_and_content)
        self.assertEqual(next(e["data"]["session_id"] for e in second_events if e["type"] == "final"), session_id)
        self.assertEqual(second_messages[-1]["role"], "user")
        self.assertIn("Continue with bleeding risk.", second_messages[-1]["content"])


if __name__ == "__main__":
    unittest.main()
