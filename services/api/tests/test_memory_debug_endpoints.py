import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.config import settings
from app.main import app


class FakeDebugStore:
    def __init__(self, tenant: str):
        self.tenant = tenant

    async def debug_ideas(self, session_id=None, limit=20):
        return [
            {
                "doc_type": "idea",
                "scope": session_id or "shared",
                "idea": "aspirin",
                "frequency": 3,
                "importance": 0.8,
            }
        ][:limit]

    async def debug_action_values(self, *, session_id=None, state_key=None, limit=20):
        return [
            {
                "doc_type": "action_value",
                "scope": session_id or "shared",
                "state_key": state_key or "state:any",
                "action_key": "ctx:recent",
                "q_value": 0.62,
                "visits": 2,
            }
        ][:limit]

    async def evidence_tables(self, session_id=None, limit=10):
        return [
            {
                "doc_type": "evidence_table",
                "session_id": session_id or "session-1",
                "answer_id": "answer-1",
                "claims": [{"claim_id": "claim-1", "status": "entailed"}],
            }
        ][:limit]


class MemoryDebugEndpointTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        self.headers = {"X-Tenant-Id": "default"}
        if settings.security.require_api_key:
            self.headers["X-API-Key"] = settings.security.api_key

    def test_idea_debug_endpoint_returns_tenant_scoped_items(self):
        with patch("app.routers.chat.MemoryStore", FakeDebugStore):
            response = self.client.get(
                "/chat/memory/ideas",
                headers=self.headers,
                params={"session_id": "session-1", "limit": 5},
            )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["tenant"], "default")
        self.assertEqual(body["session_id"], "session-1")
        self.assertEqual(body["items"][0]["idea"], "aspirin")

    def test_action_value_debug_endpoint_returns_state_filter(self):
        with patch("app.routers.chat.MemoryStore", FakeDebugStore):
            response = self.client.get(
                "/chat/memory/action-values",
                headers=self.headers,
                params={"session_id": "session-1", "state_key": "state:aspirin"},
            )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["state_key"], "state:aspirin")
        self.assertEqual(body["items"][0]["action_key"], "ctx:recent")

    def test_evidence_table_debug_endpoint_returns_recent_tables(self):
        with patch("app.routers.chat.MemoryStore", FakeDebugStore):
            response = self.client.get(
                "/chat/memory/evidence-tables",
                headers=self.headers,
                params={"session_id": "session-1"},
            )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["items"][0]["answer_id"], "answer-1")
        self.assertEqual(body["items"][0]["claims"][0]["status"], "entailed")


if __name__ == "__main__":
    unittest.main()
