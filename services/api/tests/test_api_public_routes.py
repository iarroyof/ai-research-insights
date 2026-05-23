import unittest

from fastapi.testclient import TestClient

from app.config import settings
from app.main import app


class ApiPublicRouteTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_root_is_public(self):
        response = self.client.get("/")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "ok")
        self.assertEqual(body["health"], "/health")

    def test_favicon_is_public_no_content(self):
        response = self.client.get("/favicon.ico")

        self.assertEqual(response.status_code, 204)

    def test_missing_tenant_returns_json_400(self):
        headers = {}
        if settings.security.require_api_key:
            headers["X-API-Key"] = settings.security.api_key

        response = self.client.get("/chat/memory/provider-metrics", headers=headers)

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json(), {"detail": "Missing X-Tenant-Id"})


if __name__ == "__main__":
    unittest.main()
