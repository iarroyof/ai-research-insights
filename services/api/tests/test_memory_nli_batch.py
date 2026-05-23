import unittest
from unittest.mock import patch

from app.config import settings


class FakeResponse:
    status_code = 200
    headers = {}

    def __init__(self, data):
        self._data = data

    def json(self):
        return self._data

    def raise_for_status(self):
        return None


class FakeAsyncClient:
    posts = []

    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def post(self, url, json, headers):
        self.posts.append({"url": url, "json": json, "headers": headers})
        return FakeResponse(
            [
                [
                    {"label": "entailment", "score": 0.91},
                    {"label": "neutral", "score": 0.08},
                    {"label": "contradiction", "score": 0.01},
                ],
                [
                    {"label": "contradiction", "score": 0.83},
                    {"label": "neutral", "score": 0.10},
                    {"label": "entailment", "score": 0.07},
                ],
            ]
        )


class MemoryNliBatchTests(unittest.IsolatedAsyncioTestCase):
    async def test_hf_api_nli_batch_sends_one_request_for_multiple_pairs(self):
        from app.memory import nli

        old_token = settings.memory.hf_api_token
        old_provider = settings.memory.nli_provider
        settings.memory.hf_api_token = "hf_test_token"
        settings.memory.nli_provider = "hf_api"
        FakeAsyncClient.posts = []
        try:
            with patch("httpx.AsyncClient", FakeAsyncClient):
                results = await nli._hf_api_nli_batch(
                    [
                        ("Aspirin inhibits platelet aggregation.", "Aspirin inhibits platelet aggregation."),
                        ("Aspirin inhibits platelet aggregation.", "Aspirin does not inhibit platelet aggregation."),
                    ]
                )
        finally:
            settings.memory.hf_api_token = old_token
            settings.memory.nli_provider = old_provider

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["label"], "entailment")
        self.assertEqual(results[1]["label"], "contradiction")
        self.assertEqual(len(FakeAsyncClient.posts), 1)
        self.assertIsInstance(FakeAsyncClient.posts[0]["json"]["inputs"], list)
        self.assertEqual(len(FakeAsyncClient.posts[0]["json"]["inputs"]), 2)

    async def test_score_answer_triples_batches_selected_pairs(self):
        from app.memory import nli

        calls = []

        async def fake_batch(pairs):
            calls.append(pairs)
            return [
                {"label": "entailment", "entailment": 0.9, "contradiction": 0.02, "neutral": 0.08, "provider": "fake"}
                for _ in pairs
            ]

        answer_triples = [{"subject": "Aspirin", "relation": "inhibits", "object": "platelet aggregation"}]
        retrieved_triplets = [
            {
                "subject": "Aspirin",
                "relation": "inhibits",
                "object": "platelet aggregation",
                "sentence_text": "Aspirin inhibits platelet aggregation.",
            },
            {
                "subject": "Aspirin",
                "relation": "reduces",
                "object": "thrombotic risk",
                "sentence_text": "Aspirin reduces thrombotic risk.",
            },
        ]
        with patch("app.memory.nli.classify_nli_batch", side_effect=fake_batch):
            result = await nli.score_answer_triples(answer_triples, retrieved_triplets, max_pairs=2)

        self.assertEqual(len(calls), 1)
        self.assertGreaterEqual(len(calls[0]), 1)
        self.assertEqual(result[0]["provider"], "fake")

    async def test_claim_support_uses_batch_nli_when_default_classifier_is_used(self):
        from app.memory.claim_support import assess_claim_support

        calls = []

        async def fake_batch(pairs):
            calls.append(pairs)
            return [
                {"label": "entailment", "entailment": 0.9, "contradiction": 0.02, "neutral": 0.08}
                for _ in pairs
            ]

        claims = [{"claim_id": "c1", "claim": "Aspirin inhibits platelet aggregation.", "requires_citation": True}]
        evidence = [
            {
                "evidence_id": "e1",
                "text": "Aspirin inhibits platelet aggregation.",
                "subject": "Aspirin",
                "object": "platelet aggregation",
            }
        ]

        result = await assess_claim_support(claims, evidence, nli_batch_func=fake_batch)

        self.assertEqual(len(calls), 1)
        self.assertEqual(result[0].status, "entailed")


if __name__ == "__main__":
    unittest.main()
