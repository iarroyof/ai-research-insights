import unittest
from unittest.mock import patch

from app.search.os_client import os_hybrid_query, os_multilevel_query


class FakeClient:
    def __init__(self):
        self.calls = []

    def search(self, index, body):
        self.calls.append(body)
        if "multi_match" in body.get("query", {}):
            return {
                "hits": {
                    "hits": [
                        {
                            "_score": 0.01,
                            "_source": {
                                "article_id": "paper-1",
                                "subject": "PD-L1",
                                "relation": "associated with",
                                "object": "response",
                                "sentence_text": "PD-L1 is associated with response.",
                                "subject_probably_EBio": 0.9,
                                "object_probably_EBio": 0.9,
                            },
                        }
                    ]
                }
            }
        return {
            "hits": {
                "hits": [
                    {
                        "_score": 0.8,
                        "_source": {
                            "article_id": "paper-2",
                            "subject": "checkpoint inhibitor",
                            "relation": "treats",
                            "object": "NSCLC",
                            "sentence_text": "Checkpoint inhibitors are used in NSCLC.",
                            "subject_probably_EBio": 0.8,
                            "object_probably_EBio": 0.8,
                        },
                    }
                ]
            }
        }


class FakeMultilevelClient:
    def __init__(self):
        self.calls = []

    def search(self, index, body):
        self.calls.append({"index": index, "body": body})
        if "papers" not in index:
            raise Exception("skip non-paper index")
        return {
            "hits": {
                "hits": [
                    {
                        "_score": 7.0,
                        "_source": {
                            "paper_id": "paper-title-1",
                            "title": "PD-L1 biomarkers in immunotherapy",
                            "abstract": "A review of PD-L1 and checkpoint response.",
                        },
                    }
                ]
            }
        }


class SearchVectorFallbackTests(unittest.TestCase):
    def test_bm25_results_remain_primary_and_vector_fallback_appends_when_sparse(self):
        fake = FakeClient()
        with patch("app.search.os_client.os_client", return_value=fake):
            hits = os_hybrid_query(
                "default",
                "PD-L1 response",
                {
                    "query_vector": [0.1, 0.2, 0.3],
                    "allow_vector_fallback": True,
                    "fallback_min_results": 2,
                    "confidence_min": 0.1,
                },
                3,
            )

        self.assertEqual([item["retrieval_mode"] for item in hits], ["bm25", "vector_fallback"])
        self.assertEqual(hits[0]["paper_id"], "paper-1")
        self.assertGreaterEqual(len(fake.calls), 2)

    def test_vector_fallback_is_not_called_without_query_vector(self):
        fake = FakeClient()
        with patch("app.search.os_client.os_client", return_value=fake):
            hits = os_hybrid_query(
                "default",
                "PD-L1 response",
                {"allow_vector_fallback": True, "fallback_min_results": 2, "confidence_min": 0.1},
                3,
            )

        self.assertEqual([item["retrieval_mode"] for item in hits], ["bm25"])
        self.assertEqual(len(fake.calls), 1)

    def test_multilevel_title_search_uses_title_fields_and_normalizes_paper_hit(self):
        fake = FakeMultilevelClient()
        with patch("app.search.os_client.os_client", return_value=fake):
            hits = os_multilevel_query("default", "title", "PD-L1 immunotherapy", {}, 2)

        fields = fake.calls[0]["body"]["query"]["multi_match"]["fields"]
        self.assertIn("title^8", fields)
        self.assertEqual(hits[0]["search_level"], "title")
        self.assertEqual(hits[0]["paper_id"], "paper-title-1")
        self.assertEqual(hits[0]["text"], "PD-L1 biomarkers in immunotherapy")


if __name__ == "__main__":
    unittest.main()
