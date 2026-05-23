import unittest

from app.memory.evidence import gather_evidence_candidates


class EvidenceCandidateTests(unittest.TestCase):
    def test_gather_evidence_candidates_preserves_prompt_and_triplet_provenance(self):
        candidates = gather_evidence_candidates(
            prompt_context=[
                {
                    "source": "prompt_context",
                    "paper_id": "paper-1",
                    "pmid": "123",
                    "title": "Aspirin paper",
                    "section": "Results",
                    "sent_id": "s1",
                    "sentence_text": "Aspirin inhibits platelet aggregation.",
                    "window_text": "Aspirin inhibits platelet aggregation. It reduces thrombotic risk.",
                    "_score": 3.1,
                }
            ],
            pinned_snippets=[
                {
                    "paper_id": "paper-2",
                    "pmcid": "PMC1",
                    "text": "Pinned source sentence.",
                }
            ],
            triplet_results=[
                {
                    "article_id": "paper-3",
                    "sent_id": "s3",
                    "sentence_text": "Metformin improves insulin sensitivity.",
                    "subject": "Metformin",
                    "relation": "improves",
                    "object": "insulin sensitivity",
                    "confidence": 0.91,
                }
            ],
        )

        self.assertEqual(len(candidates), 3)
        prompt = candidates[0]
        self.assertEqual(prompt.paper_id, "paper-1")
        self.assertEqual(prompt.pmid, "123")
        self.assertEqual(prompt.section, "Results")
        self.assertIs(prompt.was_in_model_prompt, True)
        self.assertEqual(prompt.retrieval_score, 3.1)

        pinned = candidates[1]
        self.assertEqual(pinned.source, "pinned")
        self.assertIs(pinned.was_in_model_prompt, True)
        self.assertEqual(pinned.pmcid, "PMC1")

        triplet = candidates[2]
        self.assertEqual(triplet.source, "triplet")
        self.assertEqual(triplet.paper_id, "paper-3")
        self.assertEqual(triplet.triplet_links[0]["subject"], "Metformin")
        self.assertEqual(triplet.retrieval_score, 0.91)
