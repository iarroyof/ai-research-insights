import unittest

from app.memory.claims import extract_atomic_claims, split_candidate_sentences


class ClaimExtractionTests(unittest.TestCase):
    def test_split_candidate_sentences_keeps_biomedical_abbreviations(self):
        text = "Aspirin inhibits platelet aggregation, e.g. in platelet assays. Metformin improves insulin sensitivity."
        sentences = split_candidate_sentences(text)
        self.assertEqual(
            [s.text for s in sentences],
            [
                "Aspirin inhibits platelet aggregation, e.g. in platelet assays.",
                "Metformin improves insulin sensitivity.",
            ],
        )

    def test_extract_atomic_claims_preserves_original_sentence_and_citation_flag(self):
        answer = "Aspirin inhibits platelet aggregation and is used to reduce thrombotic risk. Let me know if you want details."
        claims = extract_atomic_claims(answer)
        self.assertEqual(
            [claim.claim_text for claim in claims],
            [
                "Aspirin inhibits platelet aggregation.",
                "Aspirin is used to reduce thrombotic risk.",
            ],
        )
        self.assertTrue(all(claim.answer_sentence == "Aspirin inhibits platelet aggregation and is used to reduce thrombotic risk." for claim in claims))
        self.assertTrue(all(claim.requires_citation for claim in claims))
        self.assertTrue(all(claim.claim_id.startswith("claim_") for claim in claims))
