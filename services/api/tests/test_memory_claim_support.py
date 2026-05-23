import unittest

from app.memory.claims import extract_atomic_claims
from app.memory.claim_support import assess_claim_support, build_evidence_table, evidence_table_debug_payload
from app.memory.evidence import gather_evidence_candidates


ASPIRIN_PREMISE = "Aspirin inhibits platelet aggregation and is used to reduce thrombotic risk."


async def fake_nli(premise: str, hypothesis: str):
    if "does not inhibit" in hypothesis:
        return {"label": "contradiction", "entailment": 0.01, "contradiction": 0.98, "neutral": 0.01}
    if "Aspirin inhibits platelet aggregation" in hypothesis:
        return {"label": "entailment", "entailment": 0.98, "contradiction": 0.01, "neutral": 0.01}
    return {"label": "neutral", "entailment": 0.03, "contradiction": 0.02, "neutral": 0.95}


class ClaimSupportTests(unittest.IsolatedAsyncioTestCase):
    async def test_mocked_entailment_and_contradiction(self):
        evidence = gather_evidence_candidates(
            prompt_context=[
                {
                    "paper_id": "paper-1",
                    "sent_id": "s1",
                    "sentence_text": ASPIRIN_PREMISE,
                }
            ]
        )
        claims = extract_atomic_claims("Aspirin inhibits platelet aggregation. Aspirin does not inhibit platelet aggregation.")
        support = await assess_claim_support(claims, evidence, nli_func=fake_nli)

        assert [item.status for item in support] == ["entailed", "contradicted"]
        assert support[0].prompt_supported is True
        assert support[1].needs_user_confirmation is True

    async def test_wrong_evidence_skips_nli_and_marks_unsupported(self):
        calls = []

        async def nli_should_not_run(premise: str, hypothesis: str):
            calls.append((premise, hypothesis))
            return {"label": "neutral", "entailment": 0.0, "contradiction": 0.0, "neutral": 1.0}

        claims = extract_atomic_claims("A cancer biomarker predicts chemotherapy response.")
        evidence = gather_evidence_candidates(
            source_sentences=[
                {
                    "paper_id": "paper-2",
                    "sent_id": "s2",
                    "sentence_text": "Study reports platelet aggregation in cardiovascular disease.",
                }
            ]
        )
        support = await assess_claim_support(claims, evidence, nli_func=nli_should_not_run)

        assert calls == []
        assert support[0].status == "unsupported"
        assert support[0].evidence[0].status == "not_comparable"

    async def test_no_comparable_source_sentence_is_unsupported(self):
        claims = extract_atomic_claims("Metformin improves insulin sensitivity.")
        support = await assess_claim_support(claims, [], nli_func=fake_nli)
        assert support[0].status == "unsupported"
        assert support[0].candidate_count == 0

    async def test_evidence_table_debug_payload_is_compact(self):
        claims = extract_atomic_claims("Aspirin inhibits platelet aggregation.")
        evidence = gather_evidence_candidates(prompt_context=[{"sentence_text": ASPIRIN_PREMISE, "sent_id": "s1"}])
        support = await assess_claim_support(claims, evidence, nli_func=fake_nli)
        table = build_evidence_table(answer_id="answer-1", session_id="session-1", turn_index=2, claim_support=support, tenant="default")
        debug = evidence_table_debug_payload(table)

        assert table["doc_type"] == "evidence_table"
        assert table["status_counts"]["entailed"] == 1
        assert debug["claims"][0]["status"] == "entailed"
