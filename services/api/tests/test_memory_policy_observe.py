import unittest
from unittest.mock import patch

from app.memory.claim_support import ClaimSupport
from app.memory.policy import ContextPolicy


class FakeStore:
    def __init__(self):
        self.messages = []
        self.landmarks = []
        self.evidence_tables = []
        self.traces = []
        self.episodic_summaries = []
        self.lifecycle_updates = []
        self.idea_updates = []
        self.action_updates = []
        self.frames = []

    async def add_message(self, **kwargs):
        self.messages.append(kwargs)

    async def update_landmarks(self, *args, **kwargs):
        self.landmarks.append((args, kwargs))

    async def add_evidence_table(self, **kwargs):
        self.evidence_tables.append(kwargs)

    async def add_trace(self, **kwargs):
        self.traces.append(kwargs)

    async def add_episodic_summary(self, **kwargs):
        self.episodic_summaries.append(kwargs)

    async def update_memory_lifecycle(self, **kwargs):
        self.lifecycle_updates.append(kwargs)

    async def update_idea_index(self, **kwargs):
        self.idea_updates.append(kwargs)

    async def update_action_value(self, **kwargs):
        self.action_updates.append(kwargs)

    async def conversation_frame(self, session_id):
        return self.frames[-1] if self.frames else {}

    async def supported_claim_evidence(self, session_id, limit=20):
        return [
            {
                "source": "memory_claim",
                "sentence_text": "Aspirin inhibits platelet aggregation.",
                "text": "Aspirin inhibits platelet aggregation.",
                "evidence_id": "prior-claim-1",
            }
        ]

    async def update_conversation_frame(self, **kwargs):
        frame = {
            "summary": "active=aspirin, platelet",
            "active_terms": ["aspirin", "platelet"],
            "avoided_terms": [],
            "supported_claims": [{"claim": "Aspirin inhibits platelet aggregation."}],
            "contradicted_claims": [],
        }
        self.frames.append(frame)
        return frame


async def fake_assess_claim_support(claims, evidence_candidates, **kwargs):
    return [
        ClaimSupport(
            claim_id="claim-1",
            claim="Aspirin inhibits platelet aggregation.",
            answer_sentence="Aspirin inhibits platelet aggregation.",
            requires_citation=True,
            status="entailed",
            best_entailment=0.98,
            best_evidence_id="ev-1",
            candidate_count=len(list(evidence_candidates)),
            prompt_supported=True,
        )
    ]


class PolicyObserveTests(unittest.IsolatedAsyncioTestCase):
    async def test_observe_turn_persists_evidence_idea_and_action_traces(self):
        policy = ContextPolicy("default")
        fake_store = FakeStore()
        policy.store = fake_store

        with patch("app.memory.policy.assess_claim_support", side_effect=fake_assess_claim_support), patch(
            "app.memory.policy.extract_triples", None
        ):
            trace = await policy.observe_turn(
                session_id="session-1",
                turn_index=2,
                question="Does aspirin inhibit platelet aggregation?",
                answer="Aspirin inhibits platelet aggregation.",
                selected_context=[
                    {
                        "source": "pinned",
                        "sentence_text": "Aspirin inhibits platelet aggregation.",
                        "paper_id": "paper-1",
                        "sent_id": "s1",
                    }
                ],
                retrieved_triplets=[],
                pinned_snippets=[],
                source_sentences=[],
                started_at=0.0,
                token_budget=2000,
            )

        self.assertEqual(len(fake_store.messages), 2)
        self.assertEqual(len(fake_store.evidence_tables), 1)
        self.assertEqual(len(fake_store.traces), 1)
        self.assertEqual(len(fake_store.episodic_summaries), 1)
        self.assertEqual(len(fake_store.lifecycle_updates), 1)
        self.assertEqual(len(fake_store.idea_updates), 1)
        self.assertEqual(len(fake_store.action_updates), 1)
        self.assertEqual(len(fake_store.frames), 1)
        self.assertEqual(trace["evidence_table"]["status_counts"]["entailed"], 1)
        self.assertIn("action_key", trace["action"])
        self.assertGreater(trace["reward"]["citation_coverage"], 0.0)
        self.assertIn("longitudinal_consistency", trace)
        self.assertIn("conversation_frame", trace)


if __name__ == "__main__":
    unittest.main()
