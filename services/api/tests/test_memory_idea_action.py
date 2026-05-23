import unittest

from app.memory.action_value import action_key, state_key, update_action_value
from app.memory.idea_index import build_idea_updates, merge_idea_doc, normalize_idea, rank_ideas
from app.memory.rewards import reward_report


class IdeaActionMemoryTests(unittest.TestCase):
    def test_idea_updates_merge_frequency_reward_and_cooccurrence(self):
        updates = build_idea_updates(
            tenant="default",
            session_id="session-1",
            texts=["Aspirin inhibits platelet aggregation.", "Platelet aggregation matters."],
            turn_index=4,
            reward_score=0.8,
        )

        aspirin = next(item for item in updates if item["idea"] == "aspirin")
        doc = merge_idea_doc(None, aspirin)
        doc2 = merge_idea_doc(doc, aspirin)

        self.assertEqual(doc2["doc_type"], "idea")
        self.assertEqual(doc2["frequency"], 2)
        self.assertEqual(doc2["reward_count"], 2)
        self.assertAlmostEqual(doc2["reward_avg"], 0.8)
        self.assertIn("platelet", doc2["cooccurring_ideas"])
        self.assertGreater(doc2["importance"], 0.0)

    def test_idea_updates_add_synonyms_and_parent_child_edges(self):
        updates = build_idea_updates(
            tenant="default",
            session_id="session-1",
            texts=[
                "PD-1 checkpoint inhibitors are used in non-small cell lung cancer.",
                "Platelet aggregation is a measurable endpoint.",
            ],
            turn_index=7,
            reward_score=0.7,
        )
        ideas = {item["idea"]: item for item in updates}

        self.assertEqual(normalize_idea("programmed cell death protein 1"), "pd1")
        self.assertIn("pd1", ideas)
        self.assertIn("nsclc", ideas)
        self.assertEqual(ideas["pd1"]["parent_idea"], "immune checkpoint")
        self.assertEqual(ideas["nsclc"]["parent_idea"], "lung cancer")
        self.assertEqual(ideas["platelet aggregation"]["parent_idea"], "platelet")

        platelet_doc = merge_idea_doc(None, ideas["platelet"])
        self.assertIn("platelet aggregation", platelet_doc["child_ideas"])
        self.assertIn("platelet", platelet_doc["concept_path"])

    def test_rank_ideas_prefers_query_overlap_and_importance(self):
        ideas = [
            {"idea": "aspirin", "importance": 0.2, "cooccurring_ideas": ["platelet"]},
            {"idea": "metformin", "importance": 0.9, "cooccurring_ideas": []},
        ]

        ranked = rank_ideas(ideas, "Does aspirin affect platelets?", limit=2)

        self.assertEqual(ranked[0]["idea"], "aspirin")
        self.assertGreater(ranked[0]["_idea_score"], ranked[1]["_idea_score"])

    def test_rank_ideas_uses_synonym_normalization(self):
        ideas = [
            {"idea": "pd1", "normalized_idea": "pd1", "importance": 0.2, "synonyms": ["programmed death 1"]},
            {"idea": "metformin", "normalized_idea": "metformin", "importance": 0.9, "synonyms": []},
        ]

        ranked = rank_ideas(ideas, "What does PD-1 blockade do?", limit=2)

        self.assertEqual(ranked[0]["idea"], "pd1")

    def test_normalize_idea_does_not_truncate_ous_words(self):
        self.assertEqual(normalize_idea("squamous"), "squamous")
        self.assertEqual(normalize_idea("continuous"), "continuous")

    def test_action_value_is_q_like_incremental_estimate(self):
        state = state_key(["aspirin", "platelet", "aspirin"])
        action = action_key(
            {
                "selected_context_count": 3,
                "retrieved_triplet_count": 2,
                "selected_idea_count": 1,
                "web_result_count": 0,
                "evidence_candidate_count": 4,
            }
        )
        doc = update_action_value(None, tenant="default", scope="session-1", state=state, action=action, reward=0.8)
        doc = update_action_value(doc, tenant="default", scope="session-1", state=state, action=action, reward=0.4)

        self.assertEqual(doc["doc_type"], "action_value")
        self.assertEqual(doc["visits"], 2)
        self.assertAlmostEqual(doc["reward_avg"], 0.6)
        self.assertGreater(doc["q_value"], 0.0)

    def test_reward_report_penalizes_claim_contradictions(self):
        base = reward_report(
            question="Does aspirin inhibit platelet aggregation?",
            answer="Aspirin inhibits platelet aggregation.",
            selected_context=[{"text": "Aspirin inhibits platelet aggregation."}],
            conflicts=[],
            claim_support=[{"status": "entailed"}],
            elapsed_sec=0.1,
            token_budget=2000,
        )
        contradicted = reward_report(
            question="Does aspirin inhibit platelet aggregation?",
            answer="Aspirin inhibits platelet aggregation.",
            selected_context=[{"text": "Aspirin inhibits platelet aggregation."}],
            conflicts=[],
            claim_support=[{"status": "contradicted"}],
            elapsed_sec=0.1,
            token_budget=2000,
        )

        self.assertGreater(base["score"], contradicted["score"])
        self.assertEqual(contradicted["claim_contradicted_count"], 1)


if __name__ == "__main__":
    unittest.main()
