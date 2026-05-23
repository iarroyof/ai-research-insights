import unittest

from app.memory.evaluation import simulate_quality_by_session
from app.memory.lifecycle import compress_text_for_budget, select_working_set
from app.memory.rewards import context_support_score, reward_report


class MemoryEvaluationTests(unittest.TestCase):
    def test_reward_penalizes_off_topic_math_synergy_for_mechanistic_question(self):
        report = reward_report(
            question="What is the functional synergy that defines aggressive lung carcinoma?",
            answer="CI value less than 1 defines drug synergy and antagonism.",
            selected_context=[{"text": "CI value less than 1 defines synergy."}],
            conflicts=[],
            claim_support=[],
            elapsed_sec=0.0,
            token_budget=1000,
        )

        self.assertGreater(report["off_topic_penalty"], 0.0)

    def test_reward_tracks_search_evidence_assembly_quality(self):
        report = reward_report(
            question="How might a metabolic exposure affect tissue acidity?",
            answer="The retrieved evidence supports a partial mechanism.",
            selected_context=[{"text": "Metabolic lactate accumulation can lower extracellular pH."}],
            conflicts=[],
            claim_support=[],
            search_plan={
                "evidence_assembly": {
                    "assembly_quality": 0.75,
                    "refinement_quality": {
                        "accepted_feedback_term_count": 3,
                        "rejected_feedback_term_count": 2,
                        "ungrounded_feedback_term_count": 0,
                    },
                }
            },
            elapsed_sec=0.0,
            token_budget=1000,
        )

        self.assertEqual(report["evidence_assembly_quality"], 0.75)
        self.assertEqual(report["search_feedback_rejected_count"], 2)
        self.assertEqual(report["search_query_drift_penalty"], 0.0)

    def test_reward_penalizes_unsupported_bridge_when_evidence_puzzle_has_missing_edges(self):
        base_args = {
            "question": "How might diet and local pH relate to tissue growth?",
            "answer": "A partial structure is supported, but the diet-to-pH edge is not established here.",
            "selected_context": [{"text": "Lactate can lower extracellular pH."}],
            "conflicts": [],
            "elapsed_sec": 0.0,
            "token_budget": 1000,
        }
        partial = reward_report(
            **base_args,
            claim_support=[{"status": "entailed"}],
            search_plan={"evidence_assembly": {"evidence_puzzle": {"edge_support_status": "partial"}}},
        )
        bridge = reward_report(
            **{key: value for key, value in base_args.items() if key != "answer"},
            answer="Diet changes body pH, drives lactate acidosis, and therefore promotes tissue growth.",
            claim_support=[{"status": "unsupported"}, {"status": "unsupported"}],
            search_plan={"evidence_assembly": {"evidence_puzzle": {"edge_support_status": "missing"}}},
        )

        self.assertGreater(bridge["unsupported_evidence_bridge_penalty"], 0.0)
        self.assertGreater(partial["score"], bridge["score"])

    def test_working_set_prioritizes_evidence_supported_memory_over_recent_low_value(self):
        docs = [
            {
                "doc_type": "message",
                "role": "assistant",
                "turn_index": 1,
                "text": "Aspirin inhibits platelet aggregation.",
                "importance": 0.9,
                "evidence_supported": True,
            },
            {
                "doc_type": "message",
                "role": "assistant",
                "turn_index": 20,
                "text": "Thanks.",
                "importance": 0.1,
            },
        ]

        selected = select_working_set(docs, token_budget=20, current_turn_index=21, working_turns=2).selected

        self.assertEqual(selected[0]["memory_state"], "promoted")
        self.assertIn("Aspirin", selected[0]["text"])

    def test_context_support_uses_best_supporting_item(self):
        answer = "Hypoxia and stromal immune crosstalk drive tumor growth."
        exact = [{"text": "Hypoxia and stromal immune crosstalk drive tumor growth."}]
        with_extra_summary = exact + [
            {
                "text": "Session episode through turn 1. Focus: unrelated therapy toxicity adverse events dose response.",
            }
        ]

        self.assertEqual(context_support_score(answer, exact), context_support_score(answer, with_extra_summary))

    def test_compress_text_for_budget_returns_short_extractive_summary(self):
        text = " ".join(["Tumor microenvironment crosstalk promotes invasion."] * 20)
        summary = compress_text_for_budget(text, 30)

        self.assertLessEqual(len(summary.split()), len(text.split()))
        self.assertTrue(summary)

    def test_simulation_reports_current_policy_delta(self):
        docs = [
            {"doc_type": "message", "session_id": "s1", "role": "user", "turn_index": 0, "text": "What drives TME growth?"},
            {
                "doc_type": "message",
                "session_id": "s1",
                "role": "assistant",
                "turn_index": 1,
                "text": "Hypoxia and stromal immune crosstalk drive tumor growth.",
                "importance": 0.8,
                "evidence_supported": True,
            },
            {"doc_type": "message", "session_id": "s1", "role": "user", "turn_index": 2, "text": "Continue."},
            {
                "doc_type": "message",
                "session_id": "s1",
                "role": "assistant",
                "turn_index": 3,
                "text": "Hypoxia and stromal immune crosstalk drive tumor growth.",
            },
        ]
        report = simulate_quality_by_session(docs, token_budget=120)

        self.assertEqual(report["evaluated_session_count"], 1)
        self.assertIn("reward_delta", report["aggregate"])


if __name__ == "__main__":
    unittest.main()
