import unittest

from app.memory.lifecycle import (
    build_episodic_summary,
    classify_memory_state,
    estimate_tokens,
    select_working_set,
)


class MemoryLifecycleTests(unittest.TestCase):
    def test_select_working_set_respects_budget_and_preserves_evidence(self):
        docs = [
            {
                "doc_type": "message",
                "role": "user",
                "turn_index": 1,
                "text": "old filler " * 80,
                "importance": 0.1,
            },
            {
                "doc_type": "message",
                "role": "assistant",
                "turn_index": 2,
                "text": "Aspirin inhibits platelet aggregation in the cited study.",
                "summary": "Aspirin inhibits platelet aggregation.",
                "importance": 0.9,
                "evidence_supported": True,
            },
            {
                "doc_type": "message",
                "role": "user",
                "turn_index": 9,
                "text": "What about bleeding risk?",
                "importance": 0.6,
            },
        ]

        selection = select_working_set(docs, token_budget=26)
        selected_turns = [item.get("turn_index") for item in selection.selected]
        evicted_turns = [item.get("turn_index") for item in selection.evicted]

        self.assertIn(2, selected_turns)
        self.assertIn(9, selected_turns)
        self.assertIn(1, evicted_turns)
        self.assertLessEqual(selection.selected_tokens, 26)

    def test_classify_memory_state_promotes_landmarks_and_demotes_old_low_value_turns(self):
        landmark = {"doc_type": "landmark", "turn_index": 1, "summary": "accepted fact"}
        old_filler = {"doc_type": "message", "turn_index": 3, "importance": 0.05, "text": "thanks"}
        recent = {"doc_type": "message", "turn_index": 18, "importance": 0.4, "text": "new question"}

        self.assertEqual(classify_memory_state(landmark, current_turn_index=20, working_turns=4).memory_state, "promoted")
        self.assertEqual(classify_memory_state(old_filler, current_turn_index=20, working_turns=4).memory_state, "evicted")
        self.assertEqual(classify_memory_state(recent, current_turn_index=20, working_turns=4).memory_state, "working")

    def test_build_episodic_summary_keeps_focus_and_supported_facts(self):
        summary = build_episodic_summary(
            tenant="default",
            session_id="session-1",
            turn_index=6,
            messages=[
                {
                    "role": "user",
                    "turn_index": 4,
                    "text": "Does aspirin inhibit platelet aggregation?",
                    "importance": 0.6,
                },
                {
                    "role": "assistant",
                    "turn_index": 5,
                    "text": "Aspirin inhibits platelet aggregation.",
                    "importance": 0.9,
                    "evidence_supported": True,
                },
            ],
            reward_score=0.8,
        )

        self.assertEqual(summary["doc_type"], "episodic_summary")
        self.assertEqual(summary["memory_state"], "episodic")
        self.assertIn("aspirin", summary["terms"])
        self.assertIn("Evidence-supported facts", summary["summary"])
        self.assertGreater(estimate_tokens(summary["summary"]), 1)


if __name__ == "__main__":
    unittest.main()
