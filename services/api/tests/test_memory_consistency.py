import unittest

from app.memory.consistency import (
    build_conversation_frame,
    correction_terms,
    frame_alignment,
    longitudinal_consistency_report,
    render_conversation_frame,
)
from app.memory.rewards import reward_report


class MemoryConsistencyTests(unittest.TestCase):
    def test_correction_terms_are_generic_not_domain_specific(self):
        parsed = correction_terms("Please pivot from broad clinical outcomes to molecular pathway evidence.")

        self.assertIn("broad", parsed["avoid_terms"])
        self.assertIn("clinical", parsed["avoid_terms"])
        self.assertIn("molecular", parsed["preferred_terms"])
        self.assertIn("pathway", parsed["preferred_terms"])

    def test_conversation_frame_preserves_supported_claims_and_steering(self):
        frame = build_conversation_frame(
            existing={},
            question="Not clinical outcomes but molecular pathway evidence.",
            answer="EGFR signaling increases proliferation.",
            claim_support=[
                {
                    "status": "entailed",
                    "claim": "EGFR signaling increases proliferation.",
                }
            ],
            turn_index=3,
        )

        rendered = render_conversation_frame(frame)

        self.assertIn("molecular", frame["active_terms"])
        self.assertIn("clinical", frame["avoided_terms"])
        self.assertEqual(frame["supported_claims"][0]["claim"], "EGFR signaling increases proliferation.")
        self.assertIn("Preserve evidence-supported claims", rendered)

    def test_frame_alignment_penalizes_reusing_avoided_terms(self):
        frame = {
            "active_terms": ["molecular", "pathway"],
            "avoided_terms": ["clinical", "outcomes"],
        }

        aligned = frame_alignment(frame, "Molecular pathway evidence supports this mechanism.")
        drifting = frame_alignment(frame, "Clinical outcomes dominate the answer.")

        self.assertGreater(aligned["frame_alignment"], drifting["frame_alignment"])
        self.assertGreater(drifting["frame_drift_penalty"], 0.0)

    def test_longitudinal_report_detects_prior_supported_claim_negation(self):
        report = longitudinal_consistency_report(
            question="Continue.",
            answer="Aspirin does not inhibit platelet aggregation.",
            claim_support=[
                {
                    "status": "unsupported",
                    "claim": "Aspirin does not inhibit platelet aggregation.",
                }
            ],
            prior_supported_claims=[
                {
                    "claim": "Aspirin inhibits platelet aggregation.",
                }
            ],
            frame={"active_terms": ["aspirin", "platelet"], "avoided_terms": []},
        )

        self.assertEqual(report["prior_memory_conflict_count"], 1)
        self.assertTrue(any(item["type"] == "prior_memory_conflict" for item in report["warnings"]))

    def test_reward_uses_longitudinal_penalty_and_frame_alignment(self):
        base = reward_report(
            question="Continue platelet mechanism.",
            answer="Aspirin inhibits platelet aggregation.",
            selected_context=[{"text": "Aspirin inhibits platelet aggregation."}],
            conflicts=[],
            claim_support=[{"status": "entailed"}],
            longitudinal_consistency={"frame_alignment": 1.0, "longitudinal_penalty": 0.0},
            elapsed_sec=0.0,
            token_budget=1000,
        )
        penalized = reward_report(
            question="Continue platelet mechanism.",
            answer="Aspirin does not inhibit platelet aggregation.",
            selected_context=[{"text": "Aspirin inhibits platelet aggregation."}],
            conflicts=[],
            claim_support=[{"status": "unsupported"}],
            longitudinal_consistency={
                "frame_alignment": 0.0,
                "frame_drift_penalty": 0.25,
                "prior_memory_conflict_count": 1,
                "longitudinal_penalty": 0.65,
            },
            elapsed_sec=0.0,
            token_budget=1000,
        )

        self.assertGreater(base["score"], penalized["score"])
        self.assertGreater(penalized["prior_memory_conflict_penalty"], 0.0)


if __name__ == "__main__":
    unittest.main()
