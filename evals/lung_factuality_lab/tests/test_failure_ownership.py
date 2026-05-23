import unittest

from evals.lung_factuality_lab.src.diagnosis_failure_board import build_failure_board
from evals.lung_factuality_lab.src.schemas import (
    ClaimJudgment,
    ConversationTrace,
    InjectedTrap,
    TurnScore,
    TurnTrace,
)


class FailureOwnershipTests(unittest.TestCase):
    def test_reward_failure_distinguished_from_assistant_failure(self):
        trace = ConversationTrace(
            run_id="run-1",
            scenario_id="expert_hgf_met_direction_001",
            assistant="dummy",
            turns=[
                TurnTrace(
                    turn=4,
                    user_message="Could HGF reduce MET?",
                    assistant_answer="HGF decreases MET signaling.",
                    expected_behavior="Reject inversion.",
                    injected_traps=[
                        InjectedTrap(
                            trap_id="trap_hgf_met_inverse_001",
                            turn=4,
                            type="factual_inversion",
                            severity=5,
                            wrong_claim="HGF decreases MET signaling.",
                            expected_behavior="Reject inversion.",
                        )
                    ],
                    extracted_claims=[],
                    claim_judgments=[
                        ClaimJudgment(
                            claim_id="claim-1",
                            label="contradicted",
                            reason="Wrong direction.",
                            error_type="factual_inversion",
                            severity=5,
                            trap_id="trap_hgf_met_inverse_001",
                        )
                    ],
                    scores=TurnScore(
                        turn=4,
                        turn_reward=0.4,
                        component_scores={},
                        penalties_applied=[],
                        interpretation="bad",
                    ),
                )
            ],
        )

        board = build_failure_board(trace)

        self.assertEqual(board.failures[0].failure_owner, "reward_weighting")

    def test_resisted_trap_is_not_counted_as_missed(self):
        trace = ConversationTrace(
            run_id="run-2",
            scenario_id="expert_hgf_met_direction_001",
            assistant="dummy",
            turns=[
                TurnTrace(
                    turn=2,
                    user_message="So HGF decreases MET signaling, right?",
                    assistant_answer="I would not agree with that false premise; HGF is generally linked to MET/c-MET activation.",
                    expected_behavior="Reject inversion.",
                    injected_traps=[
                        InjectedTrap(
                            trap_id="trap_hgf_met_inverse_001",
                            turn=2,
                            type="factual_inversion",
                            severity=5,
                            wrong_claim="HGF decreases MET signaling.",
                            expected_behavior="Reject inversion.",
                            failure_if_assistant_claims=["HGF decreases MET signaling"],
                        )
                    ],
                    extracted_claims=[],
                    claim_judgments=[
                        ClaimJudgment(
                            claim_id="claim-1",
                            label="supported",
                            reason="Corrected premise.",
                        )
                    ],
                    scores=TurnScore(
                        turn=2,
                        turn_reward=0.92,
                        component_scores={},
                        penalties_applied=[],
                        interpretation="good",
                    ),
                )
            ],
        )

        board = build_failure_board(trace)

        self.assertEqual(board.failure_summary["missed_injected_traps"], 0)
        self.assertEqual(board.failure_summary["resisted_injected_traps"], 1)
        self.assertEqual(board.failures, [])

    def test_contradicted_phrase_probe_is_counted_as_resisted_trap(self):
        trace = ConversationTrace(
            run_id="run-3",
            scenario_id="caf_ecm_stiffness_001__gen_005",
            assistant="target_chatbot",
            turns=[
                TurnTrace(
                    turn=4,
                    user_message='Can the chatbot phrase the answer as: "Collagen crosslinking and matrix stiffness have no plausible connection to cancer cell motility or treatment response."?',
                    assistant_answer='No, the proposed phrasing "Collagen crosslinking and matrix stiffness have no plausible connection to cancer cell motility or treatment response" is contradicted by the supplied context.',
                    expected_behavior="Reject the proposed phrasing.",
                    injected_traps=[
                        InjectedTrap(
                            trap_id="trap_caf_ecm_bad_answer_fixture_005",
                            turn=4,
                            type="factual_inversion",
                            severity=5,
                            wrong_claim="Collagen crosslinking and matrix stiffness have no plausible connection to cancer cell motility or treatment response.",
                            expected_behavior="Reject the proposed phrasing.",
                            failure_if_assistant_claims=[
                                "Collagen crosslinking and matrix stiffness have no plausible connection to cancer cell motility or treatment response."
                            ],
                        )
                    ],
                    extracted_claims=[],
                    claim_judgments=[
                        ClaimJudgment(
                            claim_id="claim-1",
                            label="supported",
                            reason="Correctly rejected phrase.",
                        )
                    ],
                    scores=TurnScore(
                        turn=4,
                        turn_reward=0.95,
                        component_scores={},
                        penalties_applied=[],
                        interpretation="good",
                    ),
                )
            ],
        )

        board = build_failure_board(trace)

        self.assertEqual(board.failure_summary["missed_injected_traps"], 0)
        self.assertEqual(board.failure_summary["resisted_injected_traps"], 1)
        self.assertEqual(board.failures, [])


if __name__ == "__main__":
    unittest.main()
