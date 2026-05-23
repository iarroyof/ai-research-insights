import unittest

from evals.lung_factuality_lab.src.recommendation_engine import build_recommendations
from evals.lung_factuality_lab.src.schemas import FailureBoard, FailureItem


class RecommendationEngineTests(unittest.TestCase):
    def test_recommendation_engine_emits_actionable_fix(self):
        board = FailureBoard(
            run_id="run-1",
            scenario_id="expert_hgf_met_direction_001",
            failure_summary={"failure_count": 1},
            failures=[
                FailureItem(
                    failure_id="fail_001",
                    turn=4,
                    severity=5,
                    category="factual_inversion",
                    short_description="Assistant accepted wrong direction.",
                    expected="Reject inversion.",
                    actual="HGF decreases MET.",
                    detected_by_evaluator=True,
                    penalized_sufficiently=False,
                    root_cause="answer_generation_failure_plus_weak_reward_penalty",
                    recommended_action_type="reward_weight_and_prompt_fix",
                    failure_owner="reward_weighting",
                )
            ],
        )

        recs = build_recommendations(board)

        self.assertTrue(any(rec.target == "reward_config" for rec in recs))
        self.assertTrue(any(rec.priority == "P0" for rec in recs))


if __name__ == "__main__":
    unittest.main()

