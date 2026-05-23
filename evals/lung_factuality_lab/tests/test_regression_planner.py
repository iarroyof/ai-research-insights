import unittest

from evals.lung_factuality_lab.src.regression_planner import build_regression_tests
from evals.lung_factuality_lab.src.schemas import FailureBoard, FailureItem


class RegressionPlannerTests(unittest.TestCase):
    def test_severe_failure_becomes_regression_test(self):
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

        tests = build_regression_tests(board)

        self.assertEqual(len(tests), 1)
        self.assertIn("inverted", tests[0].invariant)
        self.assertLessEqual(tests[0].max_allowed_reward, 0.05)


if __name__ == "__main__":
    unittest.main()

