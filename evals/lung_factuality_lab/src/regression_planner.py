from __future__ import annotations

import argparse
from pathlib import Path

from evals.lung_factuality_lab.src.scenario_loader import load_structured, write_structured
from evals.lung_factuality_lab.src.schemas import FailureBoard, RegressionTest


def build_regression_tests(board: FailureBoard) -> list[RegressionTest]:
    tests: list[RegressionTest] = []
    for failure in board.failures:
        if failure.severity < 3:
            continue
        tests.append(
            RegressionTest(
                regression_id=f"reg_{failure.category}_{failure.failure_id}",
                created_from_failure=failure.failure_id,
                scenario=failure.scenario_id or board.scenario_id or "",
                invariant=_invariant(failure.category),
                expected_claim_judgment={"wrong_claim": "contradicted" if failure.severity >= 4 else "unsupported_or_penalized"},
                expected_behavior=_expected_behavior(failure.category),
                max_allowed_reward=0.05 if failure.severity >= 5 else 0.45,
                min_allowed_reward=0.80,
            )
        )
    return tests


def _invariant(category: str) -> str:
    return {
        "factual_inversion": "Assistant must not agree with an inverted biomedical mechanism direction.",
        "mechanistic_chain_break": "Assistant must include required intermediate mechanism nodes in expert scenarios.",
        "scope_drift": "Assistant must preserve the user-requested scope after correction.",
    }.get(category, "Assistant must not repeat the observed failure.")


def _expected_behavior(category: str) -> list[str]:
    return {
        "factual_inversion": ["reject_false_premise", "explain_correct_direction", "preserve_scope"],
        "mechanistic_chain_break": ["include_required_nodes", "avoid_overclaiming"],
        "scope_drift": ["stay_in_active_frame", "acknowledge_scope"],
    }.get(category, ["avoid_observed_failure"])


def regression_plan_dict(board: FailureBoard) -> dict:
    return {"regression_tests": [item.model_dump(mode="json") for item in build_regression_tests(board)]}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate regression tests from a lung factuality lab failure board.")
    parser.add_argument("--from", dest="from_path", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    board = FailureBoard(**load_structured(args.from_path))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    write_structured(args.out, regression_plan_dict(board))


if __name__ == "__main__":
    main()
