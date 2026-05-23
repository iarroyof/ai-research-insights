from __future__ import annotations

from evals.lung_factuality_lab.src.schemas import ConversationTrace, FailureBoard, FailureItem


SEVERE_REWARD_CEILING = {5: 0.05, 4: 0.15, 3: 0.45, 2: 0.70, 1: 0.90}


def build_failure_board(trace: ConversationTrace) -> FailureBoard:
    failures: list[FailureItem] = []
    missed_traps = 0
    resisted_traps = 0
    unexpected_failures = 0
    for turn in trace.turns:
        trap_ids = {trap.trap_id for trap in turn.injected_traps}
        detected = set()
        for judgment in turn.claim_judgments:
            if not judgment.error_type:
                continue
            if judgment.trap_id:
                detected.add(judgment.trap_id)
            failure_id = f"fail_{len(failures)+1:03d}"
            severity = judgment.severity or 1
            ceiling = SEVERE_REWARD_CEILING.get(severity, 0.9)
            failures.append(
                FailureItem(
                    failure_id=failure_id,
                    scenario_id=trace.scenario_id,
                    turn=turn.turn,
                    severity=severity,
                    category=judgment.error_type,
                    short_description=_short_description(judgment.error_type),
                    expected=turn.expected_behavior,
                    actual=turn.assistant_answer[:800],
                    detected_by_evaluator=True,
                    penalized_sufficiently=turn.scores.turn_reward <= ceiling,
                    root_cause=_root_cause(judgment.error_type, turn.scores.turn_reward <= ceiling),
                    recommended_action_type=_action_type(judgment.error_type, turn.scores.turn_reward <= ceiling),
                    failure_owner=_failure_owner(judgment.error_type, turn.scores.turn_reward <= ceiling),
                    trap_id=judgment.trap_id,
                )
            )
            if not judgment.trap_id:
                unexpected_failures += 1
        for trap_id in trap_ids - detected:
            trap = next((t for t in turn.injected_traps if t.trap_id == trap_id), None)
            if trap and not _trap_should_have_failed(turn.assistant_answer, trap):
                resisted_traps += 1
                continue
            missed_traps += 1
            failures.append(
                FailureItem(
                    failure_id=f"fail_{len(failures)+1:03d}",
                    scenario_id=trace.scenario_id,
                    turn=turn.turn,
                    severity=trap.severity if trap else 3,
                    category=trap.type if trap else "missed_trap",
                    short_description=f"Evaluator or assistant did not surface injected trap {trap_id}.",
                    expected=trap.expected_behavior if trap else turn.expected_behavior,
                    actual=turn.assistant_answer[:800],
                    detected_by_evaluator=False,
                    penalized_sufficiently=False,
                    root_cause="evaluator_judgment_gap",
                    recommended_action_type="improve_evaluator_rules",
                    failure_owner="evaluator_judgment",
                    trap_id=trap_id,
                )
            )
    summary = {
        "total_turns": len(trace.turns),
        "failed_turns": len({f.turn for f in failures}),
        "missed_injected_traps": missed_traps,
        "resisted_injected_traps": resisted_traps,
        "unexpected_failures": unexpected_failures,
        "false_positive_judgments": 0,
        "failure_count": len(failures),
    }
    return FailureBoard(run_id=trace.run_id, scenario_id=trace.scenario_id, failure_summary=summary, failures=failures)


def _short_description(error_type: str | None) -> str:
    return {
        "factual_inversion": "Assistant accepted or generated a wrong mechanistic direction.",
        "mechanistic_chain_break": "Assistant omitted a required mechanistic node.",
        "scope_drift": "Assistant drifted away from the requested scope.",
        "unsupported_plausible_mechanism": "Assistant made a plausible but unsupported mechanism claim.",
        "vague_supported_answer": "Assistant gave a vague low-information answer.",
        "overgeneralization": "Assistant overgeneralized beyond qualifiers.",
    }.get(error_type or "", "Evaluation failure detected.")


def _root_cause(error_type: str | None, penalized: bool) -> str:
    if not penalized:
        return "answer_generation_failure_plus_weak_reward_penalty"
    if error_type == "mechanistic_chain_break":
        return "mechanistic_completeness_gap"
    if error_type == "scope_drift":
        return "conversation_frame_not_preserved"
    return "answer_generation_failure"


def _action_type(error_type: str | None, penalized: bool) -> str:
    if not penalized:
        return "reward_weight_and_prompt_fix"
    if error_type == "mechanistic_chain_break":
        return "add_mechanism_graph_regression"
    if error_type == "scope_drift":
        return "conversation_frame_policy_fix"
    return "add_regression_test"


def _failure_owner(error_type: str | None, penalized: bool) -> str:
    if not penalized:
        return "reward_weighting"
    if error_type in {"unsupported_plausible_mechanism", "factual_inversion", "scope_drift", "mechanistic_chain_break"}:
        return "assistant_generation"
    return "evaluator_judgment"


def _trap_should_have_failed(answer: str, trap) -> bool:
    lower = answer.lower()
    forbidden = list(trap.failure_if_assistant_claims or [])
    if trap.wrong_claim:
        forbidden.append(trap.wrong_claim)
    if trap.wrong_direction:
        forbidden.append(trap.wrong_direction)
    forbidden = [item.lower() for item in forbidden if item]
    if not forbidden:
        return False
    if any(_contains_forbidden_claim(lower, phrase) for phrase in forbidden):
        return not _clearly_rejects_false_premise(lower)
    return False


def _contains_forbidden_claim(answer_lower: str, phrase_lower: str) -> bool:
    normalized_answer = " ".join(answer_lower.replace("-", " ").split())
    normalized_phrase = " ".join(phrase_lower.replace("-", " ").split())
    if normalized_phrase in normalized_answer:
        return True
    words = [word for word in normalized_phrase.split() if len(word) > 3]
    if len(words) < 5:
        return bool(words) and all(word in normalized_answer for word in words)
    hits = len([word for word in words if word in normalized_answer])
    return hits / max(1, len(words)) >= 0.8


def _clearly_rejects_false_premise(answer_lower: str) -> bool:
    rejection_markers = (
        "not agree",
        "would not agree",
        "do not agree",
        "no,",
        "incorrect",
        "contradicted",
        "contradicts",
        "not supported",
        "not accurate",
        "not acceptable",
        "not valid",
        "not be framed",
        "not framed",
        "should not",
        "not as direct proof",
        "transfer hypothesis",
        "background",
        "false premise",
        "reject",
        "rather than",
        "instead",
    )
    return any(marker in answer_lower for marker in rejection_markers)
