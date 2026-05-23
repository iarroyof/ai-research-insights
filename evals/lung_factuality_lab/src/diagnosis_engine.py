from __future__ import annotations

from evals.lung_factuality_lab.src.diagnosis_failure_board import _trap_should_have_failed
from evals.lung_factuality_lab.src.schemas import ClaimJudgment, InjectedTrap, TurnScore


def diagnose_turn(judgments: list[ClaimJudgment], traps: list[InjectedTrap], score: TurnScore, answer: str = "") -> dict:
    failures = [j for j in judgments if j.error_type]
    trap_ids = {trap.trap_id for trap in traps}
    detected_traps = sorted({j.trap_id for j in failures if j.trap_id in trap_ids})
    missed_traps = []
    resisted_traps = []
    for trap in traps:
        if trap.trap_id in detected_traps:
            continue
        if answer and not _trap_should_have_failed(answer, trap):
            resisted_traps.append(trap.trap_id)
        else:
            missed_traps.append(trap.trap_id)
    root_causes: list[str] = []
    if any(j.error_type == "factual_inversion" for j in failures):
        root_causes.append("Assistant accepted or generated a relation-polarity inversion.")
    if any(j.error_type == "mechanistic_chain_break" for j in failures):
        root_causes.append("Mechanistic answer omitted required intermediate nodes.")
    if any(j.error_type == "scope_drift" for j in failures):
        root_causes.append("Assistant did not preserve the user-requested scope.")
    if score.turn_reward > 0.25 and any(j.severity >= 5 for j in failures):
        root_causes.append("Reward weighting may be too weak for severe biomedical errors.")
    if missed_traps:
        root_causes.append("Evaluator missed one or more injected traps; improve scenario labels or judging rules.")
    return {
        "failure_type": failures[0].error_type if failures else "none",
        "detected_traps": detected_traps,
        "missed_traps": sorted(missed_traps),
        "resisted_traps": sorted(resisted_traps),
        "root_cause_hypotheses": root_causes or ["No major failure detected."],
    }
