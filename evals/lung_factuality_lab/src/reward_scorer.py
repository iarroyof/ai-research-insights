from __future__ import annotations

from typing import Any

from evals.lung_factuality_lab.src.schemas import ClaimJudgment, InjectedTrap, TurnScore


DEFAULT_WEIGHTS = {
    "factual_support": 0.30,
    "contradiction_avoidance": 0.20,
    "mechanistic_completeness": 0.15,
    "scope_alignment": 0.10,
    "correction_adherence": 0.10,
    "uncertainty_calibration": 0.10,
    "citation_or_evidence_quality": 0.05,
}


def score_turn(
    *,
    turn: int,
    judgments: list[ClaimJudgment],
    traps: list[InjectedTrap],
    reward_config: dict[str, Any],
    obeyed_correction: bool = True,
    search_telemetry: dict[str, Any] | None = None,
) -> TurnScore:
    weights = dict(DEFAULT_WEIGHTS)
    weights.update((reward_config.get("reward_components") or {}))
    weights = {k: float(v.get("weight", v) if isinstance(v, dict) else v) for k, v in weights.items()}
    penalties = dict(reward_config.get("penalties") or {})

    labels = [j.label for j in judgments]
    severe = [j for j in judgments if j.severity >= 4]
    supported = sum(1 for label in labels if label in {"supported", "partially_supported"})
    contradicted = sum(1 for label in labels if label == "contradicted")
    unsupported = sum(1 for label in labels if label == "unsupported")
    vague = sum(1 for label in labels if label == "too_vague")
    out_scope = sum(1 for label in labels if label == "out_of_scope")
    chain_breaks = [j for j in judgments if j.error_type == "mechanistic_chain_break"]
    total = max(1, len(judgments))

    component_scores = {
        "factual_support": supported / total,
        "contradiction_avoidance": 1.0 - min(1.0, contradicted / total),
        "mechanistic_completeness": 0.0 if chain_breaks else 1.0,
        "scope_alignment": 1.0 - min(1.0, out_scope / total),
        "correction_adherence": 1.0 if obeyed_correction else 0.0,
        "uncertainty_calibration": 0.4 if unsupported else 0.8,
        "citation_or_evidence_quality": supported / total,
    }
    telemetry = search_telemetry or {}
    edge_support = str((telemetry.get("evidence_puzzle") or {}).get("edge_support_status") or "")
    assembly_quality = float(telemetry.get("assembly_quality", 0.0) or 0.0)
    if telemetry:
        component_scores["evidence_assembly_quality"] = max(0.0, min(1.0, assembly_quality))
        component_scores["evidence_bridge_safety"] = 0.0 if unsupported and edge_support == "missing" else 0.5 if unsupported and edge_support == "partial" else 1.0
        weights.setdefault("evidence_assembly_quality", 0.05)
        weights.setdefault("evidence_bridge_safety", 0.08)

    base = sum(weights.get(key, 0.0) * value for key, value in component_scores.items())
    raw_penalties: list[dict[str, Any]] = []
    for judgment in judgments:
        if not judgment.error_type:
            continue
        amount = float(penalties.get(judgment.error_type, _fallback_penalty(judgment)))
        if judgment.label == "too_vague":
            amount = float(penalties.get("vague_supported_answer", amount))
        raw_penalties.append(
            {
                "type": judgment.error_type,
                "amount": round(amount, 4),
                "reason": judgment.reason,
                "claim_id": judgment.claim_id,
            }
        )
    penalties_applied = _cap_repeated_penalties(raw_penalties)
    reward = max(0.0, min(1.0, base - sum(item["amount"] for item in penalties_applied)))
    interpretation = _interpret(judgments, severe, unsupported, vague, out_scope, chain_breaks)
    return TurnScore(
        turn=turn,
        turn_reward=round(reward, 4),
        component_scores={k: round(v, 4) for k, v in component_scores.items()},
        penalties_applied=penalties_applied,
        interpretation=interpretation,
    )


def _fallback_penalty(judgment: ClaimJudgment) -> float:
    if judgment.severity >= 5:
        return 0.55
    if judgment.severity == 4:
        return 0.45
    if judgment.severity == 3:
        return 0.25
    if judgment.severity == 2:
        return 0.10
    return 0.0


def _interpret(
    judgments: list[ClaimJudgment],
    severe: list[ClaimJudgment],
    unsupported: int,
    vague: int,
    out_scope: int,
    chain_breaks: list[ClaimJudgment],
) -> str:
    if severe:
        return "Severe factual or mechanistic contradiction detected."
    if chain_breaks:
        return "The answer is broadly relevant but mechanistically incomplete."
    if out_scope:
        return "The answer drifted outside the requested conversation scope."
    if unsupported:
        return "The answer includes plausible but unsupported claims."
    if vague:
        return "The answer is directionally acceptable but too vague for the scenario."
    return "The answer is supported and aligned with the scenario."



def _cap_repeated_penalties(raw_penalties: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_type: dict[str, dict[str, Any]] = {}
    for item in raw_penalties:
        key = str(item["type"])
        existing = by_type.get(key)
        if existing is None or float(item["amount"]) > float(existing["amount"]):
            merged = dict(item)
            merged["count"] = 1
            by_type[key] = merged
        else:
            existing["count"] = int(existing.get("count", 1)) + 1
    for item in by_type.values():
        if int(item.get("count", 1)) > 1:
            item["reason"] = f"{item['reason']} Repeated {item['count']} times in this turn; penalty capped by type."
    return list(by_type.values())
