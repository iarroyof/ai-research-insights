from __future__ import annotations

from collections import defaultdict

from evals.lung_factuality_lab.src.schemas import FailureBoard, Recommendation


def build_recommendations(board: FailureBoard) -> list[Recommendation]:
    grouped = defaultdict(list)
    for failure in board.failures:
        grouped[failure.category].append(failure)
    recs: list[Recommendation] = []
    for category, failures in grouped.items():
        max_severity = max(f.severity for f in failures)
        priority = "P0" if max_severity >= 5 else "P1" if max_severity >= 3 else "P2"
        ids = [f.failure_id for f in failures]
        scenarios = _scenarios(board, failures)
        if category == "factual_inversion":
            recs.append(
                Recommendation(
                    priority=priority,
                    target="reward_config",
                    problem="Factual inversions are detected or expected but need strong penalties.",
                    evidence={"failure_ids": ids, "category": category},
                    recommended_change={
                        "file": "evals/lung_factuality_lab/configs/reward_weights.yaml",
                        "parameter": "penalties.factual_inversion",
                        "suggested_value": 0.55,
                    },
                    validation={"rerun_scenarios": scenarios, "expected_result": "wrong inversion reward stays below severity ceiling"},
                )
            )
            recs.append(
                Recommendation(
                    priority="P0",
                    target="assistant_prompt_policy",
                    problem="Assistant may over-agree with user-provided biomedical premises.",
                    evidence={"failure_ids": ids, "pattern": "user_false_premise_accepted"},
                    recommended_change={
                        "file": "chat/context policy prompt",
                        "add_rule": "Verify biomedical relation polarity against evidence before agreeing with user-proposed mechanisms.",
                    },
                    validation={"rerun_scenarios": scenarios, "expected_result": "assistant rejects the false premise and explains the correct direction"},
                )
            )
        elif category == "mechanistic_chain_break":
            recs.append(
                Recommendation(
                    priority=priority,
                    target="evidence_layer",
                    problem="Mechanistic chain scoring needs required-node awareness.",
                    evidence={"failure_ids": ids, "missed_nodes": _missing_hint(failures)},
                    recommended_change={
                        "file": "evals/lung_factuality_lab/data/evidence/mechanism_graphs.yaml",
                        "add_or_update_graph": "Add required nodes and preferred edges for the failed mechanism.",
                    },
                    validation={"rerun_scenarios": scenarios, "expected_result": "missing required nodes are classified as mechanistic_chain_break"},
                )
            )
        elif category == "scope_drift":
            recs.append(
                Recommendation(
                    priority=priority,
                    target="conversation_frame_policy",
                    problem="Assistant did not preserve the requested active scope.",
                    evidence={"failure_ids": ids},
                    recommended_change={
                        "file": "services/api/app/memory/consistency.py",
                        "change": "Strengthen semantic drift detection and correction adherence scoring.",
                    },
                    validation={"rerun_scenarios": scenarios, "expected_result": "scope-drift turns are warned and penalized"},
                )
            )
        elif category == "unsupported_plausible_mechanism":
            recs.append(
                Recommendation(
                    priority=priority,
                    target="evidence_grounding",
                    problem="Scientific-sounding unsupported mechanisms are being produced or insufficiently penalized.",
                    evidence={"failure_ids": ids},
                    recommended_change={
                        "file": "services/api/app/memory/claim_support.py",
                        "change": "Require source-sentence support for high-confidence mechanistic claims before promotion.",
                    },
                    validation={"rerun_scenarios": scenarios, "expected_result": "unsupported mechanisms are unsupported and low reward"},
                )
            )
        else:
            recs.append(
                Recommendation(
                    priority=priority,
                    target="regression_tests",
                    problem=f"Observed {category} failures.",
                    evidence={"failure_ids": ids},
                    recommended_change={"file": "evals/lung_factuality_lab/data/scenarios", "change": "Add focused regression scenario."},
                    validation={"rerun_scenarios": scenarios, "expected_result": "failure does not recur"},
                )
            )
    return recs


def _missing_hint(failures) -> list[str]:
    text = " ".join(f.short_description + " " + f.expected for f in failures).lower()
    out = []
    for node in ("MET/c-MET", "HGF", "CAF", "EMT", "CD8", "hypoxia"):
        if node.lower().split("/")[0] in text:
            out.append(node)
    return out


def _scenarios(board: FailureBoard, failures) -> list[str]:
    scenario_ids = sorted({f.scenario_id for f in failures if f.scenario_id})
    if scenario_ids:
        return scenario_ids
    return [board.scenario_id] if board.scenario_id else []


def recommendations_markdown(recs: list[Recommendation]) -> str:
    lines = ["# Recommendations", ""]
    if not recs:
        lines.append("No actionable failures detected.")
        return "\n".join(lines) + "\n"
    for idx, rec in enumerate(recs, 1):
        lines.extend(
            [
                f"## {idx}. {rec.priority} - {rec.target}",
                "",
                f"Problem: {rec.problem}",
                "",
                "Evidence:",
                "```json",
                rec.model_dump_json(indent=2),
                "```",
                "",
            ]
        )
    return "\n".join(lines)
