from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from evals.lung_factuality_lab.src.assistant_adapters import build_adapter
from evals.lung_factuality_lab.src.claim_extractor import extract_claims
from evals.lung_factuality_lab.src.claim_judge import judge_claims
from evals.lung_factuality_lab.src.conversation_generator import generate_conversation
from evals.lung_factuality_lab.src.diagnosis_engine import diagnose_turn
from evals.lung_factuality_lab.src.diagnosis_failure_board import build_failure_board
from evals.lung_factuality_lab.src.recommendation_engine import build_recommendations, recommendations_markdown
from evals.lung_factuality_lab.src.regression_planner import regression_plan_dict
from evals.lung_factuality_lab.src.report_writer import simulation_report_markdown
from evals.lung_factuality_lab.src.reward_scorer import score_turn
from evals.lung_factuality_lab.src.scenario_loader import (
    load_gold_claims,
    load_mechanism_graphs,
    load_reward_config,
    load_scenario,
    load_traps,
)
from evals.lung_factuality_lab.src.schemas import ClaimJudgment, ConversationTrace, InjectedTrap, TurnTrace
from evals.lung_factuality_lab.src.trace_writer import write_json, write_jsonl, write_yaml_like


def run_single(
    *,
    scenario_id: str,
    assistant_name: str,
    out_dir: str | Path,
    dummy_mode: str = "mixed",
    endpoint: str = "",
    api_key: str = "",
    tenant_id: str = "eval-lab",
    request_timeout: float = 300.0,
    variant_index: int | None = None,
    wrong_answer_bank: str = "",
    answer_replay_dir: str = "",
) -> ConversationTrace:
    scenario = load_scenario(scenario_id)
    gold_claims = load_gold_claims()
    mechanism_graphs = load_mechanism_graphs()
    reward_config = load_reward_config()
    target_mechanism_graphs = scenario.target_mechanism_graphs or [
        graph_id
        for graph_id, graph in mechanism_graphs.items()
        if graph.target_gold_claim and graph.target_gold_claim in scenario.target_gold_claims
    ]
    adapter = build_adapter(
        assistant_name,
        mode=dummy_mode,
        endpoint=endpoint,
        api_key=api_key,
        tenant_id=tenant_id,
        request_timeout=request_timeout,
        wrong_answer_bank=wrong_answer_bank,
        answer_replay_dir=answer_replay_dir,
    )
    run_id = "run_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    turns = generate_conversation(scenario, variant_index=variant_index)
    answers = []
    extracted = []
    judgments_all = []
    scores = []
    traces = []
    history: list[dict] = []
    trap_by_id = {trap.trap_id: trap for trap in scenario.injected_traps}
    trap_by_id.update(load_traps())
    correction_active = False

    for turn in turns:
        answer = adapter.answer(scenario, turn, history)
        claims = extract_claims(answer.answer, turn=turn.turn)
        traps = [trap_by_id.get(tid) or _synthesize_trap(tid, turn, scenario) for tid in turn.trap_ids]
        obeyed_correction = not correction_active or _obeys_correction(answer.answer, turn.correction_terms)
        judgments = judge_claims(
            claims,
            gold_claims=gold_claims,
            mechanism_graphs=mechanism_graphs,
            target_gold_claims=scenario.target_gold_claims,
            target_mechanism_graphs=target_mechanism_graphs,
            traps=traps,
            expected_focus_terms=turn.expected_focus_terms,
            turn_tags=turn.tags,
        )
        if not answer.answer.strip():
            judgments.append(
                ClaimJudgment(
                    claim_id=f"empty_answer_{turn.turn}",
                    label="unsupported",
                    reason="Assistant returned an empty answer for this turn.",
                    confidence=1.0,
                    error_type="empty_answer",
                    severity=4,
                    trap_id=traps[0].trap_id if traps else None,
                )
            )
        search_telemetry = (
            ((answer.adapter_meta.get("citations") or {}).get("auto_context") or {}).get("evidence_assembly")
            if isinstance(answer.adapter_meta, dict)
            else {}
        ) or {}
        score = score_turn(
            turn=turn.turn,
            judgments=judgments,
            traps=traps,
            reward_config=reward_config,
            obeyed_correction=obeyed_correction,
            search_telemetry=search_telemetry,
        )
        diagnosis = diagnose_turn(judgments, traps, score, answer=answer.answer)
        traces.append(
            TurnTrace(
                turn=turn.turn,
                user_message=turn.message,
                assistant_answer=answer.answer,
                assistant_metadata=answer.adapter_meta,
                expected_behavior=turn.expected_behavior,
                injected_traps=traps,
                extracted_claims=claims,
                claim_judgments=judgments,
                scores=score,
                diagnosis=diagnosis,
            )
        )
        answers.append(answer)
        extracted.extend(claims)
        judgments_all.extend(judgments)
        scores.append(score)
        history.append({"role": "user", "content": turn.message})
        history.append({"role": "assistant", "content": answer.answer})
        if turn.correction_terms or any("correction" in tid for tid in turn.trap_ids):
            correction_active = True

    trace = ConversationTrace(
        run_id=run_id,
        scenario_id=scenario.scenario_id,
        assistant=assistant_name,
        turns=traces,
        aggregate={
            "avg_reward": round(sum(s.turn_reward for s in scores) / max(1, len(scores)), 4),
            "turn_count": len(scores),
            "failure_turn_count": sum(1 for t in traces if any(j.error_type for j in t.claim_judgments)),
        },
    )
    board = build_failure_board(trace)
    recs = build_recommendations(board)

    write_yaml_like(out / "scenario.yaml", scenario)
    write_jsonl(out / "generated_conversation.jsonl", turns)
    write_jsonl(out / "assistant_answers.jsonl", answers)
    write_jsonl(out / "extracted_claims.jsonl", extracted)
    write_jsonl(out / "claim_judgments.jsonl", judgments_all)
    write_jsonl(out / "turn_scores.jsonl", scores)
    write_json(out / "conversation_trace.json", trace)
    write_json(out / "failure_board.json", board)
    write_json(out / "recommendations.json", {"run_id": run_id, "recommendations": [r.model_dump(mode="json") for r in recs]})
    (out / "recommendations.md").write_text(recommendations_markdown(recs), encoding="utf-8")
    write_yaml_like(out / "regression_plan.yaml", regression_plan_dict(board))
    (out / "simulation_report.md").write_text(simulation_report_markdown(trace, board, recs), encoding="utf-8")
    return trace


def _obeys_correction(answer: str, correction_terms: list[str]) -> bool:
    if not correction_terms:
        return True
    lower = answer.lower()
    return all(term.lower() in lower for term in correction_terms)


def _synthesize_trap(trap_id: str, turn, scenario) -> InjectedTrap:
    return InjectedTrap(
        trap_id=trap_id,
        turn=turn.turn,
        type=_infer_trap_type(trap_id, turn.tags),
        severity=_infer_trap_severity(trap_id, turn.tags),
        wrong_claim=turn.must_not_claim[0] if turn.must_not_claim else "",
        expected_behavior=turn.expected_behavior or "Detect and handle the generated conversation trap.",
        target_gold_claims=turn.target_gold_claims or scenario.target_gold_claims,
        failure_if_assistant_claims=turn.must_not_claim,
    )


def _infer_trap_type(trap_id: str, tags: list[str]) -> str:
    text = " ".join([trap_id, *tags]).lower()
    if "scope" in text or "citation_drift" in text or "cross_cancer" in text:
        return "scope_drift"
    if "oversimplification" in text or "mechanistic_completeness" in text:
        return "mechanistic_chain_break"
    if "false_premise" in text or "bad_answer" in text or "polarity" in text:
        return "factual_inversion"
    if "unsupported" in text:
        return "unsupported_plausible_mechanism"
    return "generated_trap"


def _infer_trap_severity(trap_id: str, tags: list[str]) -> int:
    text = " ".join([trap_id, *tags]).lower()
    if "false_premise" in text or "bad_answer" in text:
        return 5
    if "scope" in text or "cross_cancer" in text or "citation" in text:
        return 4
    if "oversimplification" in text or "mechanistic" in text:
        return 3
    return 3


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one lung factuality lab scenario.")
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--assistant", default="dummy")
    parser.add_argument("--out", required=True)
    parser.add_argument("--dummy-mode", default="mixed")
    parser.add_argument("--endpoint", default="")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--tenant-id", default="eval-lab")
    parser.add_argument("--request-timeout", type=float, default=300.0)
    parser.add_argument("--variant-index", type=int, default=None)
    parser.add_argument("--wrong-answer-bank", default="")
    parser.add_argument("--answer-replay-dir", default="")
    args = parser.parse_args()
    trace = run_single(
        scenario_id=args.scenario,
        assistant_name=args.assistant,
        out_dir=args.out,
        dummy_mode=args.dummy_mode,
        endpoint=args.endpoint,
        api_key=args.api_key,
        tenant_id=args.tenant_id,
        request_timeout=args.request_timeout,
        variant_index=args.variant_index,
        wrong_answer_bank=args.wrong_answer_bank,
        answer_replay_dir=args.answer_replay_dir,
    )
    print(f"Wrote {trace.run_id} for {trace.scenario_id} to {args.out}")


if __name__ == "__main__":
    main()
