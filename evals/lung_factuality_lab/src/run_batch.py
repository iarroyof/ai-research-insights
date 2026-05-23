from __future__ import annotations

import argparse
from pathlib import Path

from evals.lung_factuality_lab.src.diagnosis_failure_board import build_failure_board
from evals.lung_factuality_lab.src.recommendation_engine import build_recommendations, recommendations_markdown
from evals.lung_factuality_lab.src.report_writer import simulation_report_markdown
from evals.lung_factuality_lab.src.run_single import run_single
from evals.lung_factuality_lab.src.scenario_loader import load_batch_config, load_scenarios
from evals.lung_factuality_lab.src.schemas import ConversationTrace, FailureBoard
from evals.lung_factuality_lab.src.trace_writer import write_json


def run_batch(
    *,
    config_path: str,
    assistant_name: str,
    out_dir: str,
    dummy_mode: str = "mixed",
    endpoint: str = "",
    api_key: str = "",
    tenant_id: str = "eval-lab",
    request_timeout: float = 300.0,
    variant_index: int | None = None,
    wrong_answer_bank: str = "",
    answer_replay_dir: str = "",
) -> None:
    config = load_batch_config(config_path)
    scenarios = _resolve_scenarios(config)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    traces: list[ConversationTrace] = []
    all_failures = []
    failed_turn_keys = set()
    for scenario_id in scenarios:
        scenario_out = out / str(scenario_id)
        trace = run_single(
            scenario_id=str(scenario_id),
            assistant_name=assistant_name,
            out_dir=scenario_out,
            dummy_mode=dummy_mode,
            endpoint=endpoint,
            api_key=api_key,
            tenant_id=tenant_id,
            request_timeout=request_timeout,
            variant_index=variant_index,
            wrong_answer_bank=wrong_answer_bank,
            answer_replay_dir=answer_replay_dir,
        )
        traces.append(trace)
        board = FailureBoard(**__import__("json").loads((scenario_out / "failure_board.json").read_text(encoding="utf-8")))
        for failure in board.failures:
            failed_turn_keys.add((str(scenario_id), failure.turn))
            data = failure.model_dump(mode="json")
            data["failure_id"] = f"fail_{len(all_failures)+1:03d}"
            all_failures.append(type(failure)(**data))
    batch_trace = {
        "batch_id": config.get("batch_id", "batch"),
        "assistant": assistant_name,
        "scenario_count": len(traces),
        "avg_reward": round(sum(t.aggregate.get("avg_reward", 0.0) for t in traces) / max(1, len(traces)), 4),
        "scenarios": [
            {"scenario_id": t.scenario_id, "run_id": t.run_id, "avg_reward": t.aggregate.get("avg_reward"), "failure_turn_count": t.aggregate.get("failure_turn_count")}
            for t in traces
        ],
    }
    write_json(out / "conversation_trace.json", batch_trace)
    board = FailureBoard(
        run_id=str(config.get("batch_id", "batch")),
        scenario_id=None,
        failure_summary={
            "total_turns": sum(len(t.turns) for t in traces),
            "failed_turns": len(failed_turn_keys),
            "missed_injected_traps": sum(1 for f in all_failures if not f.detected_by_evaluator),
            "unexpected_failures": sum(1 for f in all_failures if not f.trap_id),
            "false_positive_judgments": 0,
            "failure_count": len(all_failures),
        },
        failures=all_failures,
    )
    recs = build_recommendations(board)
    write_json(out / "failure_board.json", board)
    write_json(out / "recommendations.json", {"recommendations": [r.model_dump(mode="json") for r in recs]})
    (out / "recommendations.md").write_text(recommendations_markdown(recs), encoding="utf-8")
    (out / "simulation_report.md").write_text(_batch_report(batch_trace, board), encoding="utf-8")


def _batch_report(batch_trace: dict, board: FailureBoard) -> str:
    lines = [
        f"# Batch Simulation Report: {batch_trace['batch_id']}",
        "",
        f"- Assistant: {batch_trace['assistant']}",
        f"- Scenario count: {batch_trace['scenario_count']}",
        f"- Average reward: {batch_trace['avg_reward']}",
        f"- Failure count: {board.failure_summary.get('failure_count', 0)}",
        "",
        "## Scenarios",
        "",
        "| Scenario | Reward | Failure turns |",
        "|---|---:|---:|",
    ]
    for item in batch_trace["scenarios"]:
        lines.append(f"| {item['scenario_id']} | {item['avg_reward']} | {item['failure_turn_count']} |")
    lines.extend(["", "Inspect per-scenario subdirectories for full traces, recommendations, and regression plans.", ""])
    return "\n".join(lines)


def _resolve_scenarios(config: dict) -> list[str]:
    scenarios = [str(item) for item in config.get("scenarios") or []]
    if scenarios:
        return scenarios
    scenario_filter = str(config.get("scenario_filter") or "").lower()
    all_scenarios = load_scenarios()
    if scenario_filter in {"generated", "generated_large"}:
        scenarios = sorted(sid for sid in all_scenarios if "__gen_" in sid)
    elif scenario_filter == "seed":
        scenarios = sorted(sid for sid in all_scenarios if "__gen_" not in sid)
    variant_indices = config.get("generated_variant_indices")
    if variant_indices is not None:
        allowed = {int(item) for item in variant_indices}
        scenarios = [
            sid
            for sid in scenarios
            if "__gen_" not in sid or _generated_variant_index(sid) in allowed
        ]
    base_families = config.get("generated_base_scenario_ids")
    if base_families:
        allowed_bases = {str(item) for item in base_families}
        scenarios = [
            sid
            for sid in scenarios
            if (all_scenarios[sid].base_scenario_id or sid) in allowed_bases
        ]
    limit = config.get("limit")
    if limit is not None:
        scenarios = scenarios[: int(limit)]
    return scenarios


def _generated_variant_index(scenario_id: str) -> int | None:
    if "__gen_" not in scenario_id:
        return None
    try:
        return int(scenario_id.rsplit("__gen_", 1)[1])
    except ValueError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a lung factuality lab batch.")
    parser.add_argument("--config", required=True)
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
    run_batch(
        config_path=args.config,
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
    print(f"Wrote batch outputs to {args.out}")


if __name__ == "__main__":
    main()
