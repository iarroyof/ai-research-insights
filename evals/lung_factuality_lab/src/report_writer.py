from __future__ import annotations

from evals.lung_factuality_lab.src.schemas import ConversationTrace, FailureBoard, Recommendation


def simulation_report_markdown(trace: ConversationTrace, board: FailureBoard, recs: list[Recommendation]) -> str:
    total_traps = sum(len(turn.injected_traps) for turn in trace.turns)
    missed = board.failure_summary.get("missed_injected_traps", 0)
    highest = max((f.severity for f in board.failures), default=0)
    lines = [
        f"# Simulation Report: {trace.scenario_id}",
        "",
        "## Purpose",
        "",
        "This simulation tested lung-cancer factuality, mechanistic consistency, correction adherence, scope control, and reward diagnostics.",
        "",
        "## Summary",
        "",
        f"- Total turns: {len(trace.turns)}",
        f"- Injected traps: {total_traps}",
        f"- Traps caught by evaluator: {max(0, total_traps - missed)}",
        f"- Traps missed by evaluator/chatbot trace: {missed}",
        f"- Failed turns: {board.failure_summary.get('failed_turns', 0)}",
        f"- Highest severity failure: {highest}",
        "",
        "## Turn-Level Timeline",
        "",
        "| Turn | Trap | Assistant behavior | Judgment | Reward | Diagnosis |",
        "|---:|---|---|---|---:|---|",
    ]
    for turn in trace.turns:
        trap = ", ".join(t.type for t in turn.injected_traps) or "-"
        behavior = _clip(turn.assistant_answer, 80)
        judgment = ", ".join(j.label for j in turn.claim_judgments[:3]) or "none"
        diagnosis = _clip("; ".join(turn.diagnosis.get("root_cause_hypotheses", [])), 100)
        lines.append(f"| {turn.turn} | {trap} | {behavior} | {judgment} | {turn.scores.turn_reward:.2f} | {diagnosis} |")
    lines.extend(["", "## Most Important Failure", ""])
    if board.failures:
        top = sorted(board.failures, key=lambda item: (-item.severity, item.turn))[0]
        lines.append(f"{top.short_description} Owner: `{top.failure_owner}`. Root cause: `{top.root_cause}`.")
    else:
        lines.append("No major failure detected.")
    lines.extend(["", "## What To Fix Next", ""])
    if recs:
        for idx, rec in enumerate(recs[:6], 1):
            lines.append(f"{idx}. `{rec.priority}` `{rec.target}`: {rec.problem}")
    else:
        lines.append("No fixes recommended for this run.")
    lines.extend(
        [
            "",
            "## Recommended Rerun",
            "",
            "```bash",
            "python -m evals.lung_factuality_lab.src.run_single \\",
            f"  --scenario {trace.scenario_id} \\",
            "  --assistant dummy \\",
            f"  --out evals/lung_factuality_lab/runs/rerun_{trace.scenario_id}",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def _clip(text: str, limit: int) -> str:
    text = " ".join((text or "").split()).replace("|", "/")
    return text if len(text) <= limit else text[: limit - 3] + "..."

