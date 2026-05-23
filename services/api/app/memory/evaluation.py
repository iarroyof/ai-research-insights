from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

from app.config import settings
from app.memory.lifecycle import estimate_tokens, select_working_set
from app.memory.rewards import reward_report


def _turn(doc: dict[str, Any]) -> int:
    try:
        return int(doc.get("turn_index", -1) or -1)
    except Exception:
        return -1


def _text(doc: dict[str, Any]) -> str:
    return str(doc.get("text") or doc.get("summary") or "")


def _legacy_recent_context(docs: list[dict[str, Any]], *, before_turn: int, token_budget: int, turns: int) -> list[dict[str, Any]]:
    recent = [
        dict(doc)
        for doc in docs
        if doc.get("doc_type") == "message" and 0 <= _turn(doc) < before_turn and doc.get("role") in {"user", "assistant"}
    ]
    recent = recent[-max(1, turns * 2) :]
    selected: list[dict[str, Any]] = []
    used = 0
    for doc in reversed(recent):
        cost = estimate_tokens(_text(doc))
        if used + cost > token_budget:
            continue
        selected.append(doc)
        used += cost
    return list(reversed(selected))


def _current_policy_context(docs: list[dict[str, Any]], *, before_turn: int, token_budget: int, turns: int) -> list[dict[str, Any]]:
    candidates = [
        dict(doc)
        for doc in docs
        if 0 <= _turn(doc) < before_turn
        and doc.get("doc_type") in {"message", "episodic_summary", "landmark", "user_correction"}
    ]
    return select_working_set(
        candidates,
        token_budget=token_budget,
        current_turn_index=before_turn,
        working_turns=turns,
        eviction_importance_threshold=settings.memory.eviction_importance_threshold,
        query_text=_text(next((doc for doc in docs if _turn(doc) == before_turn and doc.get("role") == "user"), {})),
    ).selected


def _paired_turns(docs: list[dict[str, Any]]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    messages = sorted(
        [dict(doc) for doc in docs if doc.get("doc_type") == "message" and doc.get("role") in {"user", "assistant"}],
        key=lambda item: (_turn(item), str(item.get("role") or "")),
    )
    by_turn = {_turn(doc): doc for doc in messages}
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for user in messages:
        if user.get("role") != "user":
            continue
        assistant = by_turn.get(_turn(user) + 1)
        if assistant and assistant.get("role") == "assistant":
            pairs.append((user, assistant))
    return pairs


def simulate_session_quality(
    docs: list[dict[str, Any]],
    *,
    token_budget: int | None = None,
    working_turns: int | None = None,
) -> dict[str, Any]:
    token_budget = token_budget or settings.memory.working_buffer_token_budget
    working_turns = working_turns or settings.memory.working_buffer_turns
    pairs = _paired_turns(docs)
    rows: list[dict[str, Any]] = []
    for user, assistant in pairs:
        before_turn = _turn(user)
        question = _text(user)
        answer = _text(assistant)
        baseline_context = _legacy_recent_context(docs, before_turn=before_turn, token_budget=token_budget, turns=working_turns)
        current_context = _current_policy_context(docs, before_turn=before_turn, token_budget=token_budget, turns=working_turns)
        baseline_reward = reward_report(
            question=question,
            answer=answer,
            selected_context=baseline_context,
            conflicts=[],
            claim_support=[],
            elapsed_sec=0.0,
            token_budget=token_budget,
        )
        current_reward = reward_report(
            question=question,
            answer=answer,
            selected_context=current_context,
            conflicts=[],
            claim_support=[],
            elapsed_sec=0.0,
            token_budget=token_budget,
        )
        rows.append(
            {
                "turn_index": before_turn,
                "baseline_reward": baseline_reward,
                "current_reward": current_reward,
                "baseline_context_count": len(baseline_context),
                "current_context_count": len(current_context),
                "baseline_context_tokens": sum(estimate_tokens(_text(item)) for item in baseline_context),
                "current_context_tokens": sum(int(item.get("token_count", 0) or estimate_tokens(_text(item))) for item in current_context),
                "current_compressed_count": sum(1 for item in current_context if item.get("memory_compressed")),
                "current_promoted_count": sum(1 for item in current_context if item.get("memory_state") == "promoted"),
            }
        )
    return _summarize_rows(rows)


def _avg(rows: list[dict[str, Any]], path: tuple[str, ...]) -> float:
    values = []
    for row in rows:
        value: Any = row
        for key in path:
            value = value.get(key, {}) if isinstance(value, dict) else {}
        if isinstance(value, (int, float)):
            values.append(float(value))
    return round(sum(values) / max(1, len(values)), 4)


def _summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    baseline_score = _avg(rows, ("baseline_reward", "score"))
    current_score = _avg(rows, ("current_reward", "score"))
    baseline_support = _avg(rows, ("baseline_reward", "context_support"))
    current_support = _avg(rows, ("current_reward", "context_support"))
    baseline_alignment = _avg(rows, ("baseline_reward", "domain_alignment"))
    current_alignment = _avg(rows, ("current_reward", "domain_alignment"))
    baseline_off_topic = _avg(rows, ("baseline_reward", "off_topic_penalty"))
    current_off_topic = _avg(rows, ("current_reward", "off_topic_penalty"))
    return {
        "turn_pairs": len(rows),
        "baseline_avg_reward": baseline_score,
        "current_avg_reward": current_score,
        "reward_delta": round(current_score - baseline_score, 4),
        "baseline_context_support": baseline_support,
        "current_context_support": current_support,
        "context_support_delta": round(current_support - baseline_support, 4),
        "baseline_domain_alignment": baseline_alignment,
        "current_domain_alignment": current_alignment,
        "domain_alignment_delta": round(current_alignment - baseline_alignment, 4),
        "baseline_off_topic_penalty": baseline_off_topic,
        "current_off_topic_penalty": current_off_topic,
        "off_topic_penalty_delta": round(current_off_topic - baseline_off_topic, 4),
        "baseline_avg_context_tokens": _avg(rows, ("baseline_context_tokens",)),
        "current_avg_context_tokens": _avg(rows, ("current_context_tokens",)),
        "current_avg_compressed_count": _avg(rows, ("current_compressed_count",)),
        "current_avg_promoted_count": _avg(rows, ("current_promoted_count",)),
        "rows": rows,
    }


def simulate_quality_by_session(docs: Iterable[dict[str, Any]], *, token_budget: int | None = None) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for doc in docs:
        session_id = str(doc.get("session_id") or "")
        if session_id:
            grouped[session_id].append(dict(doc))
    sessions = {
        session_id: simulate_session_quality(items, token_budget=token_budget)
        for session_id, items in grouped.items()
    }
    useful = [item for item in sessions.values() if item["turn_pairs"]]
    aggregate = _summarize_rows([row for item in useful for row in item.get("rows", [])])
    aggregate.pop("rows", None)
    return {
        "session_count": len(grouped),
        "evaluated_session_count": len(useful),
        "aggregate": aggregate,
        "sessions": {
            session_id: {k: v for k, v in report.items() if k != "rows"}
            for session_id, report in sessions.items()
            if report["turn_pairs"]
        },
    }
