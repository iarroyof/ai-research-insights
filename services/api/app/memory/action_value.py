from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class ActionValue:
    action_id: str
    state_key: str
    action_key: str
    visits: int
    q_value: float
    reward_avg: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _stable_id(*parts: str) -> str:
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()[:24]


def state_key(terms: Iterable[str], *, max_terms: int = 8) -> str:
    cleaned = sorted({str(t).lower() for t in terms if str(t).strip()})
    return "|".join(cleaned[:max_terms]) or "empty"


def action_key(action: dict[str, Any]) -> str:
    buckets = {
        "context": int(action.get("selected_context_count", 0) or 0),
        "triplets": int(action.get("retrieved_triplet_count", 0) or 0),
        "ideas": int(action.get("selected_idea_count", 0) or 0),
        "web": int(action.get("web_result_count", 0) or 0),
        "evidence": int(action.get("evidence_candidate_count", 0) or 0),
    }
    return "|".join(f"{k}:{min(v, 9)}" for k, v in buckets.items())


def action_id(tenant: str, scope: str, state: str, action: str) -> str:
    return "av_" + _stable_id(tenant, scope, state, action)


def update_action_value(
    existing: dict[str, Any] | None,
    *,
    tenant: str,
    scope: str,
    state: str,
    action: str,
    reward: float,
    alpha: float = 0.25,
) -> dict[str, Any]:
    existing = dict(existing or {})
    visits = int(existing.get("visits", 0) or 0) + 1
    old_q = float(existing.get("q_value", 0.0) or 0.0)
    q_value = old_q + max(0.0, min(1.0, alpha)) * (float(reward) - old_q)
    old_reward_sum = float(existing.get("reward_sum", 0.0) or 0.0)
    reward_sum = old_reward_sum + float(reward)
    return {
        **existing,
        "doc_type": "action_value",
        "tenant": tenant,
        "scope": scope,
        "action_id": existing.get("action_id") or action_id(tenant, scope, state, action),
        "state_key": state,
        "action_key": action,
        "visits": visits,
        "q_value": round(q_value, 6),
        "reward_sum": round(reward_sum, 6),
        "reward_avg": round(reward_sum / max(1, visits), 6),
    }


def best_action_value(items: Iterable[dict[str, Any]]) -> dict[str, Any] | None:
    values = [dict(item) for item in items]
    if not values:
        return None
    return max(values, key=lambda item: (float(item.get("q_value", 0.0) or 0.0), int(item.get("visits", 0) or 0)))
