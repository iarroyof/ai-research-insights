from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, List

from app.memory.rewards import important_terms


PRESERVED_DOC_TYPES = {"landmark", "user_correction", "evidence_table", "conversation_frame"}
LOW_VALUE_ROLES = {"system", "tool"}


@dataclass(frozen=True)
class MemoryState:
    memory_state: str
    token_count: int
    reason: str = ""


@dataclass(frozen=True)
class WorkingSetSelection:
    selected: List[dict[str, Any]] = field(default_factory=list)
    evicted: List[dict[str, Any]] = field(default_factory=list)
    selected_tokens: int = 0


def estimate_tokens(text: str) -> int:
    """Conservative dependency-free token estimate for prompt budgeting."""
    text = (text or "").strip()
    if not text:
        return 0
    word_estimate = math.ceil(len(re.findall(r"\S+", text)) * 1.25)
    char_estimate = math.ceil(len(text) / 4)
    return max(1, word_estimate, char_estimate)


def memory_text(doc: dict[str, Any]) -> str:
    return str(doc.get("summary") or doc.get("text") or doc.get("snippet") or doc.get("idea") or "")


def memory_doc_id(*parts: str) -> str:
    raw = "\n".join(str(p) for p in parts).encode("utf-8")
    return "mem_" + hashlib.sha256(raw).hexdigest()[:24]


def _turn_index(doc: dict[str, Any]) -> int:
    try:
        return int(doc.get("turn_index", -1) or -1)
    except Exception:
        return -1


def _importance(doc: dict[str, Any]) -> float:
    try:
        return float(doc.get("importance", 0.0) or 0.0)
    except Exception:
        return 0.0


def memory_priority(
    doc: dict[str, Any],
    *,
    current_turn_index: int,
    working_turns: int,
    query_terms: set[str] | None = None,
) -> float:
    state = str(doc.get("memory_state") or "")
    state_bonus = {
        "promoted": 1.0,
        "working": 0.65,
        "episodic": 0.45,
        "evicted": 0.0,
    }.get(state, 0.25)
    importance = _importance(doc)
    evidence_bonus = 0.25 if _is_evidence_supported(doc) else 0.0
    reward_bonus = 0.0
    try:
        reward_bonus = max(0.0, min(0.2, float(doc.get("reward_score", 0.0) or doc.get("reward_avg", 0.0) or 0.0) * 0.2))
    except Exception:
        reward_bonus = 0.0
    age = max(0, current_turn_index - _turn_index(doc))
    recency = 1.0 / (1.0 + age / max(1, working_turns))
    relevance_bonus = 0.0
    if query_terms:
        doc_terms = set(doc.get("terms") or important_terms(memory_text(doc), 14))
        overlap = len(query_terms & doc_terms)
        relevance_bonus = min(0.45, overlap / max(4, len(query_terms)) * 0.9)
    return round(state_bonus + 0.55 * importance + evidence_bonus + reward_bonus + 0.35 * recency + relevance_bonus, 4)


def _is_evidence_supported(doc: dict[str, Any]) -> bool:
    if bool(doc.get("evidence_supported")) or bool(doc.get("pinned")):
        return True
    claims = doc.get("claim_support") or doc.get("claims") or []
    if isinstance(claims, list):
        return any(isinstance(item, dict) and item.get("status") == "entailed" for item in claims)
    return False


def classify_memory_state(
    doc: dict[str, Any],
    *,
    current_turn_index: int,
    working_turns: int,
    eviction_importance_threshold: float = 0.25,
) -> MemoryState:
    text = memory_text(doc)
    token_count = int(doc.get("token_count") or estimate_tokens(text))
    doc_type = str(doc.get("doc_type") or "")
    turn_index = _turn_index(doc)

    if doc_type in PRESERVED_DOC_TYPES:
        return MemoryState("promoted", token_count, f"preserved_{doc_type}")
    if _is_evidence_supported(doc):
        return MemoryState("promoted", token_count, "evidence_supported")
    if doc_type == "episodic_summary":
        return MemoryState("episodic", token_count, "episodic_summary")
    if turn_index >= max(0, current_turn_index - max(1, working_turns)):
        return MemoryState("working", token_count, "recent_turn")
    if _importance(doc) < eviction_importance_threshold or doc.get("role") in LOW_VALUE_ROLES:
        return MemoryState("evicted", token_count, "old_low_importance")
    return MemoryState("episodic", token_count, "older_reusable_context")


def _budget_copy(doc: dict[str, Any], *, token_budget: int) -> dict[str, Any] | None:
    text = memory_text(doc)
    tokens = estimate_tokens(text)
    if tokens <= token_budget:
        out = dict(doc)
        out["token_count"] = tokens
        return out

    summary = str(doc.get("summary") or "").strip()
    if summary and estimate_tokens(summary) <= token_budget:
        out = dict(doc)
        out["text"] = summary
        out["summary"] = summary
        out["token_count"] = estimate_tokens(summary)
        out["memory_compressed"] = True
        return out
    if token_budget >= 24:
        text_summary = compress_text_for_budget(text, token_budget)
        if text_summary:
            out = dict(doc)
            out["text"] = text_summary
            out["summary"] = text_summary
            out["token_count"] = estimate_tokens(text_summary)
            out["memory_compressed"] = True
            out["memory_compression_method"] = "extractive_budget"
            return out
    return None


def compress_text_for_budget(text: str, token_budget: int) -> str:
    text = " ".join((text or "").split())
    if not text or token_budget <= 0:
        return ""
    if estimate_tokens(text) <= token_budget:
        return text
    terms = important_terms(text, 10)
    sentence_parts = re.split(r"(?<=[.!?])\s+", text)
    kept: list[str] = []
    used = 0
    for sent in sentence_parts:
        cost = estimate_tokens(sent)
        if used + cost <= max(8, token_budget - 8):
            kept.append(sent)
            used += cost
        if used >= token_budget * 0.75:
            break
    if terms and estimate_tokens(" ".join(kept) + " Focus: " + ", ".join(terms[:8])) <= token_budget:
        kept.append("Focus: " + ", ".join(terms[:8]) + ".")
    summary = " ".join(kept).strip()
    if not summary:
        words = text.split()
        summary = " ".join(words[: max(8, int(token_budget / 1.25))])
    while summary and estimate_tokens(summary) > token_budget:
        summary = " ".join(summary.split()[:-1])
    return summary


def select_working_set(
    docs: Iterable[dict[str, Any]],
    *,
    token_budget: int,
    current_turn_index: int | None = None,
    working_turns: int = 8,
    eviction_importance_threshold: float = 0.25,
    query_text: str = "",
) -> WorkingSetSelection:
    current = current_turn_index
    items = [dict(doc) for doc in docs or []]
    if current is None:
        current = max([_turn_index(doc) for doc in items] or [0])

    annotated: list[tuple[dict[str, Any], MemoryState]] = []
    for doc in items:
        state = classify_memory_state(
            doc,
            current_turn_index=current,
            working_turns=working_turns,
            eviction_importance_threshold=eviction_importance_threshold,
        )
        doc["memory_state"] = state.memory_state
        doc["memory_state_reason"] = state.reason
        doc["token_count"] = state.token_count
        annotated.append((doc, state))

    selected: list[dict[str, Any]] = []
    evicted: list[dict[str, Any]] = []
    used = 0

    def add_if_fits(doc: dict[str, Any]) -> bool:
        nonlocal used
        remaining = max(0, token_budget - used)
        candidate = _budget_copy(doc, token_budget=remaining)
        if not candidate:
            return False
        used += int(candidate.get("token_count", 0) or 0)
        selected.append(candidate)
        return True

    query_terms = set(important_terms(query_text, 12)) if query_text else set()

    promoted = [doc for doc, state in annotated if state.memory_state == "promoted"]
    promoted.sort(
        key=lambda item: (
            memory_priority(item, current_turn_index=current, working_turns=working_turns, query_terms=query_terms),
            _importance(item),
            _turn_index(item),
        ),
        reverse=True,
    )
    for doc in promoted:
        if not add_if_fits(doc):
            evicted.append(doc)

    candidates = [doc for doc, state in annotated if state.memory_state in {"working", "episodic"}]
    candidates.sort(
        key=lambda item: (
            memory_priority(item, current_turn_index=current, working_turns=working_turns, query_terms=query_terms),
            _turn_index(item),
        ),
        reverse=True,
    )
    for doc in candidates:
        if not add_if_fits(doc):
            evicted.append(doc)

    evicted.extend(doc for doc, state in annotated if state.memory_state == "evicted" and doc not in evicted)
    selected.sort(key=lambda item: (_turn_index(item), str(item.get("role") or "")))
    return WorkingSetSelection(selected=selected, evicted=evicted, selected_tokens=used)


def build_episodic_summary(
    *,
    tenant: str,
    session_id: str,
    turn_index: int,
    messages: Iterable[dict[str, Any]],
    reward_score: float = 0.0,
    max_facts: int = 6,
) -> dict[str, Any]:
    items = [dict(item) for item in messages or [] if memory_text(item)]
    focus_terms = important_terms(" ".join(memory_text(item) for item in items), 18)
    supported = [item for item in items if _is_evidence_supported(item)]
    high_value = sorted(items, key=lambda item: (_importance(item), _turn_index(item)), reverse=True)[:max_facts]
    fact_lines = []
    for item in (supported or high_value)[:max_facts]:
        role = item.get("role") or item.get("doc_type") or "memory"
        fact_lines.append(f"{role}: {memory_text(item)[:260]}")

    parts = [f"Session episode through turn {turn_index}."]
    if focus_terms:
        parts.append("Focus: " + ", ".join(focus_terms[:12]) + ".")
    if fact_lines:
        parts.append("Evidence-supported facts: " + " | ".join(fact_lines))
    summary = " ".join(parts)
    first_turn = min([_turn_index(item) for item in items if _turn_index(item) >= 0] or [turn_index])
    last_turn = max([_turn_index(item) for item in items if _turn_index(item) >= 0] or [turn_index])

    return {
        "doc_type": "episodic_summary",
        "tenant": tenant,
        "session_id": session_id,
        "summary_id": memory_doc_id(tenant, session_id, "episodic", str(turn_index)),
        "summary": summary[:1800],
        "text": summary[:1800],
        "terms": focus_terms,
        "source_turn_start": first_turn,
        "source_turn_end": last_turn,
        "turn_index": turn_index,
        "memory_state": "episodic",
        "token_count": estimate_tokens(summary),
        "importance": round(max(0.4, min(1.0, 0.45 + 0.35 * reward_score + 0.05 * len(supported))), 4),
        "reward_score": round(float(reward_score), 4),
        "evidence_supported_fact_count": len(supported),
    }
