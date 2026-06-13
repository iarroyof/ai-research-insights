from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

from app.memory.rewards import important_terms


BIOMEDICAL_SYNONYMS = {
    "pd-1": "pd1",
    "pd 1": "pd1",
    "programmed cell death protein 1": "pd1",
    "programmed death 1": "pd1",
    "pd-l1": "pdl1",
    "pd l1": "pdl1",
    "programmed death ligand 1": "pdl1",
    "non-small cell lung cancer": "nsclc",
    "non small cell lung cancer": "nsclc",
    "non-small-cell lung cancer": "nsclc",
    "lung carcinoma": "lung cancer",
    "pulmonary carcinoma": "lung cancer",
    "tumor microenvironment": "tme",
    "tumour microenvironment": "tme",
    "caf": "caf",
    "cafs": "caf",
    "caf associated": "caf",
    "cancer associated fibroblast": "caf",
    "cancer associated fibroblasts": "caf",
    "carcinoma associated fibroblast": "caf",
    "carcinoma associated fibroblasts": "caf",
    "extracellular matrix": "ecm",
    "extra cellular matrix": "ecm",
    "tumour": "tumor",
    "tumours": "tumor",
    "platelets": "platelet",
    "t-cells": "t cell",
    "t cells": "t cell",
}

SYNONYM_GROUPS: dict[str, list[str]] = {}
for _alias, _canonical in BIOMEDICAL_SYNONYMS.items():
    SYNONYM_GROUPS.setdefault(_canonical, []).append(_alias)

PHRASE_PATTERNS = [
    r"non[- ]small[- ]cell lung cancer",
    r"programmed cell death protein 1",
    r"programmed death ligand 1",
    r"programmed death 1",
    r"platelet aggregation",
    r"immune checkpoint",
    r"checkpoint inhibitor",
    r"cancer[- ]associated fibroblasts?",
    r"carcinoma[- ]associated fibroblasts?",
    r"caf[- ]associated",
    r"extracellular matrix",
    r"tumou?r microenvironment",
    r"t[- ]cell",
    r"lung carcinoma",
    r"lung cancer",
    r"pd[- ]?1",
    r"pd[- ]?l1",
]


@dataclass(frozen=True)
class IdeaRecord:
    idea_id: str
    idea: str
    normalized_idea: str = ""
    parent_idea: str | None = None
    frequency: int = 0
    session_frequency: int = 0
    reward_sum: float = 0.0
    reward_count: int = 0
    cooccurring_ideas: list[str] = field(default_factory=list)
    child_ideas: list[str] = field(default_factory=list)
    synonyms: list[str] = field(default_factory=list)
    concept_path: list[str] = field(default_factory=list)
    last_turn_index: int = 0
    importance: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def stable_idea_id(tenant: str, scope: str, idea: str) -> str:
    raw = "\n".join((tenant, scope, idea)).encode("utf-8")
    return "idea_" + hashlib.sha256(raw).hexdigest()[:24]


def normalize_idea(value: str) -> str:
    text = " ".join(re.findall(r"[a-z0-9]+", (value or "").lower().replace("-", " ")))
    if not text:
        return ""
    if text in BIOMEDICAL_SYNONYMS:
        return BIOMEDICAL_SYNONYMS[text]
    if value.lower() in BIOMEDICAL_SYNONYMS:
        return BIOMEDICAL_SYNONYMS[value.lower()]
    if text.endswith("ies") and len(text) > 4:
        return text[:-3] + "y"
    if text.endswith("s") and len(text) > 4 and not text.endswith(("ss", "ous")):
        return text[:-1]
    return text


def synonyms_for(idea: str) -> list[str]:
    canonical = normalize_idea(idea)
    values = SYNONYM_GROUPS.get(canonical, [])
    return list(dict.fromkeys([v for v in values if normalize_idea(v) == canonical and v != canonical]))[:12]


def _phrase_ideas(text: str) -> list[str]:
    lowered = (text or "").lower()
    out: list[str] = []
    for pattern in PHRASE_PATTERNS:
        for match in re.finditer(pattern, lowered):
            normalized = normalize_idea(match.group(0))
            if normalized:
                out.append(normalized)
    return out


def parent_for_idea(idea: str, all_ideas: Iterable[str] | None = None) -> str | None:
    parts = idea.split()
    if len(parts) > 1:
        parent = parts[0]
        return parent if not all_ideas or parent in set(all_ideas) else parent
    if idea in {"pd1", "pdl1"}:
        return "immune checkpoint"
    if idea == "nsclc":
        return "lung cancer"
    return None


def concept_path_for(idea: str, all_ideas: Iterable[str] | None = None) -> list[str]:
    parent = parent_for_idea(idea, all_ideas)
    if parent and parent != idea:
        return [parent, idea]
    return [idea]


def extract_ideas(*texts: str, limit: int = 32) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for text in texts:
        candidates = _phrase_ideas(text)
        candidates.extend(normalize_idea(term) for term in important_terms(text or "", limit=limit * 2))
        candidates = [item for item in candidates if item]
        for idea in candidates:
            if idea in seen:
                continue
            seen.add(idea)
            out.append(idea)
            if len(out) >= limit:
                return out
    return out


def build_idea_updates(
    *,
    tenant: str,
    session_id: str,
    texts: Iterable[str],
    turn_index: int,
    reward_score: float = 0.0,
    shared: bool = False,
    limit: int = 32,
    scope_override: str | None = None,  # WP-F-2: e.g. user_{user_id}
) -> list[dict[str, Any]]:
    ideas = extract_ideas(*list(texts), limit=limit)
    counts = Counter(ideas)
    scope = scope_override if scope_override else ("shared" if shared else session_id)  # WP-F-2
    updates: list[dict[str, Any]] = []
    for idea in ideas:
        parent = parent_for_idea(idea, ideas)
        children = [other for other in ideas if parent_for_idea(other, ideas) == idea and other != idea][:16]
        cooccurring = [other for other in ideas if other != idea][:16]
        freq = int(counts[idea])
        importance = idea_importance(
            frequency=freq,
            reward_avg=reward_score,
            recency_turn_gap=0,
            cooccurrence_count=len(cooccurring),
        )
        updates.append(
            {
                "doc_type": "idea",
                "tenant": tenant,
                "scope": scope,
                "session_id": None if shared else session_id,
                "idea_id": stable_idea_id(tenant, scope, idea),
                "idea": idea,
                "normalized_idea": idea,
                "terms": list(dict.fromkeys([idea] + synonyms_for(idea) + concept_path_for(idea, ideas))),
                "parent_idea": parent,
                "child_ideas": children,
                "synonyms": synonyms_for(idea),
                "concept_path": concept_path_for(idea, ideas),
                "frequency_delta": freq,
                "session_frequency_delta": 0 if shared else freq,
                "reward_delta": float(reward_score),
                "reward_count_delta": 1,
                "cooccurring_ideas": cooccurring,
                "last_turn_index": turn_index,
                "importance": importance,
            }
        )
    return updates


def merge_idea_doc(existing: dict[str, Any] | None, update: dict[str, Any]) -> dict[str, Any]:
    existing = dict(existing or {})
    frequency = int(existing.get("frequency", 0) or 0) + int(update.get("frequency_delta", 0) or 0)
    session_frequency = int(existing.get("session_frequency", 0) or 0) + int(update.get("session_frequency_delta", 0) or 0)
    reward_sum = float(existing.get("reward_sum", 0.0) or 0.0) + float(update.get("reward_delta", 0.0) or 0.0)
    reward_count = int(existing.get("reward_count", 0) or 0) + int(update.get("reward_count_delta", 0) or 0)
    cooccurring = list(dict.fromkeys((existing.get("cooccurring_ideas") or []) + (update.get("cooccurring_ideas") or [])))[:48]
    child_ideas = list(dict.fromkeys((existing.get("child_ideas") or []) + (update.get("child_ideas") or [])))[:48]
    synonyms = list(dict.fromkeys((existing.get("synonyms") or []) + (update.get("synonyms") or [])))[:32]
    terms = list(dict.fromkeys((existing.get("terms") or []) + (update.get("terms") or [])))[:64]
    reward_avg = reward_sum / max(1, reward_count)
    doc = {
        **existing,
        "doc_type": "idea",
        "tenant": update["tenant"],
        "scope": update["scope"],
        "session_id": update.get("session_id"),
        "idea_id": update["idea_id"],
        "idea": update["idea"],
        "normalized_idea": update.get("normalized_idea") or update["idea"],
        "terms": terms,
        "parent_idea": update.get("parent_idea", existing.get("parent_idea")),
        "child_ideas": child_ideas,
        "synonyms": synonyms,
        "concept_path": update.get("concept_path") or existing.get("concept_path") or [update["idea"]],
        "frequency": frequency,
        "session_frequency": session_frequency,
        "reward_sum": round(reward_sum, 6),
        "reward_count": reward_count,
        "reward_avg": round(reward_avg, 6),
        "cooccurring_ideas": cooccurring,
        "last_turn_index": int(update.get("last_turn_index", existing.get("last_turn_index", 0)) or 0),
        "importance": idea_importance(
            frequency=frequency,
            reward_avg=reward_avg,
            recency_turn_gap=0,
            cooccurrence_count=len(cooccurring) + len(child_ideas),
        ),
    }
    return doc


def idea_importance(
    *,
    frequency: int,
    reward_avg: float,
    recency_turn_gap: int,
    cooccurrence_count: int,
) -> float:
    freq_score = math.log1p(max(0, frequency)) / math.log(33)
    reward_score = max(0.0, min(1.0, reward_avg))
    recency_score = 1.0 / (1.0 + max(0, recency_turn_gap) / 10.0)
    graph_score = math.log1p(max(0, cooccurrence_count)) / math.log(17)
    return round(max(0.0, min(1.0, 0.40 * freq_score + 0.30 * reward_score + 0.20 * recency_score + 0.10 * graph_score)), 4)


def rank_ideas(ideas: Iterable[dict[str, Any]], query: str, limit: int = 8) -> list[dict[str, Any]]:
    query_terms = set(extract_ideas(query, limit=32))
    scored: list[dict[str, Any]] = []
    for item in ideas:
        idea = normalize_idea(str(item.get("idea") or ""))
        normalized = normalize_idea(str(item.get("normalized_idea") or idea))
        co = set(str(v) for v in item.get("cooccurring_ideas") or [])
        children = set(str(v) for v in item.get("child_ideas") or [])
        synonyms = set(normalize_idea(str(v)) for v in item.get("synonyms") or [])
        path = set(str(v) for v in item.get("concept_path") or [])
        related = co | children | synonyms | path
        overlap = (1.0 if idea in query_terms or normalized in query_terms else 0.0) + 0.15 * len(query_terms & related)
        score = float(item.get("importance", 0.0) or 0.0) + overlap
        ranked = dict(item)
        ranked["_idea_score"] = round(score, 4)
        scored.append(ranked)
    return sorted(scored, key=lambda item: item.get("_idea_score", 0.0), reverse=True)[:limit]
