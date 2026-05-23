from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Awaitable, Callable, Iterable, List

from app.clients.llm import LLMClient
from app.config import settings
from app.memory.action_value import best_action_value
from app.memory.idea_index import extract_ideas, normalize_idea, synonyms_for
from app.memory.rewards import important_terms
from app.search.hybrid import hybrid_search_multilevel, hybrid_search_sentences


SearchFn = Callable[[str, str, dict[str, Any], int], Awaitable[List[dict[str, Any]]]]
LevelSearchFn = Callable[[str, str, str, dict[str, Any], int], Awaitable[List[dict[str, Any]]]]


NOISY_FEEDBACK_TERMS = {
    "figure", "fig", "table", "show", "shows", "shown", "define", "defines", "defined",
    "study", "studies", "paper", "article", "review", "result", "results", "data",
    "lastly", "therefore", "however", "specifically", "distinct", "value", "values",
    "method", "methods", "analysis", "using", "used", "found", "reported", "role",
    "administered", "question", "questions", "questionnaire", "questionnaires",
    "radio", "button", "buttons", "free", "format", "formats", "different",
    "four", "one", "ques", "text", "test", "tests", "guarantee", "guarantees",
    "important", "characteristic", "presence", "called", "greater", "compliance",
    "population", "following", "section", "sections", "outline", "expected",
    "effect", "modality", "modalities", "provide", "thus", "did",
    "physician", "noted", "talk", "patient", "patients", "correspondence",
    "latest", "start", "explaining", "explain", "candidate", "framework",
    "frameworks", "suggested", "then", "year",
    "cell", "cells", "tumor", "cancer", "carcinoma", "lung",
}
GENERIC_REFINEMENT_TERMS = {
    "biological", "complex", "energy", "environment", "factor", "factors", "growth",
    "health", "metabolic", "process", "processes", "pathway", "pathways", "select",
    "system", "systems", "tissue",
}

MECHANISTIC_SYNERGY_BRIDGE = [
    "mechanistic synergy tumor microenvironment crosstalk lung cancer",
    "cooperative interaction stromal immune metabolic hypoxia angiogenesis NSCLC",
    "CAF tumor associated macrophage Treg MDSC cytokine ECM remodeling EMT immune evasion",
]

TME_GROWTH_BRIDGE = [
    "tumor microenvironment factors promote tumor growth NSCLC lung cancer",
    "CAF TAM MDSC Treg hypoxia angiogenesis ECM remodeling EMT immune suppression",
    "stromal immune metabolic crosstalk cytokines lactate angiogenesis invasion proliferation",
]

MATH_PHARM_SYNERGY_TERMS = {
    "combination index", "ci value", "chou talalay", "dose response", "drug synergy",
    "therapeutic agent", "therapeutic agents", "combination therapy", "cytotoxicity",
    "ic50", "ctcae", "adverse event", "irae", "toxicity", "pharmacological",
}


def _contains_any(text: str, values: set[str] | list[str] | tuple[str, ...]) -> bool:
    lowered = (text or "").lower()
    return any(value in lowered for value in values)


def _domain_search_frame(message: str, notes: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    text = (message or "").lower()
    # Search notes are strategy hints. Specialized biomedical frames must be
    # justified by the current turn or stale successful terms override a pivot.
    asks_drug_synergy = _contains_any(text, {"drug", "therapy", "therapeutic", "dose", "combination index", "ci value", "pharmacological"})
    cancer_context = _contains_any(text, {"tme", "tumor microenvironment", "lung", "carcinoma", "cancer", "nsclc"})
    synergy_context = "synergy" in text or "synerg" in text or "crosstalk" in text
    mechanistic_context = _contains_any(text, {"mechanistic", "mechanism", "pathway", "crosstalk", "interaction", "cooperative", "functional synergy", "pivot"})

    preferred: list[str] = []
    avoid: list[str] = []
    frame = "general_biomedical"
    if cancer_context and synergy_context and not asks_drug_synergy:
        frame = "mechanistic_tme_synergy"
        preferred.extend(MECHANISTIC_SYNERGY_BRIDGE)
        avoid.extend(sorted(MATH_PHARM_SYNERGY_TERMS))
    if cancer_context and (_contains_any(text, {"tme", "tumor microenvironment"}) or "growth" in text):
        frame = "tme_tumor_growth" if frame == "general_biomedical" else frame
        preferred.extend(TME_GROWTH_BRIDGE)
    if mechanistic_context and not asks_drug_synergy:
        avoid.extend(sorted(MATH_PHARM_SYNERGY_TERMS))

    return {
        "frame": frame,
        "preferred_queries": list(dict.fromkeys(preferred))[:6],
        "avoid_terms": list(dict.fromkeys(avoid))[:24],
    }


@dataclass(frozen=True)
class SearchQueryVariant:
    label: str
    query: str
    strategy: str
    source: str = "deterministic"
    frame_id: str = "literal"
    frame_label: str = "Literal user frame"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AutoContextPlan:
    state_key: str
    action_key: str
    strategy: str
    levels: list[str] = field(default_factory=list)
    variants: list[SearchQueryVariant] = field(default_factory=list)
    notes_used: list[dict[str, Any]] = field(default_factory=list)
    action_value_hints: list[dict[str, Any]] = field(default_factory=list)
    search_frame: dict[str, Any] = field(default_factory=dict)
    candidate_frames: list[dict[str, Any]] = field(default_factory=list)
    used_llm: bool = False
    planner_note: str = ""

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["variants"] = [item.to_dict() for item in self.variants]
        return data


def _bucket_count(value: int) -> str:
    if value <= 0:
        return "none"
    if value <= 2:
        return "few"
    if value <= 5:
        return "medium"
    return "many"


def _length_bucket(message: str) -> str:
    count = len(important_terms(message, 80))
    if count <= 5:
        return "short"
    if count <= 14:
        return "medium"
    return "long"


def _intent_bucket(message: str) -> str:
    lowered = (message or "").lower()
    if any(word in lowered for word in ("compare", "versus", "vs", "difference")):
        return "compare"
    if any(word in lowered for word in ("mechanism", "pathway", "how does", "why does")):
        return "mechanism"
    if any(word in lowered for word in ("evidence", "trial", "study", "paper")):
        return "evidence"
    if "?" in lowered:
        return "question"
    return "statement"


def _frame_for_variant_label(label: str) -> tuple[str, str]:
    if label == "prior_frame":
        return "prior_frame", "Prior supported search frame"
    if label in {"normalized_ideas", "biomedical_synonyms"}:
        return "conceptual", "Normalized concept frame"
    if label in {"domain_bridge", "relation_probe"}:
        return "relation_bridge", "Relation/evidence bridge frame"
    if label == "llm_refined":
        return "llm_refined", "LLM refined frame"
    return "literal", "Literal user frame"


def _candidate_frames_from_variants(variants: list[SearchQueryVariant]) -> list[dict[str, Any]]:
    frames: dict[str, dict[str, Any]] = {}
    for variant in variants:
        frame = frames.setdefault(
            variant.frame_id,
            {
                "frame_id": variant.frame_id,
                "label": variant.frame_label,
                "query_labels": [],
                "queries": [],
                "source": variant.source,
            },
        )
        frame["query_labels"].append(variant.label)
        frame["queries"].append(variant.query)
        if frame["source"] != variant.source:
            frame["source"] = "mixed"
    return list(frames.values())


def search_state_key(message: str, *, selected_context_count: int = 0) -> str:
    """
    Return a non-lexical state bucket for search policy telemetry.

    The key intentionally avoids exact biomedical terms. It describes the shape
    of the request so action values can learn search strategy patterns without
    overfitting to one literal query.
    """
    idea_count = len(extract_ideas(message, limit=16))
    return "|".join(
        [
            "search:v1",
            f"len:{_length_bucket(message)}",
            f"intent:{_intent_bucket(message)}",
            f"biomed:{'yes' if idea_count else 'no'}",
            f"selected:{_bucket_count(selected_context_count)}",
        ]
    )


def search_action_key(
    *,
    query_count: int,
    level_count: int = 1,
    strategy: str,
    uses_synonyms: bool,
    used_llm: bool,
    used_notes: bool,
) -> str:
    return "|".join(
        [
            "search:v1",
            f"queries:{_bucket_count(query_count)}",
            f"levels:{_bucket_count(level_count)}",
            f"breadth:{strategy}",
            f"synonyms:{'yes' if uses_synonyms else 'no'}",
            f"llm:{'yes' if used_llm else 'no'}",
            f"notes:{'yes' if used_notes else 'no'}",
        ]
    )


def _compact_text(value: str, limit: int = 220) -> str:
    return " ".join(str(value or "").split())[:limit]


def _dedupe_queries(values: Iterable[tuple[str, str, str, str]], limit: int) -> list[SearchQueryVariant]:
    seen: set[str] = set()
    out: list[SearchQueryVariant] = []
    for label, query, strategy, source in values:
        compact = _compact_text(query, 260)
        key = compact.lower()
        if not compact or key in seen:
            continue
        seen.add(key)
        frame_id, frame_label = _frame_for_variant_label(label)
        out.append(
            SearchQueryVariant(
                label=label,
                query=compact,
                strategy=strategy,
                source=source,
                frame_id=frame_id,
                frame_label=frame_label,
            )
        )
        if len(out) >= limit:
            break
    return out


def _strategy_from_hints(action_value_hints: list[dict[str, Any]], notes: list[dict[str, Any]]) -> str:
    best = best_action_value(action_value_hints)
    if best:
        action = str(best.get("action_key") or "")
        if "breadth:wide" in action:
            return "wide"
        if "breadth:narrow" in action:
            return "narrow"
    note_text = " ".join(str(n.get("note") or n.get("summary") or "") for n in notes).lower()
    if any(word in note_text for word in ("broaden", "synonym", "alternate")):
        return "wide"
    if any(word in note_text for word in ("narrow", "specific", "exact phrase")):
        return "narrow"
    return "medium"


def deterministic_query_variants(
    message: str,
    *,
    strategy: str = "medium",
    max_variants: int = 4,
    search_frame: dict[str, Any] | None = None,
) -> list[SearchQueryVariant]:
    terms = important_terms(message, limit=18)
    ideas = extract_ideas(message, limit=10)
    normalized = list(dict.fromkeys(normalize_idea(item) for item in ideas if normalize_idea(item)))
    synonym_terms: list[str] = []
    for idea in normalized[:6]:
        synonym_terms.extend(synonyms_for(idea)[:3])
    synonym_terms = list(dict.fromkeys(synonym_terms))

    candidates: list[tuple[str, str, str, str]] = [
        ("original", message, "narrow" if strategy == "narrow" else strategy, "deterministic"),
    ]
    for preferred in (search_frame or {}).get("preferred_queries", [])[:3]:
        candidates.append(("domain_bridge", preferred, "wide" if strategy != "narrow" else "medium", "deterministic"))
    if terms:
        candidates.append(("important_terms", " ".join(terms[:10]), strategy, "deterministic"))
    if terms and (_intent_bucket(message) in {"mechanism", "compare", "evidence"} or _query_ambiguity(message, 0) != "low"):
        relation_words = ["mechanism", "evidence", "relationship"] if _intent_bucket(message) == "mechanism" else ["evidence", "relationship"]
        candidates.append(("relation_probe", " ".join(list(dict.fromkeys(terms[:10] + relation_words))), "wide", "deterministic"))
    if normalized:
        candidates.append(("normalized_ideas", " ".join(normalized[:8]), "medium", "deterministic"))
    if synonym_terms:
        candidates.append(("biomedical_synonyms", " ".join(list(dict.fromkeys(normalized + synonym_terms))[:12]), "wide", "deterministic"))
    if strategy == "wide" and terms and normalized:
        candidates.append(("mixed_wide", " ".join(list(dict.fromkeys(normalized + terms[:10] + synonym_terms[:6]))), "wide", "deterministic"))

    return _dedupe_queries(candidates, max(1, max_variants))


FOLLOWUP_CONTEXT_MARKERS = {
    "as above", "candidate framework", "candidate frameworks", "continue", "develop",
    "earlier", "latest candidate", "previous", "start by", "suggested", "those",
    "use what is actually supported", "what is actually supported",
}


def _is_followup_reference(message: str) -> bool:
    return _contains_any(message, FOLLOWUP_CONTEXT_MARKERS)


def _prior_frame_variants(notes: list[dict[str, Any]], limit: int = 2) -> list[SearchQueryVariant]:
    candidates: list[tuple[str, str, str, str]] = []
    for note in notes:
        search_plan = note.get("search_plan") if isinstance(note, dict) else None
        for variant in (search_plan or {}).get("variants", []) or []:
            if not isinstance(variant, dict):
                continue
            query = str(variant.get("query") or "").strip()
            if not query:
                continue
            candidates.append(("prior_frame", query, str(variant.get("strategy") or "medium"), "memory"))
            if len(candidates) >= limit:
                break
        if len(candidates) >= limit:
            break
    return _dedupe_queries(candidates, limit)


def deterministic_search_levels(message: str, *, strategy: str = "medium") -> list[str]:
    """
    Pick a broad-to-specific level order.

    Title search identifies candidate papers and vocabulary, paper/chunk search
    adds broader context, and sentence/triplet search finds exact evidence.
    Later levels receive compact terms from earlier results.
    """
    intent = _intent_bucket(message)
    if strategy == "narrow":
        return ["sentence", "title", "paper"]
    if intent in {"evidence", "compare", "question", "mechanism"}:
        return ["title", "paper", "sentence"]
    return ["title", "sentence", "paper"]


def _extract_json_object(text: str) -> dict[str, Any] | None:
    cleaned = (text or "").strip()
    if not cleaned:
        return None
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()
    try:
        data = json.loads(cleaned)
        return data if isinstance(data, dict) else None
    except Exception:
        pass
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _fallback_queries_from_text(text: str, limit: int) -> list[str]:
    cleaned = re.sub(r"```(?:json)?|```", "", text or "", flags=re.IGNORECASE).strip()
    out: list[str] = []
    quoted = re.findall(r'"([^"\n]{4,180})"', cleaned)
    for item in quoted:
        if item.lower() in {"queries", "note"}:
            continue
        out.append(item)
    if len(out) < limit:
        for raw_line in cleaned.splitlines():
            line = re.sub(r"^\s*[-*\d.)]+\s*", "", raw_line).strip()
            line = re.sub(r"^(query|search|variant)\s*[:=-]\s*", "", line, flags=re.IGNORECASE).strip()
            if not line or len(line) > 180:
                continue
            lowered = line.lower()
            if any(marker in lowered for marker in ("json", "note:", "because", "should", "return ")):
                continue
            if "{" in line or "}" in line or "[" in line or "]" in line:
                continue
            out.append(line)
    deduped = list(dict.fromkeys(_compact_text(item, 180) for item in out if item.strip()))
    return deduped[:limit]


async def llm_refine_variants(
    message: str,
    *,
    base_variants: list[SearchQueryVariant],
    notes: list[dict[str, Any]],
    action_value_hints: list[dict[str, Any]],
    search_frame: dict[str, Any],
    max_variants: int,
) -> tuple[list[SearchQueryVariant], str]:
    note_lines = [
        f"- {_compact_text(str(item.get('note') or item.get('summary') or ''), 180)}"
        for item in notes[:4]
        if item.get("note") or item.get("summary")
    ]
    hint_lines = [
        f"- action={item.get('action_key')} q={item.get('q_value')} visits={item.get('visits')}"
        for item in action_value_hints[:4]
    ]
    messages = [
        {
            "role": "system",
            "content": (
                "You are a biomedical multilevel search planner for OpenSearch BM25. "
                "The retrieval levels are title, paper, and sentence. Title search finds candidate papers and vocabulary; "
                "paper/chunk search gathers broader article context; sentence/triplet search finds exact evidence sentences. "
                "Later searches will be expanded with compact terms from earlier levels. Return JSON only. "
                "Do not include hidden reasoning. Improve search breadth and terminology."
            ),
        },
        {
            "role": "user",
            "content": (
                f"User query: {message[:1000]}\n"
                f"Base queries: {[item.query for item in base_variants]}\n"
                f"Search frame: {search_frame.get('frame', 'general_biomedical')}\n"
                f"Preferred biomedical query bridges: {search_frame.get('preferred_queries', [])}\n"
                f"Avoid off-target senses/terms unless explicitly asked: {search_frame.get('avoid_terms', [])}\n"
                f"Prior search notes:\n{chr(10).join(note_lines) if note_lines else '- none'}\n"
                f"Strategy action hints:\n{chr(10).join(hint_lines) if hint_lines else '- none'}\n\n"
                "Return JSON with keys: queries (array of 1-4 concise keyword queries for all levels), note (one sentence)."
            ),
        },
    ]
    text = await LLMClient().chat_once(
        messages,
        provider=settings.llm.context_manager_provider,
        max_tokens=900,
    )
    data = _extract_json_object(text) or {}
    raw_queries = data.get("queries") or []
    if not isinstance(raw_queries, list):
        raw_queries = []
    if not raw_queries:
        raw_queries = _fallback_queries_from_text(text, max_variants)
    candidates = [
        ("llm_refined", str(query), "wide", "llm")
        for query in raw_queries
        if isinstance(query, (str, int, float)) and str(query).strip()
    ]
    note = str(data.get("note") or "").strip()
    if not note and text:
        note = "LLM refinement response was parsed with fallback query extraction."
    return _dedupe_queries(candidates, max_variants), _compact_text(note, 500)


async def plan_auto_context(
    *,
    message: str,
    selected_context_count: int,
    notes: list[dict[str, Any]],
    action_value_hints: list[dict[str, Any]],
    max_variants: int,
    allow_llm_refine: bool,
) -> AutoContextPlan:
    state = search_state_key(message, selected_context_count=selected_context_count)
    strategy = _strategy_from_hints(action_value_hints, notes)
    search_frame = _domain_search_frame(message, notes)
    base = deterministic_query_variants(message, strategy=strategy, max_variants=max_variants, search_frame=search_frame)
    prior: list[SearchQueryVariant] = []
    if _is_followup_reference(message):
        prior = _prior_frame_variants(notes)
        if prior:
            merged = [*prior, *base]
            base = _dedupe_queries(
                [(item.label, item.query, item.strategy, item.source) for item in merged],
                max_variants,
            )
    levels = deterministic_search_levels(message, strategy=strategy)
    used_llm = False
    planner_note = ""
    variants = list(base)

    if allow_llm_refine and base:
        try:
            llm_variants, planner_note = await llm_refine_variants(
                message,
                base_variants=base,
                notes=notes,
                action_value_hints=action_value_hints,
                search_frame=search_frame,
                max_variants=max_variants,
            )
            if llm_variants:
                merged_variants = [*llm_variants, *base]
                if prior:
                    merged_variants = [*prior, *merged_variants]
                variants = _dedupe_queries(
                    [(item.label, item.query, item.strategy, item.source) for item in merged_variants],
                    max_variants,
                )
                used_llm = True
        except Exception as e:
            print(f"[WARN] auto-context LLM query refinement failed: {e}")

    uses_synonyms = any(item.label == "biomedical_synonyms" or item.source == "llm" for item in variants)
    action = search_action_key(
        query_count=len(variants),
        level_count=len(levels),
        strategy=strategy,
        uses_synonyms=uses_synonyms,
        used_llm=used_llm,
        used_notes=bool(notes),
    )
    return AutoContextPlan(
        state_key=state,
        action_key=action,
        strategy=strategy,
        levels=levels,
        variants=variants,
        notes_used=notes[:4],
        action_value_hints=action_value_hints[:4],
        search_frame=search_frame,
        candidate_frames=_candidate_frames_from_variants(variants),
        used_llm=used_llm,
        planner_note=planner_note,
    )


def _snippet_key(item: dict[str, Any]) -> tuple[Any, Any, str]:
    return (
        item.get("search_level"),
        item.get("paper_id"),
        item.get("sent_id"),
        str(item.get("text") or item.get("sentence_text") or item.get("title") or "")[:300],
    )


def _level_budgets(levels: list[str], total: int) -> dict[str, int]:
    if not levels:
        return {}
    total = max(1, total)
    if set(levels) >= {"title", "paper", "sentence"}:
        title = max(1, total // 4)
        paper = max(1, total // 4)
        sentence = max(1, total - title - paper)
        return {"title": title, "paper": paper, "sentence": sentence}
    base = max(1, total // len(levels))
    budgets = {level: base for level in levels}
    budgets[levels[-1]] += max(0, total - sum(budgets.values()))
    return budgets


def _feedback_terms_from_results(results: list[dict[str, Any]], limit: int = 8) -> list[str]:
    return _feedback_term_report(results, limit=limit)["accepted_terms"]


def _feedback_text(item: dict[str, Any]) -> str:
    return " ".join(
        str(item.get(field) or "")
        for field in ("title", "text", "sentence_text", "subject", "relation", "object")
        if item.get(field)
    )


def _feedback_candidates(texts: list[str], limit: int) -> list[str]:
    ideas = extract_ideas(*texts, limit=limit)
    terms = important_terms(" ".join(texts), limit=limit * 2)
    text_term_sets = [set(important_terms(text, 96)) for text in texts]
    out: list[tuple[float, str]] = []
    for term in list(dict.fromkeys(ideas + terms)):
        cleaned = term.lower().strip()
        if cleaned in NOISY_FEEDBACK_TERMS:
            continue
        if cleaned.endswith(("ou", "ly")) and cleaned not in {"t cell"}:
            continue
        if len(cleaned) < 3:
            continue
        pieces = set(important_terms(term, 12))
        support = sum(1 for item_terms in text_term_sets if pieces & item_terms)
        phrase_bonus = 0.8 if " " in cleaned else 0.0
        repeat_bonus = min(1.5, max(0, support - 1) * 0.5)
        generic_penalty = 1.2 if cleaned in GENERIC_REFINEMENT_TERMS else 0.0
        out.append((phrase_bonus + repeat_bonus - generic_penalty, term))
    out.sort(key=lambda item: (-item[0], list(dict.fromkeys(ideas + terms)).index(item[1])))
    return [term for _, term in out[:limit]]


def _is_low_value_feedback_text(text: str, anchor_queries: list[str] | None = None) -> bool:
    lowered = (text or "").lower()
    if "correspondence" in lowered:
        return True
    if "physician" in lowered and "noted" in lowered and _contains_any(lowered, {"talk", "patient", "risk", "benefit"}):
        return True
    anchor_text = " ".join(anchor_queries or []).lower()
    is_cross_domain_probe = _contains_any(
        anchor_text,
        {"analogy", "compare", "inspired", "relationship", "relating", "framework"},
    )
    if (
        is_cross_domain_probe
        and _contains_any(lowered, {"case", "diagnosed", "patient", "received", "follow-up"})
        and not _contains_any(lowered, {"cancer", "tumor", "oncology", "malignan"})
    ):
        return True
    return False


def _feedback_term_report(
    results: list[dict[str, Any]],
    *,
    anchor_queries: list[str] | None = None,
    limit: int = 8,
) -> dict[str, Any]:
    anchor_terms = {
        term
        for term in important_terms(" ".join(anchor_queries or []), 96)
        if term not in GENERIC_REFINEMENT_TERMS
        and term not in AMBIGUITY_MARKERS
        and term not in PUZZLE_NODE_STOP_TERMS
    }
    accepted_texts: list[str] = []
    rejected_texts: list[str] = []
    rejected_result_count = 0
    for item in results[:12]:
        text = _feedback_text(item)
        if not text:
            continue
        if _is_low_value_feedback_text(text, anchor_queries):
            rejected_texts.append(text)
            rejected_result_count += 1
            continue
        item_terms = set(important_terms(text, 96))
        has_structured_relation = bool(item.get("subject") and (item.get("relation") or item.get("predicate")) and item.get("object"))
        has_anchor_overlap = bool(anchor_terms & item_terms) if anchor_terms else True
        if anchor_terms and not has_anchor_overlap and not has_structured_relation:
            rejected_texts.append(text)
            rejected_result_count += 1
            continue
        accepted_texts.append(text)

    accepted_terms = _feedback_candidates(accepted_texts, limit)
    rejected_terms = [
        term
        for term in _feedback_candidates(rejected_texts, limit)
        if term not in set(accepted_terms)
    ][:limit]
    return {
        "accepted_terms": accepted_terms,
        "rejected_terms": rejected_terms,
        "rejected_result_count": rejected_result_count,
        "ungrounded_feedback_term_count": 0,
    }


def _is_off_topic_hit(item: dict[str, Any], search_frame: dict[str, Any]) -> bool:
    avoid_terms = set(search_frame.get("avoid_terms") or [])
    if not avoid_terms:
        return False
    text = " ".join(
        str(item.get(field) or "")
        for field in ("title", "text", "sentence_text", "subject", "relation", "object")
    ).lower()
    if not _contains_any(text, avoid_terms):
        return False
    preferred_markers = {
        "tme", "tumor microenvironment", "stromal", "fibroblast", "caf", "macrophage",
        "hypoxia", "angiogenesis", "immune evasion", "ecm", "emt", "nsclc",
        "lung cancer", "lung carcinoma", "mechanistic", "crosstalk",
    }
    return not _contains_any(text, preferred_markers)


def _query_with_feedback(query: str, feedback_terms: list[str], *, level: str) -> str:
    if not feedback_terms or level == "title":
        return query
    expanded = list(dict.fromkeys([query] + feedback_terms[:6]))
    return _compact_text(" ".join(expanded), 320)


def _result_note(*, plan: AutoContextPlan, result_count: int) -> str:
    if result_count <= 0:
        return (
            f"Auto-context multilevel search found no snippets using {len(plan.variants)} query variants across {len(plan.levels)} levels; "
            "next similar searches should broaden terminology or try biomedical synonyms."
        )
    if plan.strategy == "wide":
        return (
            f"Auto-context multilevel search found {result_count} snippets with broad/synonym query variants; "
            "keep broad variants when the user query is short or concept-heavy."
        )
    return (
        f"Auto-context multilevel search found {result_count} snippets using {plan.strategy} query variants; "
        "reuse this strategy for similarly shaped evidence questions if reward remains positive."
    )


AMBIGUITY_MARKERS = {
    "something", "someone", "anything", "relationship", "relating", "related",
    "connect", "link", "somehow", "environment", "develop",
}
PUZZLE_NODE_STOP_TERMS = {
    "develop", "relating", "related", "relationship", "something", "else", "promotes",
    "promote", "environment", "question", "evidence", "mechanism", "pathway", "body",
    "latest", "start", "explain", "explaining", "conceptual", "then", "candidate",
    "framework", "frameworks", "suggested",
}


def _query_ambiguity(message: str, result_count: int) -> str:
    message_terms = set(important_terms(message, 48))
    idea_count = len(extract_ideas(message, limit=12))
    marker_count = len(message_terms & AMBIGUITY_MARKERS)
    if marker_count >= 2 or (marker_count and idea_count <= 3):
        return "high"
    if marker_count or result_count <= 1 or idea_count <= 2:
        return "medium"
    return "low"


def _evidence_assembly(
    *,
    message: str,
    plan: AutoContextPlan,
    snippets: list[dict[str, Any]],
    level_reports: list[dict[str, Any]],
) -> dict[str, Any]:
    prior_frame_queries = [item.query for item in plan.variants if item.frame_id == "prior_frame"]
    analysis_message = message
    if prior_frame_queries and _is_followup_reference(message):
        analysis_message = f"{message} {' '.join(prior_frame_queries[:2])}"
    result_levels = {
        str(report.get("level") or ""): int(report.get("result_count", 0) or 0)
        for report in level_reports
    }
    distinct_papers = {
        str(item.get("paper_id") or item.get("pmid") or item.get("pmcid") or "")
        for item in snippets
        if item.get("paper_id") or item.get("pmid") or item.get("pmcid")
    }
    nonempty_levels = sum(1 for level in plan.levels if result_levels.get(level, 0) > 0)
    frame_result_counts: dict[str, int] = {}
    for item in snippets:
        frame_id = str(item.get("auto_frame_id") or "unknown")
        frame_result_counts[frame_id] = frame_result_counts.get(frame_id, 0) + 1
    level_coverage = nonempty_levels / max(1, len(plan.levels))
    accepted_count = sum(len(report.get("feedback_terms_added") or []) for report in level_reports)
    rejected_count = sum(len(report.get("rejected_feedback_terms") or []) for report in level_reports)
    rejected_results = sum(int(report.get("rejected_feedback_result_count", 0) or 0) for report in level_reports)
    ungrounded_count = sum(int(report.get("ungrounded_feedback_term_count", 0) or 0) for report in level_reports)
    drift_penalty = ungrounded_count / max(1, accepted_count + ungrounded_count)
    breadth = min(1.0, len(distinct_papers) / max(1, min(3, len(plan.levels))))
    assembly_quality = round(max(0.0, min(1.0, (0.65 * level_coverage) + (0.25 * breadth) + (0.10 * (1.0 - drift_penalty)))), 4)
    ambiguity = _query_ambiguity(analysis_message, len(snippets))
    evidence_text = " ".join(_feedback_text(item) for item in snippets)
    evidence_terms = set(important_terms(evidence_text, 160))
    raw_query_nodes = list(dict.fromkeys(extract_ideas(analysis_message, limit=8) + important_terms(analysis_message, 12)))
    if re.search(r"\bp\s*h\b|\bph\b", analysis_message, flags=re.IGNORECASE):
        raw_query_nodes.append("pH")
    query_nodes: list[str] = []
    node_keys: set[str] = set()
    for node in raw_query_nodes:
        cleaned = str(node).strip()
        key = normalize_idea(cleaned) or cleaned.lower().rstrip("s")
        if not cleaned or cleaned.lower() in PUZZLE_NODE_STOP_TERMS or key in node_keys:
            continue
        node_keys.add(key)
        query_nodes.append(cleaned)
        if len(query_nodes) >= 10:
            break

    def node_matches_text(node: str, text: str, item_terms: set[str]) -> bool:
        if node.lower() == "ph":
            return bool(re.search(r"\bp\s*h\b|\bph\b", text, flags=re.IGNORECASE))
        return bool(set(important_terms(node, 8)) & item_terms)

    covered_nodes = [node for node in query_nodes if node_matches_text(node, evidence_text, evidence_terms)]
    missing_nodes = [node for node in query_nodes if node not in set(covered_nodes)]
    relation_markers = {
        "activates", "affects", "associated", "association", "causes", "contributes",
        "drives", "increases", "inhibits", "links", "promotes", "reduces",
        "regulates", "relationship", "through", "via",
    }
    relation_evidence_count = 0
    for item in snippets:
        item_text = _feedback_text(item)
        item_terms = set(important_terms(item_text, 96))
        node_overlap = sum(1 for node in query_nodes if node_matches_text(node, item_text, item_terms))
        relation_like = bool(item_terms & relation_markers or item.get("relation") or item.get("predicate"))
        if node_overlap >= 2 and relation_like:
            relation_evidence_count += 1
    if relation_evidence_count >= 2 and len(covered_nodes) >= 2:
        edge_status = "supported"
    elif relation_evidence_count or len(covered_nodes) >= 2:
        edge_status = "partial"
    else:
        edge_status = "missing"
    clarification_needed = (
        ambiguity == "high"
        or (ambiguity == "medium" and edge_status == "missing")
        or (bool(prior_frame_queries) and _is_followup_reference(message) and edge_status != "supported")
    )
    frame_summaries = [
        {
            "frame_id": frame.get("frame_id"),
            "label": frame.get("label"),
            "query_labels": frame.get("query_labels", [])[:4],
            "result_count": frame_result_counts.get(str(frame.get("frame_id") or ""), 0),
        }
        for frame in plan.candidate_frames
    ]
    clarification_line = ""
    if clarification_needed:
        frame_labels = [str(frame.get("label") or frame.get("frame_id")) for frame in frame_summaries[:3]]
        clarification_line = (
            "- The opening paragraph must end with one focused textual clarification before any list or pathway steps. "
            f"Ask which evidence frame should lead ({', '.join(frame_labels)}). Do not use a UI choice widget. "
            "Ask only this opening clarification; do not repeat it later as A/B/C or numbered frame choices.\n"
        )
    prompt_context = (
        "Auto-context evidence assembly:\n"
        f"- Information need: {_intent_bucket(message)}.\n"
        f"- User query ambiguity: {ambiguity}.\n"
        f"- Follow-up prior frame active: {'yes' if prior_frame_queries else 'no'}.\n"
        f"- Retrieved evidence levels: {', '.join(level for level, count in result_levels.items() if count) or 'none'}.\n"
        f"- Candidate evidence frames: {', '.join(str(frame.get('label')) for frame in frame_summaries[:4]) or 'literal'}.\n"
        f"{clarification_line}"
        "- Treat retrieved snippets as evidence pieces, not a completed causal chain. "
        "Do not assert a bridge between pieces unless the supplied context supports that bridge. "
        "Do not fill a missing edge with a new example or mediator that is absent from the supplied snippets. "
        "Do not name absent example candidates merely to illustrate a missing edge. "
        "Do not name a candidate therapy, agent, framework, pathway, or experiment unless the supplied context supports that named candidate. "
        "If the user asks for only supported evidence, do not offer a hypothetical frame that invents candidate nodes. "
        "When the requested relation is underspecified or evidence edges are missing, state the supported partial structure and ask one focused clarification."
    )
    return {
        "information_need": _intent_bucket(message),
        "query_ambiguity": ambiguity,
        "clarification_recommended": clarification_needed,
        "level_result_counts": result_levels,
        "frame_result_counts": frame_result_counts,
        "candidate_frames": frame_summaries,
        "distinct_paper_count": len(distinct_papers),
        "assembly_quality": assembly_quality,
        "refinement_quality": {
            "accepted_feedback_term_count": accepted_count,
            "rejected_feedback_term_count": rejected_count,
            "rejected_feedback_result_count": rejected_results,
            "ungrounded_feedback_term_count": ungrounded_count,
        },
        "prompt_context": prompt_context,
        "evidence_puzzle": {
            "candidate_nodes": query_nodes,
            "covered_nodes": covered_nodes,
            "missing_nodes": missing_nodes,
            "relation_evidence_count": relation_evidence_count,
            "edge_support_status": edge_status,
        },
    }


async def build_auto_context(
    *,
    tenant: str,
    session_id: str,
    message: str,
    store: Any,
    selected_context_count: int = 0,
    confidence_min: float = 0.5,
    search_fn: SearchFn | None = None,
    multilevel_search_fn: LevelSearchFn | None = None,
) -> dict[str, Any]:
    max_variants = max(1, int(settings.memory.auto_context_query_variants))
    k_total = max(1, int(settings.memory.auto_context_k))
    notes = await store.search_policy_notes(session_id=session_id, limit=4)
    state = search_state_key(message, selected_context_count=selected_context_count)
    action_value_hints = await store.action_values(session_id=session_id, state_key=state, limit=4)
    plan = await plan_auto_context(
        message=message,
        selected_context_count=selected_context_count,
        notes=notes,
        action_value_hints=action_value_hints,
        max_variants=max_variants,
        allow_llm_refine=bool(settings.memory.auto_context_llm_refine),
    )

    seen: set[tuple[Any, Any, str]] = set()
    snippets: list[dict[str, Any]] = []
    level_reports: list[dict[str, Any]] = []
    feedback_terms: list[str] = []
    skipped_off_topic = 0
    level_budgets = _level_budgets(plan.levels, k_total)

    if multilevel_search_fn is None and search_fn is not None:
        async def compat_level_search(tenant_arg: str, level: str, query: str, filters: dict[str, Any], k: int) -> list[dict[str, Any]]:
            return await search_fn(tenant_arg, query, filters, k)
        multilevel_search_fn = compat_level_search
    if multilevel_search_fn is None:
        multilevel_search_fn = hybrid_search_multilevel

    for level_index, level in enumerate(plan.levels):
        if len(snippets) >= k_total:
            break
        level_budget = max(1, min(level_budgets.get(level, 1), k_total - len(snippets)))
        per_query_k = max(2, math.ceil(level_budget / max(1, len(plan.variants))) + 1)
        level_added: list[dict[str, Any]] = []
        level_queries: list[str] = []
        for query_index, variant in enumerate(plan.variants):
            query = _query_with_feedback(variant.query, feedback_terms, level=level)
            level_queries.append(query)
            try:
                hits = await multilevel_search_fn(
                    tenant,
                    level,
                    query,
                    {
                        "confidence_min": confidence_min,
                        "feedback_terms": feedback_terms,
                        "search_level": level,
                    },
                    per_query_k,
                )
            except Exception as e:
                print(f"[WARN] auto-context level={level} search failed for variant {variant.label}: {e}")
                continue
            for rank, hit in enumerate(hits or [], start=1):
                item = dict(hit)
                if _is_off_topic_hit(item, plan.search_frame):
                    skipped_off_topic += 1
                    continue
                item["search_level"] = item.get("search_level") or level
                item.setdefault("text", item.get("sentence_text") or item.get("abstract") or item.get("title") or "")
                key = _snippet_key(item)
                if key in seen:
                    continue
                seen.add(key)
                item["source"] = "auto_context"
                item["auto_context"] = True
                item["auto_query"] = query
                item["auto_query_label"] = variant.label
                item["auto_query_source"] = variant.source
                item["auto_frame_id"] = variant.frame_id
                item["auto_frame_label"] = variant.frame_label
                item["search_strategy"] = variant.strategy
                item["search_query_index"] = query_index
                item["search_level_index"] = level_index
                item["search_rank"] = rank
                item["feedback_terms_used"] = feedback_terms[:8]
                snippets.append(item)
                level_added.append(item)
                if len(level_added) >= level_budget or len(snippets) >= k_total:
                    break
            if len(level_added) >= level_budget or len(snippets) >= k_total:
                break
        feedback_report = _feedback_term_report(
            level_added,
            anchor_queries=[message, *[item.query for item in plan.variants], *feedback_terms],
        )
        new_terms = feedback_report["accepted_terms"]
        feedback_terms = list(dict.fromkeys(feedback_terms + new_terms))[:12]
        level_reports.append(
            {
                "level": level,
                "query_count": len(level_queries),
                "queries": level_queries[:6],
                "result_count": len(level_added),
                "feedback_terms_added": new_terms[:8],
                "rejected_feedback_terms": feedback_report["rejected_terms"][:8],
                "rejected_feedback_result_count": feedback_report["rejected_result_count"],
                "ungrounded_feedback_term_count": feedback_report["ungrounded_feedback_term_count"],
                "feedback_terms_after": feedback_terms[:8],
            }
        )

    payload = plan.to_dict()
    payload.update(
        {
            "result_count": len(snippets),
            "note": _result_note(plan=plan, result_count=len(snippets)),
            "query_labels": [item.label for item in plan.variants],
            "candidate_frames": plan.candidate_frames,
            "levels": plan.levels,
            "level_reports": level_reports,
            "feedback_terms": feedback_terms[:12],
            "skipped_off_topic_count": skipped_off_topic,
            "evidence_assembly": _evidence_assembly(
                message=message,
                plan=plan,
                snippets=snippets,
                level_reports=level_reports,
            ),
        }
    )
    return {"snippets": snippets, "plan": payload}
