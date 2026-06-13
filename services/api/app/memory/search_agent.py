from __future__ import annotations

import json
import math
import copy
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Awaitable, Callable, Iterable, List

from app.clients.llm import LLMClient
from app.config import settings
from app.memory.action_value import best_action_value
from app.memory.idea_index import extract_ideas, normalize_idea, synonyms_for
from app.prompts.agent_prompts import (
    frame_system_prompt,
    intent_resolution_system_prompt,
    ner_grounding_system_prompt,
)
from app.memory.intent_router import ROUTER_CONF_THRESHOLD, classify_intent_zeroshot
from app.memory.rewards import distractor_ratio, gap_closure_score, important_terms, query_novelty
from app.search.hybrid import hybrid_search_multilevel, hybrid_search_sentences
# Modules 2/3: vocabulary store (feature-flagged; no-op when VOCAB_STORE_ENABLED!=true)
from app.memory.vocabulary_store import VocabularyStore


SearchFn = Callable[[str, str, dict[str, Any], int], Awaitable[List[dict[str, Any]]]]
LevelSearchFn = Callable[[str, str, str, dict[str, Any], int], Awaitable[List[dict[str, Any]]]]




@dataclass
class GapSpec:
    """Structured gap between confirmed evidence and what is still needed.

    Accumulated across retrieval levels inside build_auto_context().
    Consumed by _external_retry_queries() to steer retry queries toward
    genuinely missing evidence (WP-B, ReasonRAG-style).
    """

    confirmed_entities: set = field(default_factory=set)
    missing_entities: set = field(default_factory=set)
    coverage_ratio: float = 0.0
    query_entities: set = field(default_factory=set)   # NER-detected from user query
    entity_map: dict = field(default_factory=dict)     # entity -> {status, context_match}

    def update_coverage(self) -> None:
        total = len(self.confirmed_entities) + len(self.missing_entities)
        self.coverage_ratio = len(self.confirmed_entities) / total if total else 0.0

    def to_dict(self) -> dict:
        return {
            "confirmed_entities": sorted(self.confirmed_entities),
            "missing_entities": sorted(self.missing_entities),
            "coverage_ratio": round(self.coverage_ratio, 4),
            "query_entities": sorted(self.query_entities),
            "entity_map": self.entity_map,
        }

NOISY_FEEDBACK_TERMS = {
    # Structural/formatting noise ? generic across all domains
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
    "frameworks", "suggested", "then", "year", "other", "also", "including",
    "include", "includes", "essential", "play", "playing", "described",
    "happen", "happens", "khan", "role", "roles",
    # Keep "cell"/"cells" as they are too generic in biology to be useful feedback terms
    "cell", "cells",
    # Domain-specific terms removed: tumor/cancer/lung/microbe/virus/bacteriophage
    # were originally in this list due to the cancer-lab corpus but bias the feedback
    # loop against non-cancer domains ? they belong in BROAD_DOMAIN_ANCHORS instead
}
GENERIC_REFINEMENT_TERMS = {
    "biological", "complex", "energy", "environment", "factor", "factors", "growth",
    "health", "metabolic", "process", "processes", "pathway", "pathways", "select",
    "system", "systems", "tissue",
}

SEARCH_TASK_TERMS = {
    "across", "advice", "again", "answer", "ask", "biomedical", "careful", "cautious",
    "all", "clinical", "code", "control", "conversation", "conversations", "correction",
    "caveat", "concise", "data", "diagnostic", "disappear", "evaluation", "evaluator", "framing", "full", "give", "language",
    "general to particular", "general-to-particular", "interested", "material",
    "mechanistic", "model", "multi turn", "multi-turn", "must", "only", "paragraph", "question", "reward", "scope",
    "search", "sentence", "sentences", "source", "sources", "strategy", "supplementary",
    "task", "trace", "turn", "two", "use", "user", "available", "more",
}

BIOMEDICAL_ACRONYM_TERMS = {
    "caf", "cafs", "ecm", "hgf", "met", "c-met", "cmet", "tme", "tam", "tams",
    "mdsc", "mdscs", "treg", "tregs", "nsclc", "sclc", "pd-1", "pd1", "pd-l1",
    "pdl1", "emt", "lox", "hif", "vegf", "il-6", "tnf", "ifn",
}

BIOMEDICAL_ANCHOR_EXPANSIONS = {
    "tme": ["tumor microenvironment"],
    "caf": ["cancer associated fibroblast", "cancer associated fibroblasts"],
    "ecm": ["extracellular matrix"],
    "nsclc": ["non-small cell lung cancer"],
    "pd1": ["pd-1", "programmed death 1"],
    "pdl1": ["pd-l1", "programmed death ligand 1"],
}

# Coupling point: the anchor-filtering logic is domain-general, but these
# vocabularies are currently calibrated for biomedical/cancer retrieval. When
# applying the same retrieval architecture to another domain (climate, legal,
# economics, etc.), expand these sets with that domain's broad umbrella terms
# and process/outcome terms so they do not satisfy entity-anchor coverage by
# themselves. This is intentional configuration coupling, not a bug.
BROAD_DOMAIN_ANCHORS = {
    "biomedical", "cancer", "carcinoma", "disease", "evidence", "mechanism",
    "microenvironment", "oncogenesis", "tumor", "tumour", "tumor microenvironment", "tme",
}

PROCESS_ANCHORS = {
    "association", "development", "evidence", "immune", "inflammation", "mechanism",
    "metabolism", "oncogenesis", "pathogenesis", "relationship", "signaling",
    "tumor development", "tumorigenesis",
}

ANCHOR_ALIASES = {
    "fungi": {"fungi", "fungal", "fungus", "mycobiome", "mycobiota", "candida", "malassezia", "aspergillus"},
    "fungal": {"fungi", "fungal", "fungus", "mycobiome", "mycobiota", "candida", "malassezia", "aspergillus"},
    "tumorigenesis": {"tumorigenesis", "tumor development", "cancer development", "carcinogenesis", "oncogenesis"},
    "candida": {"candida", "candida albicans", "c. albicans"},
    "antifungal": {"antifungal", "anti fungal", "antimycotic"},
    "mycobiome": {"mycobiome", "mycobiota", "fungi", "fungal"},
    # Cancer-domain anchors: enable cancer synonyms to match as a second specific anchor
    # in multi-concept queries (e.g. "fungi + cancer"), allowing results that cover the
    # cancer side of the relationship even if the fungi side is absent.
    "cancer": {"cancer", "carcinoma", "malignancy", "neoplasm", "tumor", "tumour",
               "oncology", "malignant", "oncogenesis"},
    "tumor": {"tumor", "tumour", "neoplasm", "malignancy", "cancer", "carcinoma"},
    "lung": {"lung", "pulmonary", "bronchial", "alveolar", "respiratory"},
    "microbiome": {"microbiome", "microbiota", "microbiomics", "gut microbiome", "gut microbiota"},
    "bacteria": {"bacteria", "bacterial", "bacterium", "microorganism", "microbial"},
    "inflammation": {"inflammation", "inflammatory", "inflammasome", "cytokine", "immune"},
    "fibroblast": {"fibroblast", "fibroblasts", "stromal cell", "stromal cells"},
}

MATH_PHARM_SYNERGY_TERMS = {

    "combination index", "ci value", "chou talalay", "dose response", "drug synergy",
    "therapeutic agent", "therapeutic agents", "combination therapy", "cytotoxicity",
    "ic50", "ctcae", "adverse event", "irae", "toxicity", "pharmacological",
}


def _contains_any(text: str, values: set[str] | list[str] | tuple[str, ...]) -> bool:
    lowered = (text or "").lower()
    return any(value in lowered for value in values)


def _canonical_search_anchor(term: str) -> str:
    cleaned = re.sub(r"[-_/]+", " ", str(term or "").strip().lower())
    cleaned = re.sub(r"\bonly\b", " ", cleaned)
    cleaned = " ".join(cleaned.split())
    if cleaned in {"tumorigenesi", "tumorgenesi"}:
        return "tumorigenesis"
    if cleaned in BIOMEDICAL_ACRONYM_TERMS:
        return cleaned
    normalized = normalize_idea(cleaned)
    if normalized in {"tumorigenesi", "tumorgenesi"}:
        return "tumorigenesis"
    if normalized in BIOMEDICAL_ACRONYM_TERMS:
        return normalized
    result = normalized or cleaned
    # Fuzzy typo correction: if the canonical form is not a known biomedical
    # entity but looks entity-like (>=5 chars), try to match it against the
    # ANCHOR_ALIASES vocabulary.  A SequenceMatcher ratio >= 0.82 catches
    # single-character substitutions (concer->cancer, tumer->tumor) while
    # avoiding false corrections for normal short English words.
    if (
        len(result) >= 5
        and result not in ANCHOR_ALIASES
        and result not in BIOMEDICAL_ACRONYM_TERMS
        and result not in PUZZLE_NODE_STOP_TERMS
    ):
        import difflib
        vocab = list(ANCHOR_ALIASES.keys())
        matches = difflib.get_close_matches(result, vocab, n=1, cutoff=0.82)
        if matches:
            result = matches[0]
    return result


def _is_task_or_style_term(term: str) -> bool:
    cleaned = " ".join(str(term or "").lower().replace("-", " ").split())
    canonical = _canonical_search_anchor(term)
    if canonical in BIOMEDICAL_ACRONYM_TERMS:
        return False
    return cleaned in SEARCH_TASK_TERMS or canonical in SEARCH_TASK_TERMS


def _expand_biomedical_anchors(anchors: list[str], limit: int = 18) -> list[str]:
    expanded: list[str] = []
    for anchor in anchors:
        canonical = _canonical_search_anchor(anchor)
        expanded.append(canonical if canonical else anchor)
        expanded.extend(BIOMEDICAL_ANCHOR_EXPANSIONS.get(canonical, [])[:2])
    return list(dict.fromkeys(item for item in expanded if item))[:limit]


def _has_specific_search_anchor(anchors: set[str] | list[str]) -> bool:
    canonical = {_canonical_search_anchor(anchor) for anchor in anchors if anchor}
    canonical = {anchor for anchor in canonical if anchor}
    return bool(canonical - BROAD_DOMAIN_ANCHORS)


def _specific_required_anchors(anchors: set[str] | list[str]) -> list[str]:
    """Return entity anchors specific enough to require in retrieved snippets.

    Process/outcome terms such as tumorigenesis, inflammation, mechanism, or
    cancer development describe the relation being sought; they are not enough
    by themselves to license a hit when the user asked about a named entity.
    This keeps retrieval general while preventing hits that satisfy only the
    process side of a query from polluting feedback and puzzle state.
    """
    out: list[str] = []
    for anchor in anchors:
        canonical = _canonical_search_anchor(anchor)
        if not canonical or len(canonical) < 2:
            continue
        if canonical in BIOMEDICAL_ACRONYM_TERMS:
            out.append(canonical)
            continue
        if canonical in BROAD_DOMAIN_ANCHORS or canonical in PROCESS_ANCHORS:
            continue
        if canonical in ANCHOR_ALIASES:
            out.append(canonical)
    return list(dict.fromkeys(out))


def _anchor_aliases(anchor: str) -> set[str]:
    canonical = _canonical_search_anchor(anchor)
    values = {canonical}
    values.update(ANCHOR_ALIASES.get(canonical, set()))
    values.update(synonyms_for(canonical))
    return {value for value in values if value}


def _text_matches_anchor(text: str, anchor: str) -> bool:
    lowered = (text or "").lower()
    for alias in _anchor_aliases(anchor):
        if not alias:
            continue
        pattern = r"(?<![a-z0-9])" + re.escape(alias.lower()).replace(r"\ ", r"[-\s]+") + r"(?![a-z0-9])"
        if re.search(pattern, lowered):
            return True
    return False


def _hit_anchor_coverage(item: dict[str, Any], search_frame: dict[str, Any]) -> dict[str, Any]:
    anchors = _specific_required_anchors(search_frame.get("anchor_terms") or [])
    if not anchors:
        return {"required_anchors": [], "matched_anchors": [], "missing_anchors": [], "passes": True}
    text = " ".join(
        str(item.get(field) or "")
        for field in ("title", "text", "sentence_text", "abstract", "subject", "relation", "predicate", "object")
        if item.get(field)
    )
    annotations = item.get("annotations") or []
    if isinstance(annotations, list):
        text = f"{text} {' '.join(str(annotation) for annotation in annotations)}"
    matched = [anchor for anchor in anchors if _text_matches_anchor(text, anchor)]
    missing = [anchor for anchor in anchors if anchor not in set(matched)]
    # Partial-match gate: for multi-anchor queries allow one missing anchor so that
    # results covering one side of a relationship (e.g. the cancer side of a
    # fungi+cancer query) are not silently dropped.
    # N=1 ? still require the single anchor (unchanged).
    # N>=2 ? require max(1, N-1) matched (allow at most 1 miss).
    min_required = max(1, len(anchors) - 1) if len(anchors) >= 2 else len(anchors)
    return {
        "required_anchors": anchors,
        "matched_anchors": matched,
        "missing_anchors": missing,
        "passes": len(matched) >= min_required,
    }


def _constraint_excluded_anchors(message: str) -> set[str]:
    lowered = (message or "").lower()
    excluded: set[str] = set()
    if re.search(r"\bnot\s+(?:clinical\s+)?treatment\s+advice\b", lowered):
        excluded.update({"clinical", "treatment", "advice"})
    if re.search(r"\bnot\s+clinical\b", lowered):
        excluded.add("clinical")
    return excluded


def _query_anchor_terms(message: str, limit: int = 12) -> list[str]:
    anchors: list[str] = []
    excluded = _constraint_excluded_anchors(message)
    for term in important_terms(message, limit=limit * 2):
        cleaned = _canonical_search_anchor(term)
        if (
            not cleaned
            or cleaned in excluded
            or cleaned in GENERIC_REFINEMENT_TERMS
            or cleaned in AMBIGUITY_MARKERS
            or cleaned in PUZZLE_NODE_STOP_TERMS
            or _is_task_or_style_term(term)
        ):
            continue
        anchors.append(cleaned)
        if len(anchors) >= limit:
            break
    return _expand_biomedical_anchors(list(dict.fromkeys(anchors)), limit=limit)


def _task_bridge_terms(message: str) -> list[str]:
    intent = _intent_bucket(message)
    lowered = (message or "").lower()
    terms: list[str] = []
    if intent == "mechanism" or any(marker in lowered for marker in ("how", "happen", "why", "pathway", "mechanism")):
        terms.extend(["mechanism", "pathogenesis", "signaling", "inflammation", "immune", "metabolism"])
    if intent == "evidence" or any(marker in lowered for marker in ("described", "reported", "study", "paper", "evidence")):
        terms.extend(["evidence", "review", "study", "reported", "described"])
    if any(marker in lowered for marker in ("what", "which", "examples", "particular", "specific")):
        terms.extend(["examples", "types", "species", "organisms", "specific", "reported"])
    if intent == "compare":
        terms.extend(["comparison", "difference", "relationship"])
    return list(dict.fromkeys(terms))[:12]


def _domain_search_frame(message: str, notes: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    text = (message or "").lower()
    asks_drug_synergy = _contains_any(text, {"drug", "therapy", "therapeutic", "dose", "combination index", "ci value", "pharmacological"})
    synergy_context = "synergy" in text or "synerg" in text or "crosstalk" in text
    mechanistic_context = _contains_any(text, {"mechanistic", "mechanism", "pathway", "how", "why", "crosstalk", "interaction", "cooperative", "functional synergy", "pivot"})
    analogy_context = _contains_any(text, {"analogy", "inspired", "as a", "like", "relationship between", "relating"})

    anchors = _query_anchor_terms(message)
    task_terms = _task_bridge_terms(message)
    preferred: list[str] = []
    avoid: list[str] = []
    frame = "general_biomedical"
    if analogy_context:
        frame = "cross_domain_or_analogy"
    elif mechanistic_context:
        frame = "mechanism_or_pathway"
    elif _intent_bucket(message) in {"evidence", "question"}:
        frame = "evidence_question"

    if anchors:
        preferred.append(" ".join(list(dict.fromkeys(anchors + task_terms))[:14]))
    excluded_anchors = _constraint_excluded_anchors(message)
    normalized = list(dict.fromkeys(_canonical_search_anchor(item) for item in extract_ideas(message, limit=8) if _canonical_search_anchor(item)))
    normalized = [
        item
        for item in normalized
        if item not in excluded_anchors
        and item not in GENERIC_REFINEMENT_TERMS
        and item not in NOISY_FEEDBACK_TERMS
        and item not in PUZZLE_NODE_STOP_TERMS
        and not _is_task_or_style_term(item)
    ]
    normalized = _expand_biomedical_anchors(normalized, limit=12)
    if normalized:
        preferred.append(" ".join(list(dict.fromkeys(normalized + task_terms))[:14]))
    if mechanistic_context and anchors:
        preferred.append(" ".join(list(dict.fromkeys(anchors[:8] + ["mechanism", "evidence", "relationship"]))[:12]))
    if synergy_context and not asks_drug_synergy:
        avoid.extend(sorted(MATH_PHARM_SYNERGY_TERMS))

    return {
        "frame": frame,
        "preferred_queries": list(dict.fromkeys(item for item in preferred if item.strip()))[:6],
        "avoid_terms": list(dict.fromkeys(avoid))[:24],
        "anchor_terms": anchors[:12],
        "task_terms": task_terms[:8],
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
    terms = _query_anchor_terms(message, limit=18)
    ideas = [
        _canonical_search_anchor(item)
        for item in extract_ideas(message, limit=10)
        if _canonical_search_anchor(item)
        and _canonical_search_anchor(item) not in _constraint_excluded_anchors(message)
        and not _is_task_or_style_term(item)
    ]
    normalized = list(dict.fromkeys(normalize_idea(item) for item in ideas if normalize_idea(item)))
    normalized = [
        item
        for item in normalized
        if item not in GENERIC_REFINEMENT_TERMS
        and item not in AMBIGUITY_MARKERS
        and item not in PUZZLE_NODE_STOP_TERMS
        and item not in SEARCH_TASK_TERMS
    ]
    synonym_terms: list[str] = []
    for idea in normalized[:6]:
        synonym_terms.extend(synonyms_for(idea)[:3])
    synonym_terms = list(dict.fromkeys(synonym_terms))

    raw_terms = important_terms(message, limit=24)
    task_heavy_query = bool(raw_terms and terms and (len(terms) / max(1, len(raw_terms))) < 0.5)
    candidates: list[tuple[str, str, str, str]] = []
    if not task_heavy_query or not terms:
        candidates.append(("original", message, "narrow" if strategy == "narrow" else strategy, "deterministic"))
    if terms:
        candidates.append(("important_terms", " ".join(terms[:10]), strategy, "deterministic"))
    preferred_queries = list((search_frame or {}).get("preferred_queries", [])[:3])
    for preferred in preferred_queries:
        candidates.append(("domain_bridge", preferred, "wide" if strategy != "narrow" else "medium", "deterministic"))
    if synonym_terms:
        candidates.append(("biomedical_synonyms", " ".join(list(dict.fromkeys(normalized + synonym_terms))[:12]), "wide", "deterministic"))
    if terms and (_intent_bucket(message) in {"mechanism", "compare", "evidence"} or _query_ambiguity(message, 0) != "low"):
        relation_words = ["mechanism", "evidence", "relationship"] if _intent_bucket(message) == "mechanism" else ["evidence", "relationship"]
        candidates.append(("relation_probe", " ".join(list(dict.fromkeys(terms[:10] + relation_words))), "wide", "deterministic"))
    if normalized:
        candidates.append(("normalized_ideas", " ".join(normalized[:8]), "medium", "deterministic"))
    if strategy == "wide" and terms and normalized:
        candidates.append(("mixed_wide", " ".join(list(dict.fromkeys(normalized + terms[:10] + synonym_terms[:6]))), "wide", "deterministic"))

    return _dedupe_queries(candidates, max(1, max_variants))


FOLLOWUP_CONTEXT_MARKERS = {
    "as above", "candidate framework", "candidate frameworks", "continue",
    "earlier", "latest candidate", "previous", "start by", "suggested", "those",
    "use what is actually supported", "what is actually supported",
    "search more", "search again", "look further", "look more", "all available data",
    "all available sources", "available data sources", "use all your available",
    "answer again", "give me a one-paragraph version", "keep biomedical direction",
    "keep direction correct", "novice user", "one paragraph version",
    "one-paragraph version", "summarize that", "rewrite that",
    "concise answer", "essential caveat", "caveat must not disappear",
    "must not disappear", "what essential caveat", "keep the caveat",
    "do not omit the caveat", "don't omit the caveat",
    "can the chatbot phrase", "can i phrase", "could i phrase", "phrase the answer",
    "phrasing", "the statement", "this statement", "that statement",
}


def _is_followup_reference(message: str) -> bool:
    return _contains_any(message, FOLLOWUP_CONTEXT_MARKERS)


_CLARIF_REPLY_RE = re.compile(
    r"^[a-z](?:\s*(?:,\s*|and\s+)[a-z])*\.?\s*$",
    re.IGNORECASE,
)


def _is_clarification_reply(message: str) -> bool:
    """True when message is only option letters: 'a', 'b and c', 'a, b, c'."""
    stripped = message.strip()
    return len(stripped) <= 20 and bool(_CLARIF_REPLY_RE.match(stripped))


# Discourse/conversational replies that are never biomedical anchors.
# _query_anchor_terms returns ['yes'] for 'yes' and ['second'] for 'the second one',
# bypassing the short-message guard. This set short-circuits that path.
_CONTEXT_POOR_EXACT: frozenset[str] = frozenset({
    "yes", "no", "yeah", "nope", "yep", "sure", "ok", "okay", "alright",
    "i see", "got it", "understood", "noted", "right", "correct",
    "agreed", "exactly", "perfect", "great",
    # Ordinal / positional replies
    "the first one", "the second one", "the third one", "the fourth one",
    "the last one", "the first", "the second", "the third", "the fourth",
    "first one", "second one", "third one", "last one",
    "that one", "this one", "the other one", "that option", "that approach",
    # Collective replies
    "all of them", "all of those", "both of them", "all three",
    "none of them", "neither",
    # Intent-reference replies — 'meant' passes as anchor but is never biomedical
    "i meant that", "i meant this", "what i meant", "that is what i meant",
    "i mean that", "i meant the other", "i meant the first", "i meant the second",
    "that is what i asked", "that is what i was asking",
    "yes that", "yes this", "yes both", "yes all", "yes exactly",
    "i want both", "i want all", "i want the first", "i want the second",
})


def _is_context_poor(message: str) -> bool:
    """True when message is too short or vague to carry its own retrieval signal.
    Covers letter replies, 'the second one', 'yes', 'I meant that', etc.
    The model already sees conversation history and can reason correctly;
    the problem is only the BM25 search getting nonsense queries.
    """
    stripped = message.strip().lower().rstrip(" .,!?")
    if stripped in _CONTEXT_POOR_EXACT:
        return True
    # limit=2 → important_terms gets 4 candidates; needed so that medical proper
    # nouns ranked 3rd (e.g. 'Aspergillus' after 'how', 'does') still register.
    return _is_clarification_reply(message) or (
        not _query_anchor_terms(message, limit=2)
        and len(message.split()) <= 8
    )


def _prior_frame_compatible(message: str, prior_query: str) -> bool:
    if _is_followup_reference(message):
        return True
    current = set(_query_anchor_terms(message, limit=12))
    prior = set(_query_anchor_terms(prior_query, limit=12))
    if not current or not prior:
        return False
    current_specific = set(_specific_required_anchors(current)) or current
    prior_specific = set(_specific_required_anchors(prior)) or prior
    return bool(current_specific & prior_specific)


def _is_rewrite_or_diagnostic_followup(message: str) -> bool:
    return _contains_any(
        message,
        {
            "answer again",
            "give me a one-paragraph version",
            "novice user",
            "one paragraph version",
            "one-paragraph version",
            "concise answer",
            "essential caveat",
            "must not disappear",
            "summarize that",
            "rewrite that",
            "reward model",
            "evaluator",
            "trace evidence",
            "before changing code",
        },
    )


def _is_phrase_evaluation(message: str) -> bool:
    return _contains_any(
        message,
        {
            "can the chatbot phrase",
            "can i phrase",
            "could i phrase",
            "phrase the answer",
            "phrasing",
            "the statement",
            "this statement",
            "that statement",
        },
    )


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


def _note_from_conversation_frame(frame: dict[str, Any] | None) -> dict[str, Any] | None:
    if not frame:
        return None
    terms = [
        str(term)
        for term in (frame.get("active_terms") or [])
        if isinstance(term, str) and term.strip()
    ]
    anchors = _query_anchor_terms(" ".join(terms), limit=10)
    if not anchors:
        return None
    query = " ".join(anchors[:10])
    return {
        "note": "Recovered prior search frame from conversation memory.",
        "search_plan": {
            "variants": [
                {
                    "label": "prior_frame",
                    "query": query,
                    "strategy": "medium",
                    "source": "conversation_frame",
                }
            ]
        },
    }


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
    intent: str = "new_query",
    prior_frame_summary: str | None = None,
    llm_provider: str | None = None,
    llm_model: str | None = None,
    llm_api_format: str | None = None,
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
            "content": frame_system_prompt(intent, prior_frame_summary),
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
        provider=llm_provider,
        model=llm_model,
        api_format=llm_api_format,
        max_tokens=900,
        agent="frame",
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


async def resolve_message_intent(
    message: str,
    *,
    notes: list[dict[str, Any]],
    llm_provider: str | None = None,
    llm_model: str | None = None,
    llm_api_format: str | None = None,
) -> dict[str, str] | None:
    """Context-manager agent resolves the actual search intent of a context-poor message.

    The context_manager (nemotron-3-super-120b, reasoning=medium) inspects the
    message + conversation frame and returns one of three intents:
      prior_context  -- the message references prior options/context; reuse prior frame
      new_query      -- the message has its own intent; effective_query has the full text
      augment_prior  -- blend prior context with new information in effective_query
    Returns None when no prior context exists or the LLM call fails (caller uses heuristic).
    """
    active_terms: list[str] = []
    summary = ""
    recent_queries: list[str] = []
    recent_turns: list[str] = []
    for _n in (notes or [])[:8]:
        if not active_terms and _n.get("active_terms"):
            active_terms = list(_n["active_terms"])[:8]
        if not summary and _n.get("summary"):
            summary = str(_n["summary"])[:300]
        if _n.get("query"):
            recent_queries.append(str(_n["query"]))
        if not recent_turns and _n.get("recent_turns"):
            recent_turns = list(_n["recent_turns"])[:6]
    if not active_terms and not recent_queries and not recent_turns:
        return None
    _has_wbuf = bool(recent_turns)
    _has_frame = bool(active_terms or summary)
    _messages = [
        {
            "role": "system",
            "content": intent_resolution_system_prompt(_has_wbuf, _has_frame, active_terms),
        },
        {
            "role": "user",
            "content": (
                f"User message: {message!r}\n\n"
                f"Prior conversation summary: {summary or 'none'}\n"
                f"Active research terms: {active_terms}\n"
                f"Prior search queries (recent first): {recent_queries[:4]}\n"
                + (
                    "Recent conversation turns (working buffer):\n"
                    + "\n".join(recent_turns) + "\n"
                    if recent_turns else ""
                )
                + "\n"
                + 'Respond with JSON: {"intent": "prior_context|new_query|augment_prior", '
                '"effective_query": "<full biomedical query capturing actual intent>", '
                '"explanation": "<one sentence>"}'
            ),
        },
    ]
    try:
        _text = await LLMClient().chat_once(
            _messages,
            provider=llm_provider,
            model=llm_model,
            api_format=llm_api_format,
            max_tokens=250,
            agent="context_manager",
        )
        _data = _extract_json_object(_text)
        if not _data or "intent" not in _data:
            return None
        return {
            "intent": str(_data.get("intent", "prior_context")),
            "effective_query": str(_data.get("effective_query", "")),
            "explanation": str(_data.get("explanation", "")),
        }
    except Exception as _e:
        print(f"[WARN] resolve_message_intent failed: {_e}")
        return None


async def plan_auto_context(
    *,
    message: str,
    selected_context_count: int,
    notes: list[dict[str, Any]],
    action_value_hints: list[dict[str, Any]],
    max_variants: int,
    allow_llm_refine: bool,
    llm_provider: str | None = None,
    llm_model: str | None = None,
    llm_api_format: str | None = None,
) -> AutoContextPlan:
    state = search_state_key(message, selected_context_count=selected_context_count)
    strategy = _strategy_from_hints(action_value_hints, notes)
    search_frame = _domain_search_frame(message, notes)

    # ── Context-manager intent resolution ──────────────────────────────────
    # _is_context_poor() is a cheap gate: when the message is short/vague with
    # no biomedical anchors, the context_manager agent (120b, reasoning=medium)
    # decides what the actual search intent is, rather than a hardcoded rule.
    # The agent can return prior_context (use prior frame), new_query (different
    # intent), or augment_prior (blend). Falls back to heuristic on failure.
    _context_poor = _is_context_poor(message)
    _resolved: dict[str, str] | None = None
    _eff_msg = message
    if _context_poor and allow_llm_refine:
        # ── Tier-1 zero-shot intent router (P-7) ───────────────────────────
        # Cheap classifier (small NIM primary, HF MNLI fallback) decides intent
        # before the expensive 120b. A high-confidence prior_context needs NO
        # query rewrite — reuse the prior frame and skip the 120b entirely.
        # new_query/augment_prior (which need an effective_query rewrite) and
        # low-confidence cases escalate to tier-2 (resolve_message_intent).
        _router = await classify_intent_zeroshot(message, notes)
        if (
            _router
            and _router.get("intent") == "prior_context"
            and float(_router.get("confidence", 0.0)) >= ROUTER_CONF_THRESHOLD
        ):
            _resolved = {
                "intent": "prior_context",
                "effective_query": "",
                "explanation": (
                    f"tier1-router({_router.get('source')}) "
                    f"conf={float(_router.get('confidence', 0.0)):.2f}"
                ),
            }
        else:
            # ── Tier-2: 120b context_manager (rewrite / genuine ambiguity) ──
            _resolved = await resolve_message_intent(
                message,
                notes=notes,
                llm_provider=llm_provider,
                llm_model=llm_model,
                llm_api_format=llm_api_format,
            )
            if _resolved and _resolved.get("intent") in {"new_query", "augment_prior"} and _resolved.get("effective_query"):
                _eff_msg = _resolved["effective_query"]

    base = deterministic_query_variants(_eff_msg, strategy=strategy, max_variants=max_variants, search_frame=search_frame)
    prior: list[SearchQueryVariant] = []
    _skip_frame_refine = False

    if _resolved and _resolved.get("intent") == "prior_context":
        all_prior = list(_prior_frame_variants(notes))
        if all_prior:
            prior = all_prior
            base = _dedupe_queries(
                [(item.label, item.query, item.strategy, item.source) for item in prior],
                max_variants,
            )
        _skip_frame_refine = True
    elif _is_followup_reference(_eff_msg) or (_context_poor and not _resolved):
        # Heuristic fallback: LLM not invoked or call failed
        all_prior = list(_prior_frame_variants(notes))
        if _context_poor and not _resolved and all_prior:
            prior = all_prior
            base = _dedupe_queries(
                [(item.label, item.query, item.strategy, item.source) for item in prior],
                max_variants,
            )
            _skip_frame_refine = True
        else:
            prior = [
                item
                for item in all_prior
                if _prior_frame_compatible(_eff_msg, item.query)
            ]
            if prior:
                merged = [*prior, *base]
                base = _dedupe_queries(
                    [(item.label, item.query, item.strategy, item.source) for item in merged],
                    max_variants,
                )

    levels = deterministic_search_levels(_eff_msg, strategy=strategy)
    used_llm = bool(_resolved)
    planner_note = (_resolved or {}).get("explanation", "")
    variants = list(base)

    if allow_llm_refine and base and not _skip_frame_refine:
        try:
            _frame_intent = (_resolved or {}).get("intent", "new_query")
            _frame_prior = prior[0].query if prior else (
                _eff_msg[:200] if _frame_intent == "augment_prior" else None
            )
            llm_variants, planner_note = await llm_refine_variants(
                _eff_msg,
                base_variants=base,
                notes=notes,
                action_value_hints=action_value_hints,
                search_frame=search_frame,
                max_variants=max_variants,
                intent=_frame_intent,
                prior_frame_summary=_frame_prior,
                llm_provider=llm_provider,
                llm_model=llm_model,
                llm_api_format=llm_api_format,
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


DISEASE_TAG_TERMS = {
    "cancer", "tumor", "tumour", "carcinoma", "infection", "inflammation", "hypoxia",
    "diabetes", "obesity", "fibrosis", "asthma", "cardiovascular", "autoimmune",
}
MECHANISM_TAG_TERMS = {
    "activation", "signaling", "pathway", "inhibition", "suppression", "expression",
    "metabolism", "crosslinking", "remodeling", "stiffness", "barrier", "immune",
    "angiogenesis", "invasion", "migration", "resistance", "hypoxia", "cytokine",
}
EVIDENCE_TYPE_TAG_TERMS = {
    "review", "trial", "case", "cohort", "preclinical", "in vitro", "in vivo",
    "abstract", "introduction", "discussion", "results", "figure",
}


def _tag_terms(text: str, terms: set[str]) -> list[str]:
    lower = (text or "").lower()
    return sorted(term for term in terms if term in lower)[:12]


def _retrieval_tags(item: dict[str, Any]) -> dict[str, list[str]]:
    text = " ".join(
        str(item.get(field) or "")
        for field in ("title", "text", "sentence_text", "abstract", "section")
    )
    disease_tags = [str(v) for v in item.get("disease_terms") or item.get("disease_tags") or []]
    mechanism_tags = [str(v) for v in item.get("mechanism_tags") or []]
    evidence_type_tags = [str(v) for v in item.get("evidence_type_tags") or []]
    annotations = item.get("annotations") or []
    if isinstance(annotations, list):
        for annotation in annotations:
            parts = str(annotation).split("|")
            if len(parts) >= 3 and parts[2].lower() == "disease":
                disease_tags.append(parts[1])
    return {
        "disease_tags": list(dict.fromkeys(disease_tags + _tag_terms(text, DISEASE_TAG_TERMS)))[:12],
        "mechanism_tags": list(dict.fromkeys(mechanism_tags + _tag_terms(text, MECHANISM_TAG_TERMS)))[:12],
        "evidence_type_tags": list(dict.fromkeys(evidence_type_tags + _tag_terms(text, EVIDENCE_TYPE_TAG_TERMS)))[:12],
    }


_FEEDBACK_POS_STOP = {
    "meanwhile", "correlated", "correlates", "correlating", "suggesting",
    "indicated", "notably", "consistently", "respectively", "particularly",
    "consequently", "additionally", "furthermore", "moreover", "although",
    "whereas", "demonstrated", "associated", "mediated", "observed",
}


def _feedback_candidates(texts: list[str], limit: int) -> list[str]:
    ideas = extract_ideas(*texts, limit=limit)
    terms = important_terms(" ".join(texts), limit=limit * 2)
    text_term_sets = [set(important_terms(text, 96)) for text in texts]
    out: list[tuple[float, str]] = []
    for term in list(dict.fromkeys(ideas + terms)):
        cleaned = term.lower().strip()
        if cleaned in NOISY_FEEDBACK_TERMS or cleaned in PUZZLE_NODE_STOP_TERMS or cleaned in _FEEDBACK_POS_STOP:
            continue
        if re.fullmatch(r"[a-z]{7,}", cleaned) and not set(important_terms(cleaned, 4)) & set(important_terms(" ".join(texts), 96)):
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
    author_initial_count = len(re.findall(r"\b[A-Z][a-z]{2,}\s+[A-Z]\.", text or ""))
    citation_marker_count = len(re.findall(r"\b(?:doi|pmid|pharmaceutics|int\. j\.|j\.|vol\.|pp\.)\b|\b10\.\d{4,9}/", lowered))
    if author_initial_count >= 4 or (author_initial_count >= 2 and citation_marker_count >= 1):
        return True
    if citation_marker_count >= 2 and len(re.findall(r"\b[a-z]{6,}\b", lowered)) >= 8:
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
        _canonical_search_anchor(term)
        for term in important_terms(" ".join(anchor_queries or []), 96)
        if _canonical_search_anchor(term)
        and _canonical_search_anchor(term) not in GENERIC_REFINEMENT_TERMS
        and _canonical_search_anchor(term) not in AMBIGUITY_MARKERS
        and _canonical_search_anchor(term) not in PUZZLE_NODE_STOP_TERMS
        and not _is_task_or_style_term(term)
    }
    has_specific_anchor = _has_specific_search_anchor(anchor_terms)
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
    if anchor_terms and not has_specific_anchor:
        accepted_terms = [
            term
            for term in accepted_terms
            if _canonical_search_anchor(term) in anchor_terms
        ]
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
    anchor_terms = {
        term.lower()
        for term in (search_frame.get("anchor_terms") or [])
        if isinstance(term, str) and len(term) >= 3
    }
    focus_anchors = anchor_terms - {"synergy", "synergistic", "functional", "mechanistic", "effect", "effects"}
    item_terms = set(important_terms(text, 128))
    return not bool((focus_anchors or anchor_terms) & item_terms)


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
    # Original set
    "develop", "relating", "related", "relationship", "something", "else", "promotes",
    "promote", "environment", "question", "evidence", "mechanism", "pathway", "body",
    "latest", "start", "explain", "explaining", "conceptual", "then", "candidate",
    "framework", "frameworks", "suggested", "described", "playing", "role",
    "roles", "essential", "how", "happen", "happens", "all", "available",
    "data", "source", "sources", "search", "more", "supplementary", "material",
    # Quantifiers / modal adjectives that appear in user queries but are not entities
    "possible", "likely", "various", "certain", "multiple", "specific", "known",
    "given", "new", "old", "big", "small", "high", "low", "more", "less",
    "any", "some", "each", "both", "other", "such", "one", "two", "three",
    # Common verbs / gerunds
    "involving", "involve", "involves", "include", "including", "using", "use", "used",
    "based", "show", "shows", "shown", "found", "suggest", "suggests", "affect",
    "does", "play", "make", "have", "been", "are", "was", "will", "can", "may",
    "shall",
    # Prepositions / logical connectors
    "between", "through", "without", "within", "during", "before", "after",
    "under", "over", "around", "another", "among", "across", "about", "toward",
    "because", "therefore", "however", "although", "that", "with", "from", "this",
    "what", "why", "when", "where", "which", "who", "the", "and", "for",
    "its", "not", "but", "also",
    # Generic nouns that aren't biomedical entities
    "life", "style", "type", "form", "part", "way", "time", "case", "level",
    "effect", "function", "process", "activity", "response", "outcome",
    "interested", "general to particular", "general-to-particular", "particular",
    "ph",  # pH handled separately via regex; raw "ph" is not a searchable anchor
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
    anchor_rejected_count = sum(int(report.get("anchor_mismatch_result_count", 0) or 0) for report in level_reports)
    drift_penalty = (ungrounded_count + anchor_rejected_count) / max(1, accepted_count + ungrounded_count + anchor_rejected_count)
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
        key = _canonical_search_anchor(cleaned) or cleaned.lower().rstrip("s")
        is_explicit_ph = key == "ph" and bool(re.search(r"\bp\s*h\b|\bph\b", cleaned, flags=re.IGNORECASE))
        if (
            not cleaned
            or ((cleaned.lower() in PUZZLE_NODE_STOP_TERMS or key in PUZZLE_NODE_STOP_TERMS) and not is_explicit_ph)
            or _is_task_or_style_term(cleaned)
            or key in node_keys
        ):
            continue
        node_keys.add(key)
        query_nodes.append(key)
        if len(query_nodes) >= 10:
            break

    def node_matches_text(node: str, text: str, item_terms: set[str]) -> bool:
        if node.lower() == "ph":
            return bool(re.search(r"\bp\s*h\b|\bph\b", text, flags=re.IGNORECASE))
        return bool(set(important_terms(node, 8)) & item_terms)

    covered_nodes = [node for node in query_nodes if node_matches_text(node, evidence_text, evidence_terms)]
    missing_nodes = [node for node in query_nodes if node not in set(covered_nodes)]
    task_terms = set(plan.search_frame.get("task_terms") or [])
    mechanistic_task_terms = {
        "mechanism", "mechanistic", "pathogenesis", "signaling", "inflammation",
        "immune", "metabolism", "pathway", "pathways", "via", "through",
    }
    requires_mechanistic_evidence = _intent_bucket(message) == "mechanism" or bool(task_terms & mechanistic_task_terms)
    mechanistic_terms_present = bool(evidence_terms & mechanistic_task_terms)
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
    if requires_mechanistic_evidence and not mechanistic_terms_present and edge_status == "supported":
        edge_status = "partial"
    node_coverage = len(covered_nodes) / max(1, len(query_nodes))
    if query_nodes and node_coverage < 0.35:
        assembly_quality = round(min(assembly_quality, 0.45 + (0.25 * node_coverage)), 4)
    if edge_status == "missing":
        assembly_quality = round(min(assembly_quality, 0.55), 4)
    elif edge_status == "partial":
        assembly_quality = round(min(assembly_quality, 0.78), 4)
    underspecified_umbrella_frame = (
        bool(query_nodes)
        and not _has_specific_search_anchor(query_nodes)
        and edge_status == "missing"
    )
    suppress_clarification_hold = _is_rewrite_or_diagnostic_followup(message) or _is_phrase_evaluation(message)
    clarification_needed = (
        not suppress_clarification_hold
        and (
            underspecified_umbrella_frame
            or ambiguity == "high"
            or (ambiguity == "medium" and edge_status == "missing" and bool(missing_nodes))
            or (bool(prior_frame_queries) and _is_followup_reference(message) and edge_status != "supported")
        )
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
            "Ask only this opening clarification; do not repeat it later as A/B/C or numbered frame choices. "
            "If the retrieved evidence only covers an umbrella domain term, do not choose an incidental named mechanism from the snippets; "
            "ask for the concrete mechanism, disease context, entity, or evidence frame to prioritize.\n"
        )
    phrase_eval_line = ""
    if _is_phrase_evaluation(message):
        phrase_eval_line = (
            "- The user is asking whether a proposed phrasing/statement is acceptable. "
            "Judge the proposed wording first as supported, unsupported, contradicted, or too broad; do not answer an earlier content request instead.\n"
        )
    prompt_context = (
        "Auto-context evidence assembly:\n"
        f"- Information need: {_intent_bucket(message)}.\n"
        f"- User query ambiguity: {ambiguity}.\n"
        f"- Follow-up prior frame active: {'yes' if prior_frame_queries else 'no'}.\n"
        f"- Retrieved evidence levels: {', '.join(level for level, count in result_levels.items() if count) or 'none'}.\n"
        f"- Candidate evidence frames: {', '.join(str(frame.get('label')) for frame in frame_summaries[:4]) or 'literal'}.\n"
        f"{clarification_line}"
        f"{phrase_eval_line}"
        "- Treat retrieved snippets as evidence pieces, not a completed causal chain. "
        "Do not assert a bridge between pieces unless the supplied context supports that bridge. "
        "Absence of a relation from the current snippets is not evidence that the relation has no plausible connection; "
        "for quoted negative/exclusion claims, answer that the supplied context is insufficient unless evidence directly supports the exclusion. "
        "Do not fill a missing edge with a new example or mediator that is absent from the supplied snippets. "
        "Do not name absent example candidates merely to illustrate a missing edge. "
        "Do not name a candidate therapy, agent, framework, pathway, or experiment unless the supplied context supports that named candidate. "
        "Do not use outside-field knowledge to expand an unsupported section; only say the bridge is not present in the supplied context. "
        "Avoid phrases such as 'known', 'plausible', 'implies', 'suggests', or 'likely' for a relation unless a cited snippet directly supports that exact relation. "
        "If a cited snippet supports only one part of a chain, keep the unsupported part separate and do not convert it into a mechanistic claim. "
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
            "anchor_mismatch_result_count": anchor_rejected_count,
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




def _snippet_utility(
    snippet: dict,
    query_anchors: set,
    gap_spec: "GapSpec | None" = None,
) -> float:
    """Score a retrieved snippet for context-slot utility (WP-C, SEAL-RAG).

    Components:
      0.35 * bm25_norm    ? normalised retrieval score (engines return ~0-10)
      0.35 * anchor_ratio ? fraction of query anchors covered by snippet text
      0.30 * gap_closing  ? fraction of missing_entities mentioned in snippet
    """
    score_raw = float(snippet.get("retrieval_score") or snippet.get("score") or 0.5)
    bm25_norm = min(score_raw / 10.0, 1.0)

    text_lower = (snippet.get("text") or snippet.get("snippet") or "").lower()
    if query_anchors:
        anchor_ratio = sum(1 for a in query_anchors if a in text_lower) / len(query_anchors)
    else:
        anchor_ratio = 0.0

    missing = getattr(gap_spec, "missing_entities", set()) if gap_spec is not None else set()
    if missing:
        gap_closing = sum(1 for e in missing if e.lower() in text_lower) / len(missing)
    else:
        gap_closing = 0.0

    return 0.35 * bm25_norm + 0.35 * anchor_ratio + 0.30 * gap_closing



async def llm_ground_entities(
    message: str,
    *,
    snippets: list[dict],
    gap_spec,
    llm_provider=None,
    llm_model=None,
    llm_api_format=None,
) -> None:
    """NER-bootstrap: classify query entities against retrieved context via LLM.

    status values
    -------------
    confirmed : entity appears verbatim in the retrieved context.
    synonym   : semantically equivalent term present under a different surface form
                (e.g. query "lung cancer", context has "NSCLC").
    absent    : entity not represented in any retrieved source.

    Works with both GapSpec dataclass (build_auto_context) and dict (policy.plan).
    Enriches confirmed_entities with synonym-grounded forms and populates
    entity_map for downstream gap-steering and feedback-term filtering.
    """
    query_ents = list(dict.fromkeys(
        _query_anchor_terms(message, limit=14)
        + [_canonical_search_anchor(e) for e in extract_ideas(message, limit=8)
           if _canonical_search_anchor(e)
           and _canonical_search_anchor(e) not in PUZZLE_NODE_STOP_TERMS]
    ))[:16]
    if not query_ents or not snippets:
        return

    ctx_pool = []
    for item in snippets[:20]:
        for _etype, _eids in (item.get("pubtator_entities") or {}).items():
            ctx_pool.extend(f"{_etype}:{eid}" for eid in (_eids or []))
        text = " ".join(
            str(item.get(f) or "")
            for f in ("title", "sentence_text", "text", "snippet", "subject", "object")
            if item.get(f)
        )
        ctx_pool.extend(important_terms(text, 12))
    ctx_entities = list(dict.fromkeys(ctx_pool))[:40]
    if not ctx_entities:
        return

    _conf = getattr(gap_spec, "confirmed_entities", None)
    if _conf is None and isinstance(gap_spec, dict):
        _conf = gap_spec.get("confirmed_entities") or []
    _ner_confirmed_count = len(_conf or [])
    _ner_is_discovery = _ner_confirmed_count == 0
    messages = [
        {
            "role": "system",
            "content": ner_grounding_system_prompt(_ner_is_discovery, _ner_confirmed_count),
        },
        {
            "role": "user",
            "content": (
                f"User query: {message[:400]}\n"
                f"Query entities to classify: {query_ents}\n"
                f"Context entity pool (all retrieved sources): {ctx_entities[:30]}\n\n"
                'Return: {"grounded": [{"entity": str, "status": "confirmed|synonym|absent",'
                ' "context_match": str_or_null}]}'
            ),
        },
    ]
    try:
        raw = await LLMClient().chat_once(
            messages,
            provider=llm_provider,
            model=llm_model,
            api_format=llm_api_format,
            max_tokens=500,
            agent="ner_grounding",
        )
        data = _extract_json_object(raw) or {}
        _confirmed: set[str] = set()
        _absent: set[str] = set()
        _emap: dict = {}
        for rec in data.get("grounded") or []:
            ent = str(rec.get("entity") or "").strip().lower()
            status = str(rec.get("status") or "").lower()
            match_ = str(rec.get("context_match") or "").strip() or None
            if not ent or status not in {"confirmed", "synonym", "absent"}:
                continue
            _emap[ent] = {"status": status, "context_match": match_}
            (_confirmed if status in {"confirmed", "synonym"} else _absent).add(ent)

        if isinstance(gap_spec, GapSpec):
            gap_spec.query_entities.update(query_ents)
            gap_spec.entity_map.update(_emap)
            gap_spec.confirmed_entities.update(_confirmed)
            gap_spec.missing_entities.update(_absent - gap_spec.confirmed_entities)
            gap_spec.missing_entities -= gap_spec.confirmed_entities
            gap_spec.update_coverage()
        else:
            _prev = set(gap_spec.get("confirmed_entities") or [])
            _prev.update(_confirmed)
            gap_spec["confirmed_entities"] = sorted(_prev)
            _miss = set(gap_spec.get("missing_entities") or []) | _absent
            _miss -= _prev
            gap_spec["missing_entities"] = sorted(_miss)
            gap_spec["query_entities"] = sorted(
                set(gap_spec.get("query_entities") or []) | set(query_ents)
            )
            gap_spec.setdefault("entity_map", {}).update(_emap)
    except Exception:
        pass  # non-fatal: gap_spec retains its anchor-based state


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
    llm_provider: str | None = None,
    llm_model: str | None = None,
    llm_api_format: str | None = None,
    # Modules 3/4: optional run identifier for eval-level vocab tracking
    run_id: str | None = None,
) -> dict[str, Any]:
    max_variants = max(1, int(settings.memory.auto_context_query_variants))
    k_total = max(1, int(settings.memory.auto_context_k))
    notes = await store.search_policy_notes(session_id=session_id, limit=4)
    # Inject conversation_frame into notes both for followup references AND
    # context-poor messages (short/vague, no biomedical anchors) so that
    # resolve_message_intent() has the current topic's active_terms to reason from.
    if _is_followup_reference(message) or _is_context_poor(message):
        try:
            frame_note = _note_from_conversation_frame(await store.conversation_frame(session_id))
        except Exception:
            frame_note = None
        if frame_note:
            notes = [frame_note, *notes]

    # Pass recent turns to resolve_message_intent() so context_manager sees
    # the actual model output (e.g. 'a) ... b) ...') before the vague reply.
    if _is_context_poor(message):
        try:
            _recent = await store.recent_messages(session_id, 3, token_budget=4000)
            if _recent:
                _turns_text = [
                    f"[Turn {r.get('turn_index', i + 1)}] "
                    f"{r.get('role', '?')}: "
                    f"{str(r.get('content', ''))[:300]}"
                    for i, r in enumerate(_recent[-6:])
                ]
                notes = [
                    *notes,
                    {
                        "note": "Recent conversation turns (working buffer)",
                        "recent_turns": _turns_text,
                    },
                ]
        except Exception:
            pass

    state = search_state_key(message, selected_context_count=selected_context_count)
    action_value_hints = await store.action_values(session_id=session_id, state_key=state, limit=4)
    plan = await plan_auto_context(
        message=message,
        selected_context_count=selected_context_count,
        notes=notes,
        action_value_hints=action_value_hints,
        max_variants=max_variants,
        allow_llm_refine=bool(settings.memory.auto_context_llm_refine),
        llm_provider=llm_provider,
        llm_model=llm_model,
        llm_api_format=llm_api_format,
    )

    seen: set[tuple[Any, Any, str]] = set()
    snippets: list[dict[str, Any]] = []
    level_reports: list[dict[str, Any]] = []
    feedback_terms: list[str] = []
    skipped_off_topic = 0
    # WP-B: GapSpec accumulation across levels
    gap_spec = GapSpec()
    gap_spec_before = GapSpec()
    previous_query_terms: list[set] = []
    step_rewards_collected: list[dict] = []
    _query_anchors_set = set(_query_anchor_terms(message))
    level_budgets = _level_budgets(plan.levels, k_total)

    # Module 2 Point A: inject high-utility session vocabulary terms into anchor_terms
    if VocabularyStore.enabled() and session_id:
        _vs_inject = VocabularyStore()
        _session_terms = _vs_inject.session_top_terms(session_id, limit=20)
        _existing_anchors = set(plan.search_frame.get("anchor_terms") or [])
        _new_anchors = [
            t for t, _ in _session_terms if t not in _existing_anchors
        ][:max(0, 16 - len(_existing_anchors))]
        if _new_anchors:
            plan.search_frame["anchor_terms"] = list(_existing_anchors) + _new_anchors
            plan.search_frame["session_vocab_injected"] = _new_anchors

    if multilevel_search_fn is None and search_fn is not None:
        async def compat_level_search(tenant_arg: str, level: str, query: str, filters: dict[str, Any], k: int) -> list[dict[str, Any]]:
            return await search_fn(tenant_arg, query, filters, k)
        multilevel_search_fn = compat_level_search
    if multilevel_search_fn is None:
        multilevel_search_fn = hybrid_search_multilevel

    for level_index, level in enumerate(plan.levels):
        gap_spec_before = copy.copy(gap_spec)  # snapshot for per-step reward
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
                anchor_coverage = _hit_anchor_coverage(item, plan.search_frame)
                item["anchor_coverage"] = anchor_coverage
                if not anchor_coverage["passes"]:
                    skipped_off_topic += 1
                    level_added.append(
                        {
                            "_rejected_anchor_mismatch": True,
                            "anchor_coverage": anchor_coverage,
                            "title": item.get("title"),
                            "text": item.get("text") or item.get("sentence_text"),
                        }
                    )
                    continue
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
                item["retrieval_rank"] = rank
                item["retrieval_score"] = float(item.get("score", item.get("_score", 0.0)) or 0.0)
                item["bm25_score"] = item["retrieval_score"]
                item["source_sentence_id"] = item.get("sent_id") or item.get("sentence_id")
                item.update(_retrieval_tags(item))
                item["feedback_terms_used"] = feedback_terms[:8]
                # WP-B: accumulate GapSpec from this accepted hit
                for _anc in anchor_coverage.get("matched_anchors") or []:
                    gap_spec.confirmed_entities.add(_anc.lower())
                    gap_spec.missing_entities.discard(_anc.lower())
                for _anc in anchor_coverage.get("missing_anchors") or []:
                    if _anc.lower() not in gap_spec.confirmed_entities:
                        gap_spec.missing_entities.add(_anc.lower())
                # absorb PubTator entities if present
                for _etype, _eids in (item.get("pubtator_entities") or {}).items():
                    for _eid in _eids:
                        gap_spec.confirmed_entities.add(f"{_etype}:{_eid}")
                gap_spec.update_coverage()
                snippets.append(item)
                level_added.append(item)
                if len(level_added) >= level_budget or len(snippets) >= k_total:
                    break
            if len(level_added) >= level_budget or len(snippets) >= k_total:
                break
        feedback_report = _feedback_term_report(
            [item for item in level_added if not item.get("_rejected_anchor_mismatch")],
            anchor_queries=[message, *[item.query for item in plan.variants], *feedback_terms],
        )
        anchor_mismatch_count = sum(1 for item in level_added if item.get("_rejected_anchor_mismatch"))
        accepted_level_added = [item for item in level_added if not item.get("_rejected_anchor_mismatch")]
        new_terms = feedback_report["accepted_terms"]
        feedback_terms = list(dict.fromkeys(feedback_terms + new_terms))[:12]
        level_reports.append(
            {
                "level": level,
                "query_count": len(level_queries),
                "queries": level_queries[:6],
                "result_count": len(accepted_level_added),
                "feedback_terms_added": new_terms[:8],
                "rejected_feedback_terms": feedback_report["rejected_terms"][:8],
                "rejected_feedback_result_count": feedback_report["rejected_result_count"],
                "ungrounded_feedback_term_count": feedback_report["ungrounded_feedback_term_count"],
                "anchor_mismatch_result_count": anchor_mismatch_count,
                "feedback_terms_after": feedback_terms[:8],
            }
        )
        # WP-D: compute per-step rewards and record
        _level_query_terms = set()
        for _q in level_queries:
            _level_query_terms |= set(important_terms(_q, 16))
        _step_reward = {
            "level": level,
            "query_novelty": query_novelty(_level_query_terms, previous_query_terms),
            "gap_closure_score": gap_closure_score(gap_spec_before, gap_spec),
            "distractor_ratio": distractor_ratio(
                accepted_level_added, _query_anchors_set, gap_spec
            ),
        }
        step_rewards_collected.append(_step_reward)
        # WP-D: backfill per-step reward metrics into corresponding level_report
        level_reports[-1].update({
            "gap_closure_score": _step_reward["gap_closure_score"],
            "distractor_ratio": _step_reward["distractor_ratio"],
            "query_novelty": _step_reward["query_novelty"],
        })
        # Module 3: per-term reward credit (side-channel from existing reward computation)
        if VocabularyStore.enabled() and session_id:
            _vs_reward = VocabularyStore()
            _gc = _step_reward["gap_closure_score"]
            _dr = _step_reward["distractor_ratio"]
            for _term in (plan.search_frame.get("anchor_terms") or []):
                _canon = _canonical_search_anchor(_term)
                if _canon in ANCHOR_ALIASES or _canon in BIOMEDICAL_ACRONYM_TERMS:
                    _vs_reward.record_outcome(
                        _term, "session", session_id,
                        gap_delta=_gc, distractor_pen=_dr, base_reward=0.5,
                    )
                    if run_id:
                        _vs_reward.record_outcome(
                            _term, "run", run_id,
                            gap_delta=_gc, distractor_pen=_dr, base_reward=0.5,
                        )
        previous_query_terms.append(_level_query_terms)

    # WP-C: replace semantics ? keep top-k by utility (SEAL-RAG style)
    _max_ctx = int(getattr(settings.memory, "max_context_snippets", 12))
    if len(snippets) > _max_ctx:
        snippets.sort(
            key=lambda _s: _snippet_utility(_s, _query_anchors_set, gap_spec),
            reverse=True,
        )
        snippets = snippets[:_max_ctx]

    # NER bootstrap: LLM entity grounding over all locally retrieved snippets
    if getattr(settings.memory, "entity_grounding_enabled", True) and snippets:
        await llm_ground_entities(
            message,
            snippets=snippets,
            gap_spec=gap_spec,
            llm_provider=llm_provider,
            llm_model=llm_model,
            llm_api_format=llm_api_format,
        )

    payload = plan.to_dict()
    retrieval_records = [
        {
            "rank": idx + 1,
            "level": item.get("search_level"),
            "level_index": item.get("search_level_index"),
            "level_rank": item.get("search_rank"),
            "bm25_score": item.get("bm25_score"),
            "retrieval_score": item.get("retrieval_score"),
            "query": item.get("auto_query"),
            "query_label": item.get("auto_query_label"),
            "query_source": item.get("auto_query_source"),
            "frame_id": item.get("auto_frame_id"),
            "frame_label": item.get("auto_frame_label"),
            "source_sentence_id": item.get("source_sentence_id"),
            "paper_id": item.get("paper_id"),
            "pmid": item.get("pmid"),
            "pmcid": item.get("pmcid"),
            "disease_tags": item.get("disease_tags", []),
            "mechanism_tags": item.get("mechanism_tags", []),
            "evidence_type_tags": item.get("evidence_type_tags", []),
        }
        for idx, item in enumerate(snippets)
    ]
    payload.update(
        {
            "result_count": len(snippets),
            "note": _result_note(plan=plan, result_count=len(snippets)),
            "query_labels": [item.label for item in plan.variants],
            "candidate_frames": plan.candidate_frames,
            "levels": plan.levels,
            "level_reports": level_reports,
            "feedback_terms": feedback_terms[:12],
            "retrieval_records": retrieval_records,
            "skipped_off_topic_count": skipped_off_topic,
            "gap_spec": gap_spec.to_dict(),
            "step_rewards": step_rewards_collected,
            "evidence_assembly": _evidence_assembly(
                message=message,
                plan=plan,
                snippets=snippets,
                level_reports=level_reports,
            ),
        }
    )
    # Module 2 Point B: persist this turn's GapSpec entities to session vocab
    if VocabularyStore.enabled() and session_id:
        _vs_persist = VocabularyStore()
        for _ent in gap_spec.confirmed_entities:
            _vs_persist.record_outcome(
                _ent, "session", session_id,
                gap_delta=gap_spec.coverage_ratio,
                distractor_pen=0.0,
                base_reward=0.5,
            )
        for _ent in gap_spec.missing_entities:
            _vs_persist.record_outcome(
                _ent, "session", session_id,
                gap_delta=0.0,
                distractor_pen=0.2,
                base_reward=0.5,
            )
        _vs_persist.expire_session(session_id)
    return {"snippets": snippets, "plan": payload}
