from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List

from app.clients.llm import LLMClient
from app.config import settings
from app.memory.action_value import action_key, state_key
from app.memory.claim_support import assess_claim_support, build_evidence_table, evidence_table_debug_payload
from app.memory.claims import extract_atomic_claims
from app.memory.consistency import longitudinal_consistency_report, render_conversation_frame
from app.memory.evidence import evidence_to_dicts, gather_evidence_candidates
from app.memory.nli import score_answer_triples
from app.memory.idea_index import extract_ideas, normalize_idea
from app.memory.rewards import detect_triplet_conflicts, important_terms, reward_report, terms
from app.memory.store import MemoryStore
from app.prompts.agent_prompts import (
    external_query_system_prompt,
    reflection_system_prompt,
)
from app.memory.search_agent import _is_context_poor, llm_ground_entities
from app.memory.web_search import (
    duckduckgo_search,
    litsense2_search,
    pmc_relevant_sentence_search,
    pubmed_fetch_by_pmids,
    pubmed_pmc_search,
    pubtator3_search,
)

try:
    from app.integrations.extraction_client import extract_triples
except Exception:
    extract_triples = None

try:
    from app.triplets.search import search_triplets
except Exception:
    search_triplets = None


def _env_int(name: str, default: int) -> int:
    try:
        return max(1, int(os.getenv(name, str(default))))
    except (TypeError, ValueError):
        return default


# ── Configurable caps (no hardcoding — ARCHITECTURE.md rule 13) ──────────────
# G6: per-call LLM output budgets + claim-verification fan-out.
EXTERNAL_QUERY_MAX_TOKENS = _env_int("EXTERNAL_QUERY_MAX_TOKENS", 700)   # _llm_external_query_variants
REFLECT_MAX_TOKENS = _env_int("REFLECT_MAX_TOKENS", 160)                 # _reflect
MAX_NLI_PAIRS_PER_CLAIM = _env_int("MAX_NLI_PAIRS_PER_CLAIM", 8)         # assess_claim_support fan-out
# G5: memory-item fetch/render counts (how many of each memory level reaches the answer prompt).
MEMORY_SUMMARIES_FETCH = _env_int("MEMORY_SUMMARIES_FETCH", 3)
MEMORY_TRACES_FETCH = _env_int("MEMORY_TRACES_FETCH", 3)
MEMORY_IDEAS_FETCH = _env_int("MEMORY_IDEAS_FETCH", 8)
MEMORY_LANDMARKS_RENDER = _env_int("MEMORY_LANDMARKS_RENDER", 8)
MEMORY_IDEAS_RENDER = _env_int("MEMORY_IDEAS_RENDER", 8)
MEMORY_REFLECTIONS_RENDER = _env_int("MEMORY_REFLECTIONS_RENDER", 3)


@dataclass
class ContextPlan:
    turn_index: int
    context_prefix: str
    selected_context: List[Dict[str, Any]] = field(default_factory=list)
    retrieved_triplets: List[Dict[str, Any]] = field(default_factory=list)
    web_results: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)


def _render_recent(messages: list[dict]) -> str:
    if not messages:
        return ""
    lines = ["Recent conversation buffer:"]
    for m in messages:
        role = m.get("role", "message")
        text = (m.get("summary") or m.get("text") or "").strip()
        if text:
            lines.append(f"- {role}: {text[:900]}")
    return "\n".join(lines)


def _render_memory(hits: list[dict]) -> str:
    if not hits:
        return ""
    lines = ["Retrieved session memory:"]
    for h in hits:
        role = h.get("role", "memory")
        text = (h.get("summary") or h.get("text") or "").strip()
        score = h.get("_score")
        score_text = f" score={score:.3f}" if isinstance(score, (int, float)) else ""
        lines.append(f"- {role}{score_text}: {text[:700]}")
    return "\n".join(lines)


def _render_landmarks(items: list[dict]) -> str:
    if not items:
        return ""
    lines = ["Conversation landmarks:"]
    for item in items[:MEMORY_LANDMARKS_RENDER]:
        name = item.get("name", "landmark")
        summary = (item.get("summary") or "").strip()
        if summary:
            lines.append(f"- {name}: {summary[:500]}")
    return "\n".join(lines)


def _render_summaries(items: list[dict]) -> str:
    if not items:
        return ""
    lines = ["Episodic session summaries:"]
    for item in items[:4]:
        summary = (item.get("summary") or item.get("text") or "").strip()
        if summary:
            lines.append(f"- {summary[:700]}")
    return "\n".join(lines)


def _render_triplets(triplets: list[dict]) -> str:
    if not triplets:
        return ""
    lines = ["Relevant semantic triplets:"]
    for t in triplets[: settings.memory.triplet_k]:
        subj = t.get("subject") or ""
        rel = t.get("relation") or t.get("predicate") or ""
        obj = t.get("object") or ""
        sent = t.get("sentence_text") or t.get("text") or ""
        if subj or obj:
            lines.append(f"- ({subj}; {rel}; {obj}) {sent[:280]}")
    return "\n".join(lines)


def _render_web(results: list[dict]) -> str:
    if not results:
        return ""
    lines = ["Privacy-filtered external biomedical grounding:"]
    for r in results[: settings.memory.web_k]:
        source = r.get("source") or "web"
        title = r.get("title") or "web result"
        snippet = r.get("snippet") or ""
        url = r.get("url") or ""
        pmid = r.get("pmid") or ""
        pmcid = r.get("pmcid") or ""
        provenance = " ".join(item for item in (f"PMID {pmid}" if pmid else "", f"PMCID {pmcid}" if pmcid else "") if item)
        prefix = f"- {source}" + (f" | {provenance}" if provenance else "") + f" | {title}:"
        lines.append(f"{prefix} {snippet[:520]} {url}".strip())
    return "\n".join(lines)


def _external_result_key(result: dict) -> str:
    return str(result.get("pmid") or result.get("pmcid") or result.get("url") or result.get("title") or "")


def _compact_text(value: str, limit: int = 220) -> str:
    return " ".join(str(value or "").split())[:limit]


def _policy_query_terms(query: str, limit: int = 18) -> list[str]:
    noisy = {
        "what", "which", "how", "why", "does", "described", "describe", "playing",
        "role", "roles", "essential", "happen", "happens", "something", "else",
        "relationship", "relating", "related", "develop", "provide", "explain",
        "search", "more", "again", "all", "available", "data", "source", "sources",
        "supplementary", "material", "strategy",
    }
    out: list[str] = []
    for term in important_terms(query, limit * 2):
        cleaned = str(term or "").strip().lower()
        if not cleaned or cleaned in noisy or len(cleaned) < 3:
            continue
        if cleaned in {"tumorgenesis", "tumorigenesi"}:
            term = "tumorigenesis"
        out.append(term)
        if len(out) >= limit:
            break
    return list(dict.fromkeys(out))


def _conversation_frame_query(frame: dict | None, limit: int = 14) -> str:
    if not isinstance(frame, dict):
        return ""
    raw_terms = frame.get("active_terms") or []
    if not isinstance(raw_terms, list):
        return ""
    terms_out: list[str] = []
    for item in raw_terms:
        text = str(item or "").strip()
        if not text:
            continue
        for term in _policy_query_terms(text, limit=4):
            terms_out.append(term)
    return " ".join(list(dict.fromkeys(terms_out))[:limit])


def _is_external_followup_request(query: str) -> bool:
    text = f" {query or ''} ".lower()
    markers = (
        " search more ",
        " search again ",
        " look further ",
        " look more ",
        " dig deeper ",
        " all available data ",
        " all available source",
        " available data source",
        " use all your available",
        " more on this",
        " more about this",
    )
    return any(marker in text for marker in markers)


def _external_retrieval_seed(query: str, conversation_frame: dict | None = None) -> str:
    frame_query = _conversation_frame_query(conversation_frame)
    if _is_external_followup_request(query) and frame_query:
        task_terms = _policy_task_terms(query)
        if not task_terms:
            task_terms = ["evidence", "mechanism", "examples", "review"]
        return " ".join(list(dict.fromkeys([frame_query, *task_terms]))[:18])
    return query


def _clean_external_query(query: str) -> str:
    cleaned = _compact_text(str(query or "").replace('"', " ").replace("'", " "), 220)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" -:,;")
    if not cleaned:
        return ""
    terms_out = _policy_query_terms(cleaned, 24)
    if len(terms_out) < 2 and len(cleaned.split()) < 2:
        return ""
    return " ".join(terms_out) if terms_out else cleaned


def _policy_task_terms(query: str) -> list[str]:
    text = (query or "").lower()
    out: list[str] = []
    if any(marker in text for marker in ("how", "why", "mechanism", "pathway", "happen")):
        out.extend(["mechanism", "pathogenesis", "signaling", "immune", "inflammation", "metabolism"])
    if any(marker in text for marker in ("what", "which", "specific", "particular", "described", "reported")):
        out.extend(["examples", "specific", "species", "organisms", "reported", "review"])
    if any(marker in text for marker in ("evidence", "study", "paper", "trial")):
        out.extend(["evidence", "study", "review"])
    if any(marker in text for marker in ("compare", "versus", " vs ")):
        out.extend(["comparison", "difference"])
    return list(dict.fromkeys(out))[:12]


def _external_query_variants(query: str, limit: int = 4) -> list[str]:
    """Build generic privacy-safe biomedical query bridges for external literature search.

    This intentionally avoids topic-specific expansions. It preserves user terms,
    adds normalized concepts, and adds task words such as mechanism/evidence when
    the user asks for mechanisms, examples, or support.
    """
    anchors = _policy_query_terms(query)
    normalized = [normalize_idea(item) for item in extract_ideas(query, limit=8)]
    normalized = [item for item in normalized if item]
    task_terms = _policy_task_terms(query)
    candidates = [query]
    if anchors:
        candidates.append(" ".join(list(dict.fromkeys(anchors + task_terms))[:16]))
    if normalized:
        candidates.append(" ".join(list(dict.fromkeys(normalized + task_terms))[:16]))
    alias_terms: list[str] = []
    for anchor in _external_required_anchors(query):
        aliases = [item for item in sorted(_external_anchor_aliases(anchor), key=lambda value: (len(value), value)) if item != anchor]
        alias_terms.extend(aliases[:3])
    if alias_terms:
        candidates.append(" ".join(list(dict.fromkeys(alias_terms + task_terms))[:16]))
    if anchors and task_terms:
        candidates.append(" ".join(list(dict.fromkeys(anchors[:10] + task_terms))[:16]))

    deduped: list[str] = []
    seen: set[str] = set()
    for item in candidates:
        cleaned = _clean_external_query(item)
        key = cleaned.lower()
        if cleaned and key not in seen:
            seen.add(key)
            deduped.append(cleaned)
        if len(deduped) >= max(1, limit):
            break
    return deduped


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
    out: list[str] = []
    for raw in (text or "").splitlines():
        line = re.sub(r"^\s*[-*\d.)]+\s*", "", raw).strip()
        if not line or len(line) > 220:
            continue
        lowered = line.lower()
        if any(marker in lowered for marker in ("json", "note:", "because", "should", "return ")):
            continue
        if any(ch in line for ch in "{}[]"):
            continue
        out.append(line)
        if len(out) >= limit:
            break
    return list(dict.fromkeys(_compact_text(item, 220) for item in out if item.strip()))[:limit]


def _external_query_preserves_seed(seed_query: str, candidate_query: str) -> bool:
    """Reject planner variants that keep anchors but import unrelated source-specific trails.

    The rule is intentionally topic-general: preserve required anchors, allow a
    small amount of synonym/broader-vocabulary expansion, and reject long query
    strings dominated by novel terms not present in the user seed or generic
    task vocabulary.
    """
    cleaned = _clean_external_query(candidate_query)
    if not cleaned:
        return False
    seed_terms = set(_policy_query_terms(seed_query, 64))
    candidate_terms = set(_policy_query_terms(cleaned, 64))
    if not candidate_terms:
        return False
    for anchor in _external_required_anchors(seed_query):
        aliases = _external_anchor_aliases(anchor)
        if not any(alias in candidate_terms or re.search(rf"\b{re.escape(alias)}\b", cleaned.lower()) for alias in aliases):
            return False
    allowed_general = set(_policy_task_terms(seed_query)) | _EXTERNAL_TASK_OR_PROCESS_TERMS
    for anchor in _external_required_anchors(seed_query):
        allowed_general.update(_external_anchor_aliases(anchor))
    novel_terms = candidate_terms - seed_terms - allowed_general
    if len(candidate_terms) <= 12:
        return True
    novel_ratio = len(novel_terms) / max(1, len(candidate_terms))
    return len(novel_terms) <= 6 and novel_ratio <= 0.45


async def _llm_external_query_variants(query: str, base_variants: list[str], limit: int = 4) -> tuple[list[str], str]:
    messages = [
        {
            "role": "system",
            "content": external_query_system_prompt(),
        },
        {
            "role": "user",
            "content": (
                f"User query: {query[:1000]}\n"
                f"Deterministic base queries: {base_variants}\n"
                "Return JSON with keys: queries (1-4 concise keyword queries), note (one sentence explaining retrieval intent)."
            ),
        },
    ]
    text = await LLMClient().chat_once(
        messages,
        max_tokens=EXTERNAL_QUERY_MAX_TOKENS,
        agent="context_manager",
    )
    data = _extract_json_object(text) or {}
    raw_queries = data.get("queries") if isinstance(data, dict) else []
    if not isinstance(raw_queries, list):
        raw_queries = []
    queries = [str(item).strip() for item in raw_queries if str(item).strip()]
    queries = [item for item in (_clean_external_query(item) for item in queries) if item]
    queries = [item for item in queries if _external_query_preserves_seed(query, item)]
    if not queries:
        queries = _fallback_queries_from_text(text, limit)
        queries = [item for item in (_clean_external_query(item) for item in queries) if item]
        queries = [item for item in queries if _external_query_preserves_seed(query, item)]
    note = _compact_text(str(data.get("note") or ""), 400) if isinstance(data, dict) else ""
    return queries[:limit], note


_EXTERNAL_TASK_OR_PROCESS_TERMS = {
    "development", "evidence", "example", "examples", "immune",
    "immunity", "inflammation", "mechanism", "mechanisms", "metabolic",
    "metabolism", "organism", "organisms", "pathogenesis",
    "reported", "review", "signaling", "specific", "species", "study",
}


def _external_anchor_aliases(term: str) -> set[str]:
    lowered = str(term or "").lower()
    aliases = {lowered}
    if lowered.endswith("s") and len(lowered) > 4:
        aliases.add(lowered[:-1])
    if lowered == "fungi":
        aliases.update({"fungi", "fungal", "fungus", "mycobiome", "mycobiota"})
    elif lowered == "fungal":
        aliases.update({"fungi", "fungal", "fungus", "mycobiome", "mycobiota"})
    elif lowered == "tumorgenesis":
        aliases.add("tumorigenesis")
    elif lowered in {"tumorigenesis", "oncogenesis", "carcinogenesis"}:
        aliases.update({"tumorigenesis", "tumorigenic", "oncogenesis", "carcinogenesis", "cancer", "tumor", "tumors"})
    elif lowered in {"cancer", "tumor", "tumors"}:
        aliases.update({"cancer", "tumor", "tumors", "neoplasm", "neoplasms", "malignant", "malignancy"})
    return {item for item in aliases if item}


def _external_required_anchors(query: str) -> list[str]:
    anchors: list[str] = []
    for term in _policy_query_terms(query, 32):
        lowered = str(term or "").lower()
        if lowered in _EXTERNAL_TASK_OR_PROCESS_TERMS:
            continue
        anchors.append(lowered)
    return list(dict.fromkeys(anchors))[:6]


def _external_anchor_coverage(query: str, result: dict) -> tuple[list[str], list[str]]:
    required = _external_required_anchors(query)
    if not required:
        return [], []
    text = " ".join(str(result.get(key) or "") for key in ("title", "snippet")).lower()
    covered: list[str] = []
    missing: list[str] = []
    for anchor in required:
        aliases = _external_anchor_aliases(anchor)
        if any(re.search(rf"\b{re.escape(alias)}\b", text) for alias in aliases):
            covered.append(anchor)
        else:
            missing.append(anchor)
    return covered, missing


def _expanded_external_query_terms(query: str) -> set[str]:
    terms_set = set(_policy_query_terms(query, 64))
    for idea in extract_ideas(query, limit=16):
        normalized = normalize_idea(idea)
        if normalized:
            terms_set.update(important_terms(normalized, 8))
    terms_set.update(_policy_task_terms(query))
    return terms_set


def _rank_external_results(query: str, results: list[dict]) -> list[dict]:
    query_terms = _expanded_external_query_terms(query)
    if not query_terms:
        return results

    def score(result: dict) -> tuple[float, float]:
        source = str(result.get("source") or "")
        title_terms = set(important_terms(str(result.get("title") or ""), 80))
        snippet_terms = set(important_terms(str(result.get("snippet") or ""), 160))
        text_terms = title_terms | snippet_terms
        overlap = len(query_terms & text_terms) / max(1, len(query_terms))
        title_overlap = len(query_terms & title_terms) / max(1, min(len(query_terms), 12))
        semantic = 0.0
        if source.startswith("litsense2"):
            semantic += 0.16
        elif source == "pubtator3":
            semantic += 0.12
        elif source == "pmc_fulltext_sentence":
            semantic += 0.14
        elif source in {"pubmed", "pmc"}:
            semantic += 0.04
        if result.get("pmid") or result.get("pmcid"):
            semantic += 0.04
        if overlap <= 0.0 and title_overlap <= 0.0:
            semantic -= 0.20
        provider_score = float(result.get("score", 0.0) or 0.0)
        return overlap + (0.45 * title_overlap) + semantic, provider_score

    ranked = [dict(result) for result in results]
    ranked.sort(key=score, reverse=True)
    for result in ranked:
        result["external_rank_score"] = round(score(result)[0], 4)
        covered, missing = _external_anchor_coverage(query, result)
        if covered or missing:
            result["external_anchor_covered"] = covered
            result["external_anchor_missing"] = missing
    return ranked


def _merge_external_results(
    pubmed_results: list[dict],
    pubtator_results: list[dict],
    k: int,
    litsense_results: list[dict] | None = None,
    query: str = "",
) -> list[dict]:
    litsense_results = litsense_results or []
    pubmed_results = _rank_external_results(query, pubmed_results)
    pubtator_results = _rank_external_results(query, pubtator_results)
    litsense_results = _rank_external_results(query, litsense_results)
    if not query:
        auxiliary_result_sets = [items for items in (litsense_results, pubtator_results) if items]
        reserved_auxiliary_slots = min(len(auxiliary_result_sets), max(0, k - (1 if pubmed_results else 0)))
        keep_pubmed = max(0, k - reserved_auxiliary_slots)
        candidates = [
            *pubmed_results[:keep_pubmed],
            *litsense_results,
            *pubtator_results,
            *pubmed_results[keep_pubmed:],
        ]
    else:
        candidates = [*litsense_results, *pubtator_results, *pubmed_results]
        candidates.sort(
            key=lambda item: (float(item.get("external_rank_score", 0.0) or 0.0), float(item.get("score", 0.0) or 0.0)),
            reverse=True,
        )
        required_anchors = _external_required_anchors(query)
        if required_anchors:
            required_set = set(required_anchors)
            fully_anchored = [
                item
                for item in candidates
                if required_set.issubset(set(item.get("external_anchor_covered") or []))
            ]
            partially_anchored = [
                item
                for item in candidates
                if item.get("external_anchor_covered") and item not in fully_anchored
            ]
            if fully_anchored:
                candidates = fully_anchored + partially_anchored + [
                    item for item in candidates if item not in fully_anchored and item not in partially_anchored
                ]
            elif partially_anchored:
                candidates = partially_anchored + [item for item in candidates if item not in partially_anchored]
    merged: list[dict] = []
    seen: set[str] = set()
    for result in candidates:
        key = _external_result_key(result)
        if key and key in seen:
            continue
        if key:
            seen.add(key)
        merged.append(result)
        if len(merged) >= k:
            return merged
    return merged


def _external_attempt_quality(query: str, results: list[dict]) -> dict[str, Any]:
    required = set(_external_required_anchors(query))
    source_count = len({str(item.get("source") or "") for item in results if item.get("source")})
    if not results:
        return {
            "score": 0.0,
            "full_anchor_result_count": 0,
            "partial_anchor_result_count": 0,
            "source_count": 0,
            "stop_reason": "no_results",
        }
    full = 0
    partial = 0
    for item in results:
        covered = set(item.get("external_anchor_covered") or [])
        if required and required.issubset(covered):
            full += 1
        elif covered:
            partial += 1
    anchor_score = (full + (0.35 * partial)) / max(1, len(results))
    source_score = min(1.0, source_count / 3)
    score = round((0.78 * anchor_score) + (0.22 * source_score), 4)
    return {
        "score": score,
        "full_anchor_result_count": full,
        "partial_anchor_result_count": partial,
        "source_count": source_count,
        "stop_reason": "sufficient_relevance" if score >= 0.72 or full >= 2 else "retry_recommended",
    }


def _external_retry_queries(seed_query: str, prior_queries: list[str], results: list[dict], limit: int = 3, gap_spec: dict | None = None, entity_synonyms: dict[str, str] | None = None) -> list[str]:  # WP-F-1
    base_terms = _policy_query_terms(seed_query, 12)
    required = set(_external_required_anchors(seed_query))
    # WP-B: steer toward evidence gaps tracked in GapSpec
    if gap_spec:
        _missing = sorted(gap_spec.get("missing_entities") or [])[:4]
        _confirmed = set(gap_spec.get("confirmed_entities") or [])
        # WP-F-1: substitute bandit-selected synonym for each missing entity
        if entity_synonyms:
            _missing = [entity_synonyms.get(e, e) for e in _missing]
        base_terms = _missing + [t for t in base_terms if t not in _confirmed]
    candidates: list[str] = []
    for item in results[:6]:
        covered = set(item.get("external_anchor_covered") or [])
        if required and not required.issubset(covered):
            continue
        text = " ".join(str(item.get(key) or "") for key in ("title", "snippet"))
        feedback_terms = [
            term
            for term in important_terms(text, 18)
            if term not in _EXTERNAL_TASK_OR_PROCESS_TERMS
        ]
        query = _clean_external_query(" ".join(list(dict.fromkeys(base_terms + feedback_terms))[:18]))
        if query:
            candidates.append(query)
    candidates.append(_clean_external_query(" ".join(list(dict.fromkeys(base_terms + _policy_task_terms(seed_query)))[:18])))
    seen = {item.lower() for item in prior_queries}
    out: list[str] = []
    for item in candidates:
        key = item.lower()
        if item and key not in seen and _external_query_preserves_seed(seed_query, item):
            seen.add(key)
            out.append(item)
        if len(out) >= limit:
            break
    return out


async def _enrich_external_results(results: list[dict]) -> list[dict]:
    pmids = [
        str(item.get("pmid") or "").strip()
        for item in results
        if str(item.get("source") or "") == "pubtator3"
        and str(item.get("pmid") or "").strip()
    ]
    if not pmids:
        return results
    try:
        fetched = await pubmed_fetch_by_pmids(pmids)
    except Exception as e:
        print(f"[WARN] ContextPolicy PubTator abstract enrichment failed: {e}")
        return results
    enriched: list[dict] = []
    for item in results:
        pmid = str(item.get("pmid") or "")
        replacement = fetched.get(pmid)
        if replacement and replacement.get("snippet"):
            merged = dict(item)
            merged["snippet"] = replacement.get("snippet") or item.get("snippet") or ""
            merged["title"] = item.get("title") or replacement.get("title") or ""
            merged["pmcid"] = item.get("pmcid") or replacement.get("pmcid") or ""
            merged["pmc_url"] = item.get("pmc_url") or replacement.get("pmc_url") or ""
            merged["abstract_enriched"] = True
            enriched.append(merged)
        else:
            enriched.append(item)
    return enriched


def _render_ideas(ideas: list[dict]) -> str:
    """WP-F-3: onomasiological vocabulary guide for puzzle/answer agent."""
    if not ideas:
        return ""
    lines = ["Vocabulary guide (known concepts -- canonical names, aliases, hierarchy):"]
    for item in ideas[:MEMORY_IDEAS_RENDER]:
        idea = item.get("idea") or ""
        if not idea:
            continue
        synonyms = (item.get("synonyms") or [])[:4]
        parent = item.get("parent_idea")
        concept_path = item.get("concept_path") or []
        reward = float(item.get("reward_avg") or 0.0)
        freq = int(item.get("frequency") or item.get("session_frequency") or 0)
        parts = [f"  * {idea}"]
        if synonyms:
            _syn_str = ", ".join(synonyms)
            parts.append(f"(also: {_syn_str})")
        if parent and parent != idea:
            parts.append(f"[parent: {parent}]")
        elif len(concept_path) > 1:
            _path_str = " > ".join(concept_path)
            parts.append(f"[path: {_path_str}]")
        parts.append(f"(seen {freq}x, reward={reward:.3f})")
        lines.append(" ".join(parts))
    return "\n".join(lines)


def _policy_instruction() -> str:
    return (
        "Memory policy guidance:\n"
        "- Treat the context below as an OS-like working set selected from recent turns, indexed memory, landmarks, triplets, and optional web grounding.\n"
        "- Prefer facts supported by pinned snippets, retrieved triplets, privacy-filtered external biomedical grounding, or explicit user statements.\n"
        "- When local snippets are sparse but external PubMed/PMC/LitSense/PubTator grounding is present, use it with provenance and caveats instead of saying no context exists.\n"
        "- If a warning says facts may be inconsistent, briefly mention the uncertainty and ask the user which fact should be treated as authoritative.\n"
        "- Do not expose hidden reward scores or policy internals unless the user asks for diagnostics."
    )


class ContextPolicy:
    def __init__(self, tenant: str):
        self.tenant = tenant
        self.store = MemoryStore(tenant)

    async def plan(
        self,
        *,
        session_id: str,
        message: str,
        allow_web_search: bool,
        confidence_min: float,
        evidence_assembly: dict | None = None,
        gap_spec: dict | None = None,  # WP-B: GapSpec from prior auto_context pass
        user_id: str | None = None,  # WP-F-2: user-scope idea persistence
    ) -> ContextPlan:
        turn_index = await self.store.next_turn_index(session_id)
        working_token_budget = min(
            settings.memory.working_buffer_token_budget,
            max(256, int(settings.llm.max_input_tokens * settings.memory.token_budget_ratio)),
        )
        recent = await self.store.recent_messages(
            session_id,
            settings.memory.working_buffer_turns,
            token_budget=working_token_budget,
            query_text=message,
        )
        # Fetch conversation_frame first so active_terms can anchor memory search
        # for context-poor messages (short/vague, no biomedical anchors). This
        # ensures search_memory/ideas/triplets use the real topic, not 'a and b'.
        conversation_frame = await self.store.conversation_frame(session_id)
        _mem_query = message
        if _is_context_poor(message):
            _active_terms = (conversation_frame or {}).get("active_terms") or []
            if _active_terms:
                _mem_query = " ".join(str(_t) for _t in _active_terms[:8])
        memory_hits = await self.store.search_memory(session_id, _mem_query, settings.memory.memory_k)
        landmarks = await self.store.landmarks(session_id)
        summaries = await self.store.episodic_summaries(session_id, MEMORY_SUMMARIES_FETCH)
        latest_traces = await self.store.latest_traces(session_id, MEMORY_TRACES_FETCH)
        idea_hits = await self.store.search_ideas(session_id, _mem_query, min(MEMORY_IDEAS_FETCH, settings.memory.memory_k), user_id=user_id)  # WP-F-2
        state = state_key(important_terms(message))
        action_value_hints = await self.store.action_values(session_id, state, 3)

        triplets: list[dict] = []
        if search_triplets:
            try:
                triplets = await search_triplets(
                    self.tenant,
                    _mem_query,
                    confidence_min=confidence_min,
                )
            except Exception as e:
                print(f"[WARN] ContextPolicy triplet search failed: {e}")

        web_payload = {"results": [], "query": "", "redacted": False}
        local_sparse = len(memory_hits) + len(triplets) < 3
        puzzle = (evidence_assembly or {}).get("evidence_puzzle") or {}
        if not isinstance(puzzle, dict):
            puzzle = {}
        missing_nodes = [str(item) for item in (puzzle.get("missing_nodes") or []) if str(item).strip()]
        edge_support_status = str(puzzle.get("edge_support_status") or "").lower()
        try:
            relation_evidence_count = int(puzzle.get("relation_evidence_count") or 0)
        except (TypeError, ValueError):
            relation_evidence_count = 0
        local_evidence_weak = bool(
            edge_support_status in {"missing", "partial"}
            or (
                missing_nodes
                and (
                    relation_evidence_count < 2
                    or len(missing_nodes) >= max(2, len((puzzle.get("covered_nodes") or [])))
                )
            )
        )
        external_followup = _is_external_followup_request(message)
        run_external_grounding = bool(allow_web_search and (local_sparse or local_evidence_weak or external_followup))
        if run_external_grounding:
            pubmed_payload = {"results": [], "query": "", "redacted": False}
            pubtator_payload = {"results": [], "query": "", "redacted": False}
            litsense_payload = {"results": [], "query": "", "redacted": False}
            external_seed_query = _external_retrieval_seed(message, conversation_frame)
            deterministic_external_queries = _external_query_variants(external_seed_query, limit=4)
            external_queries = deterministic_external_queries
            external_planner_note = ""
            if settings.memory.auto_context_llm_refine:
                try:
                    llm_queries, external_planner_note = await _llm_external_query_variants(external_seed_query, external_queries, limit=4)
                    external_queries = list(dict.fromkeys([*llm_queries, *external_queries]))[:4]
                except Exception as e:
                    print(f"[WARN] ContextPolicy external query planning failed: {e}")
            pubmed_results: list[dict] = []
            pubtator_results: list[dict] = []
            litsense_results: list[dict] = []
            pmc_sentence_results: list[dict] = []
            external_attempts: list[dict[str, Any]] = []
            for external_query in external_queries:
                remaining = max(1, settings.memory.web_k - len(pubmed_results))
                try:
                    payload = await pubmed_pmc_search(external_query, remaining)
                    pubmed_payload = payload if not pubmed_payload.get("query") else pubmed_payload
                    for result in payload.get("results") or []:
                        item = dict(result)
                        item["external_query"] = external_query
                        item["external_query_source"] = "external_planner" if external_query not in deterministic_external_queries else "deterministic"
                        pubmed_results.append(item)
                except Exception as e:
                    print(f"[WARN] ContextPolicy PubMed/PMC search failed: {e}")
                remaining = max(1, settings.memory.web_k - len(pubtator_results))
                try:
                    payload = await pubtator3_search(external_query, remaining)
                    pubtator_payload = payload if not pubtator_payload.get("query") else pubtator_payload
                    for result in payload.get("results") or []:
                        item = dict(result)
                        item["external_query"] = external_query
                        item["external_query_source"] = "external_planner" if external_query not in deterministic_external_queries else "deterministic"
                        pubtator_results.append(item)
                except Exception as e:
                    print(f"[WARN] ContextPolicy PubTator 3 search failed: {e}")
                remaining = max(1, settings.memory.web_k - len(litsense_results))
                try:
                    payload = await litsense2_search(external_query, remaining)
                    litsense_payload = payload if not litsense_payload.get("query") else litsense_payload
                    for result in payload.get("results") or []:
                        item = dict(result)
                        item["external_query"] = external_query
                        item["external_query_source"] = "external_planner" if external_query not in deterministic_external_queries else "deterministic"
                        litsense_results.append(item)
                except Exception as e:
                    print(f"[WARN] ContextPolicy LitSense 2.0 search failed: {e}")
            pmcids = [
                str(item.get("pmcid") or "").strip()
                for item in [*pubmed_results, *pubtator_results]
                if str(item.get("pmcid") or "").strip()
            ]
            if pmcids:
                try:
                    payload = await pmc_relevant_sentence_search(" ".join(external_queries), pmcids[:5], settings.memory.web_k)
                    for result in payload.get("results") or []:
                        item = dict(result)
                        item["external_query"] = " ".join(external_queries)
                        item["external_query_source"] = "pmc_fulltext_deepening"
                        pmc_sentence_results.append(item)
                except Exception as e:
                    print(f"[WARN] ContextPolicy PMC full-text sentence search failed: {e}")
            if pmc_sentence_results:
                pubmed_results = [*pmc_sentence_results, *pubmed_results]
            attempt_results = _merge_external_results(
                pubmed_results,
                pubtator_results,
                settings.memory.web_k,
                litsense_results,
                " ".join(external_queries),
            )
            attempt_quality = _external_attempt_quality(" ".join(external_queries), attempt_results)
            external_attempts.append(
                {
                    "attempt": 1,
                    "query_variants": external_queries,
                    "source_result_counts": {
                        "pubmed_pmc": len(pubmed_results),
                        "pubtator3": len(pubtator_results),
                        "litsense2": len(litsense_results),
                        "pmc_fulltext_sentence": len(pmc_sentence_results),
                    },
                    "quality": attempt_quality,
                }
            )
            if attempt_quality.get("stop_reason") == "retry_recommended":
                # WP-F-1+F-4: concept-aware synonym selection for missing entities
                _entity_synonyms: dict[str, str] = {}
                if gap_spec and idea_hits:
                    from app.memory.vocabulary_store import VocabularyStore as _VS
                    if _VS.enabled():
                        _vs = _VS()
                        _session_scores = dict(_vs.session_top_terms(session_id, limit=60))
                        _idea_syns: dict[str, list[str]] = {}
                        for _r in idea_hits:
                            _canon = str(_r.get("normalized_idea") or _r.get("idea") or "")
                            if _canon:
                                _idea_syns[_canon] = list(dict.fromkeys(
                                    (_r.get("synonyms") or []) + (_r.get("terms") or []) + [_canon]
                                ))
                        for _ent in (gap_spec.get("missing_entities") or []):
                            _cands = list(dict.fromkeys([_ent] + _idea_syns.get(_ent, [])))
                            # WP-F-4: exploration tiebreaker -- prefer longest unseen synonym
                            _entity_synonyms[_ent] = max(
                                _cands,
                                key=lambda t: (_session_scores.get(t, 0.0), len(t)),
                            )
                retry_queries = _external_retry_queries(external_seed_query, external_queries, attempt_results, limit=2, gap_spec=gap_spec, entity_synonyms=_entity_synonyms)
                if retry_queries:
                    retry_pubmed_results: list[dict] = []
                    retry_pubtator_results: list[dict] = []
                    retry_litsense_results: list[dict] = []
                    retry_pmc_sentence_results: list[dict] = []
                    for external_query in retry_queries:
                        try:
                            payload = await pubmed_pmc_search(external_query, max(1, settings.memory.web_k - len(retry_pubmed_results)))
                            for result in payload.get("results") or []:
                                item = dict(result)
                                item["external_query"] = external_query
                                item["external_query_source"] = "retry_feedback"
                                retry_pubmed_results.append(item)
                        except Exception as e:
                            print(f"[WARN] ContextPolicy retry PubMed/PMC search failed: {e}")
                        try:
                            payload = await pubtator3_search(external_query, max(1, settings.memory.web_k - len(retry_pubtator_results)))
                            for result in payload.get("results") or []:
                                item = dict(result)
                                item["external_query"] = external_query
                                item["external_query_source"] = "retry_feedback"
                                retry_pubtator_results.append(item)
                        except Exception as e:
                            print(f"[WARN] ContextPolicy retry PubTator 3 search failed: {e}")
                        try:
                            payload = await litsense2_search(external_query, max(1, settings.memory.web_k - len(retry_litsense_results)))
                            for result in payload.get("results") or []:
                                item = dict(result)
                                item["external_query"] = external_query
                                item["external_query_source"] = "retry_feedback"
                                retry_litsense_results.append(item)
                        except Exception as e:
                            print(f"[WARN] ContextPolicy retry LitSense 2.0 search failed: {e}")
                    retry_pmcids = [
                        str(item.get("pmcid") or "").strip()
                        for item in [*retry_pubmed_results, *retry_pubtator_results]
                        if str(item.get("pmcid") or "").strip()
                    ]
                    if retry_pmcids:
                        try:
                            payload = await pmc_relevant_sentence_search(" ".join(retry_queries), retry_pmcids[:5], settings.memory.web_k)
                            for result in payload.get("results") or []:
                                item = dict(result)
                                item["external_query"] = " ".join(retry_queries)
                                item["external_query_source"] = "retry_pmc_fulltext_deepening"
                                retry_pmc_sentence_results.append(item)
                        except Exception as e:
                            print(f"[WARN] ContextPolicy retry PMC full-text sentence search failed: {e}")
                    if retry_pmc_sentence_results:
                        retry_pubmed_results = [*retry_pmc_sentence_results, *retry_pubmed_results]
                    retry_results = _merge_external_results(
                        retry_pubmed_results,
                        retry_pubtator_results,
                        settings.memory.web_k,
                        retry_litsense_results,
                        " ".join(retry_queries),
                    )
                    retry_quality = _external_attempt_quality(" ".join(retry_queries), retry_results)
                    external_attempts.append(
                        {
                            "attempt": 2,
                            "query_variants": retry_queries,
                            "source_result_counts": {
                                "pubmed_pmc": len(retry_pubmed_results),
                                "pubtator3": len(retry_pubtator_results),
                                "litsense2": len(retry_litsense_results),
                                "pmc_fulltext_sentence": len(retry_pmc_sentence_results),
                            },
                            "quality": retry_quality,
                            "feedback_from_attempt": 1,
                        }
                    )
                    pubmed_results.extend(retry_pubmed_results)
                    pubtator_results.extend(retry_pubtator_results)
                    litsense_results.extend(retry_litsense_results)
                    external_queries = list(dict.fromkeys([*external_queries, *retry_queries]))[:6]

            web_payload = (
                pubmed_payload
                if pubmed_payload.get("results")
                else litsense_payload
                if litsense_payload.get("results")
                else pubtator_payload
            )
            web_payload["results"] = await _enrich_external_results(
                _merge_external_results(
                    pubmed_results,
                    pubtator_results,
                    settings.memory.web_k,
                    litsense_results,
                    " ".join(external_queries),
                )
            )
            if external_planner_note:
                web_payload["planner_note"] = external_planner_note
            web_payload["query_variants"] = external_queries
            web_payload["query_seed"] = external_seed_query
            web_payload["query_seed_source"] = "conversation_frame" if external_seed_query != message else "current_message"
            web_payload["multi_search_attempts"] = external_attempts
            web_payload["trigger"] = {
                "local_sparse": local_sparse,
                "local_evidence_weak": local_evidence_weak,
                "external_followup": external_followup,
                "edge_support_status": edge_support_status,
                "missing_node_count": len(missing_nodes),
                "relation_evidence_count": relation_evidence_count,
            }
            if not web_payload.get("results"):
                try:
                    web_payload = await duckduckgo_search(external_seed_query, settings.memory.web_k)
                except Exception as e:
                    print(f"[WARN] ContextPolicy DuckDuckGo search failed: {e}")
            # WP-B: absorb PubTator entities from external results into gap_spec
            if gap_spec is not None:
                _confirmed = set(gap_spec.get("confirmed_entities") or [])
                for _item in web_payload.get("results") or []:
                    for _etype, _eids in (_item.get("pubtator_entities") or {}).items():
                        for _eid in (_eids or []):
                            _confirmed.add(f"{_etype}:{_eid}")
                gap_spec["confirmed_entities"] = sorted(_confirmed)

        warnings: list[str] = []
        if any(w in set(terms(message)) for w in ("not", "never", "without", "contradict", "conflict")) and triplets:
            warnings.append(
                "The current question may negate or challenge facts found in retrieved triplets. Treat the answer as potentially inconsistent unless the evidence resolves it."
            )

        sections = [
            _policy_instruction(),
            render_conversation_frame(conversation_frame),
            _render_landmarks(landmarks),
            _render_summaries(summaries),
            _render_recent(recent),
            _render_ideas(idea_hits),
            _render_memory(memory_hits),
            _render_triplets(triplets),
            _render_web(web_payload.get("results") or []),
        ]
        if latest_traces:
            reflection_lines = [
                t.get("reflection")
                for t in latest_traces
                if isinstance(t.get("reflection"), str) and t.get("reflection")
            ]
            if reflection_lines:
                sections.append("Recent policy reflections:\n" + "\n".join(f"- {r[:500]}" for r in reflection_lines[:MEMORY_REFLECTIONS_RENDER]))

        prefix = "\n\n".join(s for s in sections if s)
        selected_context = []
        selected_context.extend({"source": "recent", **m} for m in recent)
        selected_context.extend({"source": "episodic_summary", "text": m.get("summary") or m.get("text") or "", **m} for m in summaries)
        selected_context.extend({"source": "idea", "text": str(m.get("idea") or ""), **m} for m in idea_hits)
        selected_context.extend({"source": "memory", **m} for m in memory_hits)
        selected_context.extend({"source": "triplet", "text": t.get("sentence_text") or t.get("text") or "", **t} for t in triplets)
        selected_context.extend({"source": "web", "text": r.get("snippet") or "", **r} for r in web_payload.get("results") or [])

        # NER bootstrap: LLM entity grounding over all context sources (local + web)
        if gap_spec is not None and selected_context and getattr(settings.memory, "entity_grounding_enabled", True):
            try:
                await llm_ground_entities(message, snippets=selected_context, gap_spec=gap_spec)
            except Exception as _eg:
                print(f"[WARN] ContextPolicy entity grounding failed: {_eg}")

        return ContextPlan(
            turn_index=turn_index,
            context_prefix=prefix,
            selected_context=selected_context,
            retrieved_triplets=triplets,
            web_results=web_payload.get("results") or [],
            warnings=warnings,
            meta={
                "turn_index": turn_index,
                "recent_count": len(recent),
                "recent_token_count": sum(int(m.get("token_count", 0) or 0) for m in recent),
                "working_token_budget": working_token_budget,
                "episodic_summary_count": len(summaries),
                "memory_hit_count": len(memory_hits),
                "idea_count": len(idea_hits),
                "action_value_hint_count": len(action_value_hints),
                "conversation_frame_terms": conversation_frame.get("active_terms", [])[:12] if conversation_frame else [],
                "conversation_frame_avoid_terms": conversation_frame.get("avoided_terms", [])[:12] if conversation_frame else [],
                "triplet_count": len(triplets),
                "web_result_count": len(web_payload.get("results") or []),
                "web_query_redacted": web_payload.get("redacted", False),
                "web_query": web_payload.get("query", ""),
                "web_query_variants": web_payload.get("query_variants", []),
                "web_query_seed": web_payload.get("query_seed", ""),
                "web_query_seed_source": web_payload.get("query_seed_source", ""),
                "web_multi_search_attempts": web_payload.get("multi_search_attempts", []),
            },
        )

    async def observe_turn(
        self,
        *,
        session_id: str,
        turn_index: int,
        question: str,
        answer: str,
        selected_context: List[Dict[str, Any]],
        retrieved_triplets: List[Dict[str, Any]],
        pinned_snippets: List[Dict[str, Any]] | None = None,
        source_sentences: List[Dict[str, Any]] | None = None,
        search_plan: Dict[str, Any] | None = None,
        started_at: float,
        token_budget: int,
        user_id: str | None = None,  # WP-F-2: user-scope idea persistence
    ) -> Dict[str, Any]:
        answer_triples: list[dict] = []
        if extract_triples:
            try:
                result = await extract_triples([answer[:1800]], timeout_sec=60, num_extractions=6)
                answer_triples = result.get("triples", []) or []
            except Exception as e:
                print(f"[WARN] ContextPolicy answer triplet extraction failed: {e}")

        conflicts = detect_triplet_conflicts(
            answer_triples,
            retrieved_triplets,
            threshold=settings.memory.contradiction_threshold,
        )
        nli_evidence = await score_answer_triples(answer_triples, retrieved_triplets)
        claims = extract_atomic_claims(answer)
        prior_frame = await self.store.conversation_frame(session_id)
        prior_supported_claims = await self.store.supported_claim_evidence(session_id, 24)
        source_items = list(source_sentences or [])
        source_items.extend(prior_supported_claims)
        evidence_candidates = gather_evidence_candidates(
            prompt_context=selected_context,
            pinned_snippets=pinned_snippets or [],
            source_sentences=source_items,
            triplet_results=retrieved_triplets,
        )
        claim_support = await assess_claim_support(
            claims,
            evidence_candidates,
            max_nli_pairs_per_claim=MAX_NLI_PAIRS_PER_CLAIM,
        )
        answer_id = f"answer_{session_id}_{turn_index + 1}"
        evidence_table = build_evidence_table(
            answer_id=answer_id,
            session_id=session_id,
            turn_index=turn_index,
            claim_support=claim_support,
            tenant=self.tenant,
        )
        claim_support_dicts = [item.to_dict() for item in claim_support]
        longitudinal = longitudinal_consistency_report(
            question=question,
            answer=answer,
            claim_support=claim_support_dicts,
            prior_supported_claims=prior_supported_claims,
            frame=prior_frame,
        )
        _step_rewards = (search_plan or {}).get("step_rewards") or None
        reward = reward_report(
            question=question,
            answer=answer,
            selected_context=selected_context,
            conflicts=conflicts,
            nli_evidence=nli_evidence,
            claim_support=claim_support_dicts,
            longitudinal_consistency=longitudinal,
            search_plan=search_plan,
            elapsed_sec=max(0.0, time.monotonic() - started_at),
            token_budget=token_budget,
            step_rewards=_step_rewards,
        )

        await self.store.add_message(
            session_id=session_id,
            role="user",
            text=question,
            turn_index=turn_index,
            importance=0.65,
        )
        await self.store.add_message(
            session_id=session_id,
            role="assistant",
            text=answer,
            turn_index=turn_index + 1,
            triples=answer_triples,
            importance=0.5 + 0.5 * reward["score"],
        )
        await self.store.add_episodic_summary(
            session_id=session_id,
            turn_index=turn_index + 1,
            messages=[
                {
                    "role": "user",
                    "text": question,
                    "turn_index": turn_index,
                    "importance": 0.65,
                },
                {
                    "role": "assistant",
                    "text": answer,
                    "turn_index": turn_index + 1,
                    "importance": 0.5 + 0.5 * reward["score"],
                    "claim_support": claim_support_dicts,
                    "evidence_supported": any(item.status == "entailed" for item in claim_support),
                },
            ],
            reward_score=float(reward["score"]),
        )
        updated_frame = await self.store.update_conversation_frame(
            session_id=session_id,
            question=question,
            answer=answer,
            claim_support=claim_support_dicts,
            turn_index=turn_index + 1,
        )
        await self.store.update_memory_lifecycle(session_id=session_id, current_turn_index=turn_index + 1)
        await self.store.update_landmarks(session_id, question, reward)
        await self.store.add_evidence_table(
            session_id=session_id,
            turn_index=turn_index,
            answer_id=answer_id,
            evidence_table=evidence_table,
        )

        reflection = ""
        if settings.memory.use_llm_reflection:
            reflection = await self._reflect(question, answer, reward, conflicts)
        state = state_key(important_terms(question))
        action = {
            "selected_context_count": len(selected_context),
            "retrieved_triplet_count": len(retrieved_triplets),
            "selected_idea_count": sum(1 for item in selected_context if item.get("source") == "idea"),
            "web_result_count": sum(1 for item in selected_context if item.get("source") == "web"),
            "evidence_candidate_count": len(evidence_candidates),
            "auto_context_result_count": int((search_plan or {}).get("result_count", 0) or 0),
            "search_query_count": len((search_plan or {}).get("variants", []) or []),
            "search_level_count": len((search_plan or {}).get("levels", []) or []),
            "search_used_llm": bool((search_plan or {}).get("used_llm", False)),
        }
        action = {**action, "action_key": action_key(action)}
        await self.store.update_idea_index(
            session_id=session_id,
            texts=[question, answer],
            turn_index=turn_index,
            reward_score=float(reward["score"]),
            shared=bool(settings.memory.shared_policy_enabled),
            user_id=user_id,  # WP-F-2
        )
        await self.store.update_action_value(
            session_id=session_id,
            state_key=state,
            action_key=action["action_key"],
            reward_score=float(reward["score"]),
            shared=bool(settings.memory.shared_policy_enabled),
        )
        if search_plan and search_plan.get("state_key") and search_plan.get("action_key"):
            await self.store.update_action_value(
                session_id=session_id,
                state_key=str(search_plan["state_key"]),
                action_key=str(search_plan["action_key"]),
                reward_score=float(reward["score"]),
                shared=bool(settings.memory.shared_policy_enabled),
            )
            # P-3: credit the intent-resolution decision (tier-1 router / 120b /
            # heuristic) for context-poor turns with this turn's reward, so the
            # system can learn which resolution strategy pays off. Uses the same
            # ActionValue table (no new Q-layer — engineering rule 1/9).
            _ir = search_plan.get("intent_resolution") or {}
            if _ir.get("state_key") and _ir.get("action_key"):
                await self.store.update_action_value(
                    session_id=session_id,
                    state_key=str(_ir["state_key"]),
                    action_key=str(_ir["action_key"]),
                    reward_score=float(reward["score"]),
                    shared=bool(settings.memory.shared_policy_enabled),
                )
            note_parts = [
                str(search_plan.get("planner_note") or "").strip(),
                str(search_plan.get("note") or "").strip(),
            ]
            note = " ".join(part for part in note_parts if part).strip()
            if note:
                await self.store.add_search_policy_note(
                    session_id=session_id,
                    turn_index=turn_index,
                    note=note,
                    search_plan=search_plan,
                    reward_score=float(reward["score"]),
                )

        trace = {
            "state_terms": important_terms(question),
            "state_key": state,
            "action": action,
            "search_plan": search_plan or {},
            "reward": reward,
            "conflicts": conflicts,
            "nli_evidence": nli_evidence,
            "claim_support": claim_support_dicts,
            "longitudinal_consistency": longitudinal,
            "conversation_frame": {
                "summary": updated_frame.get("summary", ""),
                "active_terms": updated_frame.get("active_terms", [])[:16],
                "avoided_terms": updated_frame.get("avoided_terms", [])[:16],
                "supported_claim_count": len(updated_frame.get("supported_claims") or []),
                "contradicted_claim_count": len(updated_frame.get("contradicted_claims") or []),
            },
            "evidence_candidates": evidence_to_dicts(evidence_candidates),
            "evidence_table": evidence_table_debug_payload(evidence_table),
            "reflection": reflection,
            "answer_triple_count": len(answer_triples),
        }
        if settings.memory.reward_trace_enabled:
            await self.store.add_trace(session_id=session_id, turn_index=turn_index, trace=trace)
        return trace

    async def _reflect(self, question: str, answer: str, reward: dict, conflicts: list[dict]) -> str:
        _reward_score = float((reward or {}).get("score") or 0.5)
        _polarity = (
            "positive" if _reward_score >= 0.55
            else "negative" if _reward_score < 0.45
            else "mixed"
        )
        messages = [
            {
                "role": "system",
                "content": reflection_system_prompt(_polarity),
            },
            {
                "role": "user",
                "content": (
                    f"Question: {question[:1000]}\n"
                    f"Answer: {answer[:1000]}\n"
                    f"Reward: {reward}\n"
                    f"Conflicts: {len(conflicts)}"
                ),
            },
        ]
        try:
            text = await LLMClient().chat_once(
                messages,
                max_tokens=REFLECT_MAX_TOKENS,
                agent="reflection",
            )
            return " ".join(text.split())[:700]
        except Exception as e:
            print(f"[WARN] ContextPolicy reflection failed: {e}")
            return ""
