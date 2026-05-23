from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List


POSITIVE = {"thanks", "good", "great", "excellent", "useful", "clear", "correct", "helpful"}
NEGATIVE = {"wrong", "bad", "confused", "unclear", "incorrect", "useless", "hallucinated", "frustrated"}
NEGATORS = {"no", "not", "never", "without", "fails", "failed", "cannot", "can't", "doesn't", "isn't"}
MECHANISTIC_CONTEXT_TERMS = {
    "mechanistic", "mechanism", "tme", "tumor", "microenvironment", "stromal", "immune",
    "metabolic", "hypoxia", "angiogenesis", "crosstalk", "fibroblast", "macrophage",
    "invasion", "proliferation", "emt", "ecm", "nsclc", "carcinoma",
}
MATH_PHARM_SYNERGY_TERMS = {
    "combination", "index", "dose", "drug", "therapeutic", "toxicity", "ctcae",
    "adverse", "irae", "antagonism", "pharmacological",
}


def terms(text: str) -> list[str]:
    return [t.lower() for t in re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", text or "")]


def important_terms(text: str, limit: int = 24) -> list[str]:
    stop = {
        "the", "and", "for", "that", "this", "with", "from", "into", "what", "when",
        "where", "which", "would", "could", "should", "about", "there", "their",
        "your", "you", "are", "was", "were", "have", "has", "had", "not",
    }
    out: list[str] = []
    seen: set[str] = set()
    for t in terms(text):
        if t in stop or t in seen:
            continue
        seen.add(t)
        out.append(t)
        if len(out) >= limit:
            break
    return out


def lexical_overlap(a: str, b: str) -> float:
    aa = set(important_terms(a, 80))
    bb = set(important_terms(b, 120))
    if not aa or not bb:
        return 0.0
    return len(aa & bb) / max(1, len(aa | bb))


def context_support_score(answer: str, selected_context: List[Dict[str, Any]]) -> float:
    if not selected_context:
        return 0.0
    context_texts = [str(c.get("text") or c.get("snippet") or c.get("summary") or c.get("title") or "") for c in selected_context]
    message_texts = [
        str(c.get("text") or c.get("snippet") or c.get("summary") or c.get("title") or "")
        for c in selected_context
        if c.get("doc_type") == "message" or c.get("role") in {"user", "assistant"}
    ]
    global_support = lexical_overlap(answer, "\n".join(context_texts))
    message_support = lexical_overlap(answer, "\n".join(message_texts)) if message_texts else 0.0
    best_item_support = max((lexical_overlap(answer, item) for item in context_texts), default=0.0)
    return max(global_support, message_support, best_item_support)


def sentiment_score(text: str) -> float:
    toks = set(terms(text))
    pos = len(toks & POSITIVE)
    neg = len(toks & NEGATIVE)
    if pos == neg == 0:
        return 0.0
    return max(-1.0, min(1.0, (pos - neg) / max(1, pos + neg)))


def domain_alignment(question: str, answer: str, selected_context: List[Dict[str, Any]]) -> float:
    q_terms = set(important_terms(question, 40))
    if not q_terms:
        return 0.0
    evidence_terms = set(important_terms(answer, 80))
    for item in selected_context or []:
        evidence_terms.update(important_terms(str(item.get("text") or item.get("title") or ""), 80))
        for field in ("subject", "relation", "object"):
            evidence_terms.update(important_terms(str(item.get(field) or ""), 20))
    overlap = len(q_terms & evidence_terms) / max(1, len(q_terms))
    return max(0.0, min(1.0, overlap))


def off_topic_penalty(question: str, answer: str, selected_context: List[Dict[str, Any]]) -> float:
    q = set(terms(question))
    asks_mechanistic = bool(q & MECHANISTIC_CONTEXT_TERMS) or "functional synergy" in (question or "").lower()
    asks_pharm = bool(q & {"drug", "dose", "therapy", "therapeutic", "pharmacological"})
    if not asks_mechanistic or asks_pharm:
        return 0.0
    answer_hits = len(set(terms(answer or "")) & MATH_PHARM_SYNERGY_TERMS)
    context_text = " ".join(
        str(item.get("text") or item.get("title") or "")
        for item in selected_context or []
    )
    context_hits = len(set(terms(context_text)) & MATH_PHARM_SYNERGY_TERMS)
    hit_count = answer_hits + (0.25 * context_hits if answer_hits else 0.0)
    return round(min(1.0, hit_count / 4.0), 4)


def relation_similarity(a: str, b: str) -> float:
    return lexical_overlap(a or "", b or "")


def _norm(v: Any) -> str:
    return " ".join(terms(str(v or "")))


def detect_triplet_conflicts(
    candidate_triples: Iterable[Dict[str, Any]],
    known_triples: Iterable[Dict[str, Any]],
    *,
    threshold: float = 0.35,
) -> list[dict]:
    conflicts: list[dict] = []
    known = list(known_triples or [])
    for cand in candidate_triples or []:
        cs = _norm(cand.get("subject"))
        co = _norm(cand.get("object"))
        cr = cand.get("relation") or cand.get("predicate") or ""
        if not cs or not co:
            continue
        for old in known:
            os = _norm(old.get("subject"))
            oo = _norm(old.get("object"))
            old_rel = old.get("relation") or old.get("predicate") or ""
            if not os or not oo:
                continue
            same_pair = (cs == os and co == oo) or (cs == oo and co == os)
            if not same_pair:
                continue
            rel_sim = relation_similarity(cr, old_rel)
            cand_neg = bool(set(terms(str(cr))) & NEGATORS)
            old_neg = bool(set(terms(str(old_rel))) & NEGATORS)
            if rel_sim < threshold or cand_neg != old_neg:
                conflicts.append(
                    {
                        "candidate": cand,
                        "known": old,
                        "reason": "same entities but relation differs or polarity changes",
                    }
                )
    return conflicts[:5]


def reward_report(
    *,
    question: str,
    answer: str,
    selected_context: List[Dict[str, Any]],
    conflicts: List[Dict[str, Any]],
    nli_evidence: List[Dict[str, Any]] | None = None,
    claim_support: List[Dict[str, Any]] | None = None,
    longitudinal_consistency: Dict[str, Any] | None = None,
    search_plan: Dict[str, Any] | None = None,
    elapsed_sec: float,
    token_budget: int,
) -> Dict[str, Any]:
    relevance = lexical_overlap(question, answer)
    support = context_support_score(answer, selected_context)
    alignment = domain_alignment(question, answer, selected_context)
    off_topic = off_topic_penalty(question, answer, selected_context)
    sentiment_delta = sentiment_score(answer) - sentiment_score(question)
    conflict_penalty = min(1.0, len(conflicts) * 0.35)
    nli_items = nli_evidence or []
    if nli_items:
        entailment_avg = sum(float(i.get("entailment", 0.0) or 0.0) for i in nli_items) / len(nli_items)
        contradiction_avg = sum(float(i.get("contradiction", 0.0) or 0.0) for i in nli_items) / len(nli_items)
    else:
        entailment_avg = 0.0
        contradiction_avg = 0.0
    factuality = max(0.0, min(1.0, entailment_avg - contradiction_avg))
    claim_items = claim_support or []
    claim_count = len(claim_items)
    if claim_items:
        entailed_count = sum(1 for item in claim_items if item.get("status") == "entailed")
        contradicted_count = sum(1 for item in claim_items if item.get("status") == "contradicted")
        unsupported_count = sum(1 for item in claim_items if item.get("status") == "unsupported")
        citation_coverage = entailed_count / max(1, claim_count)
        claim_contradiction_penalty = contradicted_count / max(1, claim_count)
        unsupported_penalty = unsupported_count / max(1, claim_count)
    else:
        entailed_count = contradicted_count = unsupported_count = 0
        citation_coverage = 0.0
        claim_contradiction_penalty = 0.0
        unsupported_penalty = 0.0
    latency_penalty = min(1.0, elapsed_sec / 30.0)
    token_penalty = min(1.0, max(0, len(answer) // 4) / max(1, token_budget))
    longitudinal = longitudinal_consistency or {}
    frame_alignment = float(longitudinal.get("frame_alignment", 0.0) or 0.0)
    frame_drift_penalty = float(longitudinal.get("frame_drift_penalty", 0.0) or 0.0)
    prior_memory_conflict_penalty = min(1.0, float(longitudinal.get("prior_memory_conflict_count", 0.0) or 0.0) / 2.0)
    longitudinal_penalty = float(longitudinal.get("longitudinal_penalty", 0.0) or 0.0)
    assembly = (search_plan or {}).get("evidence_assembly") or {}
    refinement = assembly.get("refinement_quality") or {}
    assembly_quality = max(0.0, min(1.0, float(assembly.get("assembly_quality", 0.0) or 0.0)))
    feedback_accepted_count = int(refinement.get("accepted_feedback_term_count", 0) or 0)
    feedback_rejected_count = int(refinement.get("rejected_feedback_term_count", 0) or 0)
    feedback_rejected_result_count = int(refinement.get("rejected_feedback_result_count", 0) or 0)
    ungrounded_feedback_count = int(refinement.get("ungrounded_feedback_term_count", 0) or 0)
    search_query_drift_penalty = min(
        1.0,
        ungrounded_feedback_count / max(1, feedback_accepted_count + ungrounded_feedback_count),
    )
    edge_support = str((assembly.get("evidence_puzzle") or {}).get("edge_support_status") or "")
    unsupported_bridge_penalty = 0.0
    if unsupported_count and edge_support in {"missing", "partial"}:
        edge_gap = 1.0 if edge_support == "missing" else 0.5
        unsupported_bridge_penalty = min(1.0, edge_gap * unsupported_penalty)
    score = (
        0.22 * relevance
        + 0.24 * support
        + 0.14 * factuality
        + 0.16 * citation_coverage
        + 0.08 * alignment
        + 0.06 * frame_alignment
        + 0.05 * max(0.0, sentiment_delta)
        + 0.08 * (1.0 if selected_context else 0.25)
        + 0.08 * (1.0 - conflict_penalty)
        + 0.03 * assembly_quality
        - 0.20 * claim_contradiction_penalty
        - 0.10 * unsupported_penalty
        - 0.10 * off_topic
        - 0.10 * frame_drift_penalty
        - 0.15 * prior_memory_conflict_penalty
        - 0.08 * longitudinal_penalty
        - 0.08 * search_query_drift_penalty
        - 0.10 * unsupported_bridge_penalty
        - 0.05 * latency_penalty
        - 0.05 * token_penalty
    )
    return {
        "score": round(max(0.0, min(1.0, score)), 4),
        "relevance_to_question": round(relevance, 4),
        "context_support": round(support, 4),
        "domain_alignment": round(alignment, 4),
        "off_topic_penalty": round(off_topic, 4),
        "frame_alignment": round(frame_alignment, 4),
        "frame_drift_penalty": round(frame_drift_penalty, 4),
        "prior_memory_conflict_penalty": round(prior_memory_conflict_penalty, 4),
        "longitudinal_penalty": round(longitudinal_penalty, 4),
        "evidence_assembly_quality": round(assembly_quality, 4),
        "search_feedback_accepted_count": feedback_accepted_count,
        "search_feedback_rejected_count": feedback_rejected_count,
        "search_feedback_rejected_result_count": feedback_rejected_result_count,
        "search_query_drift_penalty": round(search_query_drift_penalty, 4),
        "unsupported_evidence_bridge_penalty": round(unsupported_bridge_penalty, 4),
        "sentiment_delta": round(sentiment_delta, 4),
        "triplet_conflict_penalty": round(conflict_penalty, 4),
        "nli_entailment_avg": round(entailment_avg, 4),
        "nli_contradiction_avg": round(contradiction_avg, 4),
        "nli_factuality": round(factuality, 4),
        "claim_count": claim_count,
        "claim_entailed_count": entailed_count,
        "claim_contradicted_count": contradicted_count,
        "claim_unsupported_count": unsupported_count,
        "citation_coverage": round(citation_coverage, 4),
        "claim_contradiction_penalty": round(claim_contradiction_penalty, 4),
        "unsupported_claim_penalty": round(unsupported_penalty, 4),
        "latency_penalty": round(latency_penalty, 4),
        "token_penalty": round(token_penalty, 4),
    }
