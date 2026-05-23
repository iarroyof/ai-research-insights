# services/api/app/rag/context.py
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

from app.search.store import get_sentences_by_ids
from app.search.hybrid import hybrid_search_sentences


# ---------- Config helpers (supporting old and new layouts) ----------

def _llm_max_input_tokens() -> int:
    # Lazy import to avoid circular dependency
    from app.config import settings
    
    # new: settings.llm.max_input_tokens ; old: settings.llm_max_input_tokens
    if hasattr(settings, "llm") and getattr(settings.llm, "max_input_tokens", None):
        return int(settings.llm.max_input_tokens)
    if hasattr(settings, "llm_max_input_tokens"):
        return int(settings.llm_max_input_tokens)
    return 6000  # sane default

def _retrieval_k_default() -> int:
    # Lazy import to avoid circular dependency
    from app.config import settings
    
    # new: settings.search.get("k") ; old: settings.vec_k or settings.retrieval.vec_k
    if hasattr(settings, "search"):
        if isinstance(settings.search, dict):
            return int(settings.search.get("k", 12))
        k = getattr(settings.search, "k", None)
        if k is not None:
            return int(k)
    # old names
    if hasattr(settings, "vec_k"):
        return int(settings.vec_k)
    if hasattr(settings, "retrieval") and isinstance(settings.retrieval, dict):
        return int(settings.retrieval.get("vec_k", 12))
    return 12


# ---------- Public API ----------

async def fetch_pinned_snippets(tenant: str, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Resolve [{paper_id, sent_id?}] into rich snippets with text and metadata.

    Returns each snippet as:
      {
        "paper_id": str, "sent_id": int | None, "text": str,
        "title": str | None, "pmid": str | None, "pmcid": str | None,
        "page": int | None,
        # Optional fields if available from store
        "subject": str | None, "predicate": str | None, "object": str | None,
      }
    """
    if not items:
        return []
    return await get_sentences_by_ids(tenant, items)


async def build_prompt_and_citations(
    tenant: str,
    message: str,
    pinned: List[Dict[str, Any]],
    options: Dict[str, Any],
) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
    """
    Build a grounded prompt and citation list.

    options:
      - allow_extra_retrieval: bool (default False)
      - extra_k: int (override default retrieval k)
      - token_budget_ratio: float in (0,1], ratio of LLM input budget for context (default 0.6)
      - include_spo_when_available: bool (default True) -> render [S; P; O] under snippet text if present
      - system_message: str (override default system instruction)
    """
    allow_extra = bool(options.get("allow_extra_retrieval", False))
    extra_k = int(options.get("extra_k", max(3, _retrieval_k_default())))
    token_budget_ratio = float(options.get("token_budget_ratio", 0.6))
    include_spo = bool(options.get("include_spo_when_available", True))

    # 1) Start with pinned snippets
    snippets: List[Dict[str, Any]] = list(pinned)

    # 2) Optionally augment with extra retrieval (only if pinned is sparse)
    if allow_extra and len(snippets) < 3:
        extra_hits = await hybrid_search_sentences(
            tenant=tenant,
            query=message,
            filters=options.get("filters", {}) or {},
            k=extra_k,
        )
        # extra_hits is expected to be list[dict] with at least text/paper_id/sent_id
        snippets.extend(extra_hits)

    # 3) Deduplicate on (paper_id, sent_id, text)
    seen: set[tuple] = set()
    uniq: List[Dict[str, Any]] = []
    for s in snippets:
        key = (s.get("paper_id"), s.get("sent_id"), s.get("text"))
        if key in seen:
            continue
        seen.add(key)
        uniq.append(s)
    snippets = uniq

    # 4) Trim context to fit a budget (rough token estimate)
    #    We use a simple heuristic: ~4 chars per token; reserve system + user + separators.
    max_input = _llm_max_input_tokens()
    budget_for_context_tokens = max(256, int(max_input * max(0.1, min(1.0, token_budget_ratio))))
    trimmed = _trim_snippets_to_budget(snippets, budget_for_context_tokens, include_spo)

    # 5) Build the prompt blocks
    system_default = (
        "You are a careful biomedical assistant. "
        "Answer concisely and ground your statements in the numbered context snippets. "
        "Do not treat absence of a relation from the snippets as evidence that the relation has no plausible connection; "
        "state that the context is insufficient unless a snippet directly supports that exclusion. "
        "When appropriate, reference them with [#]. Do not invent references."
    )
    system_msg = options.get("system_message") or system_default

    context_block = _render_context_block(trimmed, include_spo)
    prompt = (
        f"{system_msg}\n\n"
        f"Context snippets:\n{context_block}\n\n"
        f"User: {message}\n"
        f"Assistant:"
    )

    # 6) Citations payload
    citations = _build_citations(trimmed)

    meta = {
        "num_snippets": len(trimmed),
        "context_tokens_budget": budget_for_context_tokens,
        "max_input_tokens": max_input,
        "allow_extra_retrieval": allow_extra,
    }
    return prompt, citations, meta


# ---------- Internal helpers ----------

def _estimate_tokens(text: str) -> int:
    """
    Very rough token estimate: 1 token ~= 4 chars (safe-ish default).
    This avoids pulling in a tokenizer dependency here.
    """
    if not text:
        return 0
    # clamp to avoid tiny results for short strings
    est = max(1, len(text) // 4)
    return est


def _render_one_snippet(snippet: Dict[str, Any], index: int, include_spo: bool) -> str:
    """
    Render a single snippet to text. Prefer sentence text; optionally include S;P;O line.
    """
    base = snippet.get("text") or ""
    lines = [f"[{index}] {base}".strip()]
    if include_spo:
        sub = (snippet.get("subject") or "").strip()
        pred = (snippet.get("predicate") or "").strip()
        obj = (snippet.get("object") or "").strip()
        if any([sub, pred, obj]):
            lines.append(f"    S: {sub} | P: {pred} | O: {obj}")
    return "\n".join(lines)


def _render_context_block(snippets: List[Dict[str, Any]], include_spo: bool) -> str:
    parts: List[str] = []
    for i, s in enumerate(snippets, start=1):
        parts.append(_render_one_snippet(s, i, include_spo))
    return "\n\n".join(parts)


def _snippet_token_cost(snippet: Dict[str, Any], include_spo: bool) -> int:
    cost = _estimate_tokens(snippet.get("text") or "")
    if include_spo:
        # small bonus cost if S/P/O exist
        if any([(snippet.get("subject") or "").strip(),
                (snippet.get("predicate") or "").strip(),
                (snippet.get("object") or "").strip()]):
            cost += 12
    # include some separator margin
    return cost + 4


def _trim_snippets_to_budget(
    snippets: List[Dict[str, Any]],
    budget_tokens: int,
    include_spo: bool,
) -> List[Dict[str, Any]]:
    """
    Greedy keep-first trimming; you can swap to better selection (e.g., MMR) later.
    """
    out: List[Dict[str, Any]] = []
    used = 0
    for s in snippets:
        cost = _snippet_token_cost(s, include_spo)
        if used + cost > budget_tokens:
            break
        out.append(s)
        used += cost
    # Always return at least one snippet (if any exist)
    return out if out else (snippets[:1] if snippets else [])


def _build_citations(snippets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    cites: List[Dict[str, Any]] = []
    for s in snippets:
        pmid = s.get("pmid")
        pmcid = s.get("pmcid")
        links = {
            "pubmed": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else None,
            "pmc": f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/" if pmcid else None,
        }
        cites.append(
            {
                "paper_id": s.get("paper_id"),
                "title": s.get("title"),
                "text": s.get("text"),
                "pmid": pmid,
                "pmcid": pmcid,
                "page": s.get("page"),
                "sent_id": s.get("sent_id"),
                "source_sentence_id": s.get("source_sentence_id") or s.get("sent_id"),
                "search_level": s.get("search_level"),
                "retrieval_rank": s.get("retrieval_rank") or s.get("search_rank"),
                "bm25_score": s.get("bm25_score") if s.get("bm25_score") is not None else s.get("score"),
                "retrieval_score": s.get("retrieval_score") if s.get("retrieval_score") is not None else s.get("score"),
                "auto_query": s.get("auto_query"),
                "auto_query_label": s.get("auto_query_label"),
                "disease_tags": s.get("disease_tags", []),
                "mechanism_tags": s.get("mechanism_tags", []),
                "evidence_type_tags": s.get("evidence_type_tags", []),
                "links": links,
            }
        )
    return cites
