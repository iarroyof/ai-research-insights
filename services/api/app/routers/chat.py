# services/api/app/routers/chat.py
import asyncio
import hashlib
import re
import uuid
import json
import time
from typing import List, Optional, Dict, Any
import httpx
from fastapi import APIRouter, Request
from starlette.responses import StreamingResponse
from pydantic import BaseModel, Field

# Config
from app.config import settings

# LLM client
from app.clients.llm import LLMClient
from app.memory.policy import ContextPolicy
from app.prompts.agent_prompts import answer_system_prompt
from app.memory.search_agent import build_auto_context
from app.memory.store import MemoryStore
from app.services.provider_metrics import snapshot_provider_metrics

# SSE helper
from app.utils.sse import sse_stream

# Pinned-context resolution + prompt/citation builders
try:
    from app.rag.context import fetch_pinned_snippets
except Exception:
    fetch_pinned_snippets = None

try:
    from app.rag.context import build_prompt_and_citations
except Exception:
    build_prompt_and_citations = None

try:
    from app.rag.citations import build_prompt
except Exception:
    build_prompt = None


router = APIRouter(prefix="/chat", tags=["chat"])


# ---------- Request/response models ----------
class ChatItem(BaseModel):
    paper_id: str
    sent_id: Optional[str] = None
    # Additional fields from search results (optional)
    text: Optional[str] = None
    subject: Optional[str] = None
    relation: Optional[str] = None
    object: Optional[str] = None
    title: Optional[str] = None
    pmid: Optional[str] = None
    pmcid: Optional[str] = None
    page: Optional[int] = None
    confidence: Optional[float] = None


class ChatOptions(BaseModel):
    allow_extra_retrieval: bool = Field(default=False)
    allow_auto_context: bool = Field(default=True)
    allow_memory: bool = Field(default=True)
    allow_web_search: Optional[bool] = Field(default=None)
    token_budget: int = Field(default=2000)
    confidence_min: float = Field(default=0.5)
    expose_memory_debug: bool = Field(default=False)
    chat_provider: Optional[str] = Field(default=None)
    chat_model: Optional[str] = Field(default=None)
    chat_api_format: Optional[str] = Field(default=None)
    context_provider: Optional[str] = Field(default=None)
    context_model: Optional[str] = Field(default=None)
    context_api_format: Optional[str] = Field(default=None)


class ChatRequest(BaseModel):
    message: str
    items: List[ChatItem] = Field(default_factory=list)
    session_id: Optional[str] = None
    options: ChatOptions = Field(default_factory=ChatOptions)


class ChatCorrectionRequest(BaseModel):
    session_id: str
    correction: str = Field(min_length=1, max_length=4000)
    conflicting_claim: Optional[str] = Field(default=None, max_length=2000)
    authoritative_fact: Optional[str] = Field(default=None, max_length=2000)
    turn_index: Optional[int] = None


def _native_history_messages(context_plan, limit: int = 12) -> List[Dict[str, str]]:
    """
    Convert selected recent-memory docs into native chat messages.

    The memory prefix is still used for summaries, evidence, ideas, and policy
    guidance; native role messages give the chat model normal turn continuity.
    """
    if not context_plan:
        return []
    messages: List[Dict[str, str]] = []
    for item in context_plan.selected_context:
        if item.get("source") != "recent":
            continue
        role = item.get("role")
        if role not in {"user", "assistant"}:
            continue
        content = (item.get("text") or item.get("summary") or "").strip()
        if not content:
            continue
        messages.append({"role": role, "content": content[:4000]})
    return messages[-limit:]


def _display_terms(items: Any, *, limit: int = 5) -> str:
    if not isinstance(items, list):
        return ""
    out: List[str] = []
    for item in items:
        text = str(item or "").strip()
        if not text or text.lower() in {"answer", "again", "now", "question", "evidence"}:
            continue
        if text not in out:
            out.append(text)
        if len(out) >= limit:
            break
    return ", ".join(out)


def _opening_clarification_prefix(evidence_assembly: Dict[str, Any] | None) -> str:
    assembly = evidence_assembly or {}
    if not assembly.get("clarification_recommended"):
        return ""
    frames = [
        str(item.get("label") or item.get("frame_id") or "").strip()
        for item in assembly.get("candidate_frames", [])[:3]
        if item.get("label") or item.get("frame_id")
    ]
    frame_text = ", ".join(frames) if frames else "the most relevant evidence frame"
    puzzle = assembly.get("evidence_puzzle") or {}
    covered = _display_terms(puzzle.get("covered_nodes"), limit=5)
    missing = _display_terms(puzzle.get("missing_nodes"), limit=5)
    edge_status = str(puzzle.get("edge_support_status") or "uncertain").strip() or "uncertain"
    level_counts = assembly.get("level_result_counts") or {}
    level_summary = ", ".join(
        f"{level}:{count}"
        for level, count in level_counts.items()
        if count
    )
    reasoning_parts: List[str] = []
    if covered:
        reasoning_parts.append(f"retrieval covers {covered}")
    if missing:
        reasoning_parts.append(f"the unresolved bridge includes {missing}")
    reasoning_parts.append(f"edge support is {edge_status}")
    if level_summary:
        reasoning_parts.append(f"retrieved levels {level_summary}")
    reasoning = "; ".join(reasoning_parts)
    return (
        "I will assemble the supported evidence pieces first and keep missing links explicit. "
        f"From the current evidence puzzle, {reasoning}. "
        f"To refine the explanation, which interpretation should lead: {frame_text}?\n\n"
    )


def _hold_generation_for_clarification(evidence_assembly: Dict[str, Any] | None) -> bool:
    assembly = evidence_assembly or {}
    if not assembly.get("clarification_recommended"):
        return False
    edge_status = ((assembly.get("evidence_puzzle") or {}).get("edge_support_status") or "").lower()
    if edge_status != "missing":
        return False
    puzzle = assembly.get("evidence_puzzle") or {}
    missing_nodes = [
        str(item).strip()
        for item in (puzzle.get("missing_nodes") or [])
        if str(item or "").strip()
    ]
    level_counts = assembly.get("level_result_counts") or {}
    retrieved_count = sum(int(count or 0) for count in level_counts.values())
    if not missing_nodes and retrieved_count > 0:
        return False
    return True


def _provider_error_payload(exc: Exception) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "message": "Hosted chat generation failed. The endpoint will return retrieved evidence and diagnostics instead of closing the stream.",
        "error_type": exc.__class__.__name__,
    }
    if isinstance(exc, httpx.HTTPStatusError):
        payload["status_code"] = exc.response.status_code
        payload["provider_url"] = str(exc.request.url)
    return payload


def _evidence_only_fallback_answer(
    *,
    citations: Dict[str, Any] | None,
    context_plan,
    auto_context_plan: Dict[str, Any] | None,
) -> str:
    snippets = (citations or {}).get("snippets", []) if isinstance(citations, dict) else []
    web_results = list(getattr(context_plan, "web_results", []) or [])
    puzzle = ((auto_context_plan or {}).get("evidence_assembly") or {}).get("evidence_puzzle") or {}
    lines = [
        "Hosted chat generation failed before a complete answer was produced. I can still report the retrieved evidence state:",
    ]
    if puzzle:
        covered = _display_terms(puzzle.get("covered_nodes"), limit=6)
        missing = _display_terms(puzzle.get("missing_nodes"), limit=6)
        edge_status = str(puzzle.get("edge_support_status") or "uncertain")
        lines.append(f"- Evidence puzzle: edge support is {edge_status}.")
        if covered:
            lines.append(f"- Covered nodes: {covered}.")
        if missing:
            lines.append(f"- Missing nodes: {missing}.")
    if snippets:
        lines.append("- Local evidence:")
        for idx, snippet in enumerate(snippets[:3], start=1):
            text = str(snippet.get("text") or snippet.get("title") or "").strip()
            if text:
                lines.append(f"  {idx}. {text[:360]}")
    if web_results:
        lines.append("- External biomedical grounding:")
        for idx, result in enumerate(web_results[:3], start=1):
            title = str(result.get("title") or result.get("source") or "external result").strip()
            snippet = str(result.get("snippet") or "").strip()
            provenance = result.get("pmid") or result.get("pmcid") or ""
            suffix = f" ({provenance})" if provenance else ""
            lines.append(f"  {idx}. {title}{suffix}: {snippet[:360]}")
    if len(lines) == 1:
        lines.append("- No usable evidence was retrieved before the provider failure.")
    lines.append("Please retry generation; retrieval diagnostics and citations are included below.")
    return "\n".join(lines) + "\n"


def _retrieval_pipeline_trace(
    *,
    auto_context_plan: Dict[str, Any] | None,
    context_plan,
    answer_mode: str,
    external_grounding_covers_puzzle: bool,
) -> Dict[str, Any]:
    auto_context_plan = auto_context_plan or {}
    evidence_assembly = auto_context_plan.get("evidence_assembly") or {}
    meta = getattr(context_plan, "meta", {}) or {}
    return {
        "version": "search_retrieval_pipeline:v1",
        "sequence": [
            {
                "step": "frame_interpretation",
                "owner": "SearchAgent",
                "output": {
                    "search_frame": auto_context_plan.get("search_frame", {}),
                    "state_key": auto_context_plan.get("state_key"),
                    "action_key": auto_context_plan.get("action_key"),
                },
            },
            {
                "step": "local_multilevel_retrieval",
                "owner": "SearchAgent",
                "output": {
                    "levels": auto_context_plan.get("levels", []),
                    "result_count": auto_context_plan.get("result_count", 0),
                    "skipped_off_topic_count": auto_context_plan.get("skipped_off_topic_count", 0),
                    "retrieval_record_count": len(auto_context_plan.get("retrieval_records", []) or []),
                },
            },
            {
                "step": "evidence_puzzle_assembly",
                "owner": "EvidenceAssembly",
                "output": {
                    "clarification_recommended": bool(evidence_assembly.get("clarification_recommended")),
                    "puzzle": evidence_assembly.get("evidence_puzzle", {}),
                    "refinement_quality": evidence_assembly.get("refinement_quality", {}),
                },
            },
            {
                "step": "session_memory_retrieval",
                "owner": "MemoryAgent",
                "output": {
                    "selected_context_count": len(getattr(context_plan, "selected_context", []) or []),
                    "triplet_count": len(getattr(context_plan, "retrieved_triplets", []) or []),
                    "idea_count": meta.get("idea_count", 0),
                },
            },
            {
                "step": "external_biomedical_grounding",
                "owner": "ContextPolicy",
                "output": {
                    "web_result_count": len(getattr(context_plan, "web_results", []) or []),
                    "query_seed_source": meta.get("web_query_seed_source"),
                    "query_variants": meta.get("web_query_variants", []),
                    "multi_search_attempts": meta.get("web_multi_search_attempts", []),
                    "external_grounding_covers_puzzle": external_grounding_covers_puzzle,
                },
            },
            {
                "step": "answer_policy",
                "owner": "AnswerPolicyAgent",
                "output": {
                    "answer_mode": answer_mode,
                    "external_grounding_override": external_grounding_covers_puzzle,
                },
            },
            {
                "step": "post_generation_verification",
                "owner": "PostGenerationVerifier",
                "output": {
                    "guard_expected": _needs_post_generation_guard(answer_mode, evidence_assembly),
                },
            },
        ],
    }


_PUZZLE_NODE_STOPWORDS = frozenset({
    # Core function words / verbs / prepositions
    "does", "play", "role", "make", "have", "been", "that", "with", "from", "this",
    "what", "how", "why", "when", "where", "which", "who", "the", "and", "for",
    "are", "was", "its", "not", "but", "also", "can", "may", "will", "shall",
    # Common adjectives / quantifiers
    "possible", "likely", "various", "certain", "multiple", "specific",
    "known", "given", "new", "old", "big", "small", "high", "low", "more", "less",
    "all", "any", "some", "each", "both", "other", "such", "one", "two", "three",
    # Common verbs / gerunds appearing in queries
    "involving", "include", "including", "using", "use", "used", "based", "related",
    "associated", "show", "shows", "shown", "found", "suggest", "suggests", "affect",
    # Generic nouns
    "life", "style", "body", "type", "form", "part", "way", "time", "case", "level",
    "effect", "function", "system", "process", "activity", "response", "outcome",
    "lung",
    # Logical connectors
    "between", "through", "without", "within", "during", "before", "after",
    "under", "over", "around", "another", "among", "across", "about", "toward",
    "because", "therefore", "however", "although",
})


def _specific_puzzle_nodes(puzzle):
    """Filter puzzle nodes to specific biomedical terms only (exclude common English words)."""
    raw = puzzle.get("missing_nodes") or puzzle.get("candidate_nodes") or []
    return [
        str(item).strip().lower()
        for item in raw
        if str(item or "").strip()
        and str(item).strip().lower() not in _PUZZLE_NODE_STOPWORDS
        and len(str(item).strip()) > 2
    ]


def _external_grounding_covers_puzzle(evidence_assembly: Dict[str, Any] | None, web_results: List[Dict[str, Any]] | None) -> bool:
    """Return True when external web results are sufficient to attempt a direct answer.

    Old logic required ALL specific puzzle nodes in ONE result (too strict).
    New logic: if any web results exist, override clarification and attempt direct_answer.
    The direct_answer contract already requires the LLM to separate supported facts from
    unsupported bridges, so it handles partial or off-topic evidence gracefully.
    """
    assembly = evidence_assembly or {}
    if not _hold_generation_for_clarification(assembly):
        return False  # Clarification was not recommended; nothing to override
    if not web_results:
        return False  # No web evidence at all; clarification may genuinely help
    puzzle = assembly.get("evidence_puzzle") or {}
    nodes = _specific_puzzle_nodes(puzzle)
    if not nodes:
        return True  # All query terms are stopwords; any web result is sufficient
    # Collective coverage: any specific node appearing anywhere across all results
    all_covered: set = set()
    for result in web_results:
        for item in (result.get("external_anchor_covered") or []):
            all_covered.add(str(item or "").strip().lower())
        text = " ".join(str(result.get(key) or "") for key in ("title", "snippet")).lower()
        for node in nodes:
            if node in text:
                all_covered.add(node)
    # Whether or not coverage matched, web results exist: attempt direct_answer.
    # The LLM will express uncertainty about missing evidence explicitly.
    return True

ANSWER_MODE_CONTRACTS: Dict[str, str] = {
    "direct_answer": "Answer directly from supplied evidence. Separate supported facts from unsupported bridges.",
    "novice_rewrite": (
        "Rewrite for a novice by compressing only supported puzzle edges. Preserve caveats. "
        "Do not introduce new named mechanisms, new outcomes, or broad-to-specific mechanism conversions. "
        "If edge support is missing, state that the available evidence supports only the broader direction."
    ),
    "expert_mechanism": "Explain mechanism edges only when the supplied evidence supports the edge direction and required intermediate nodes.",
    "phrase_evaluation": "Judge the proposed wording first as supported, unsupported, contradicted, or too broad; do not answer a stale prior topic.",
    "diagnostic_trace_answer": (
        "Answer about trace/evaluator evidence using diagnostic fields, not biomedical inference. "
        "Do not retrieve or discuss unrelated biomedical snippets for this mode. "
        "Name the trace fields the developer should inspect: user turn, expected behavior, extracted claims, claim judgments, source sentence IDs, BM25/retrieval records, evidence puzzle, answer mode, reward penalties, failure owner, and recommendations. "
        "If the user references a correction, preserve that correction and do not reverse a prior false-premise rejection."
    ),
    "correction_acknowledgement": "Acknowledge the user correction and update scope without adding new evidence claims.",
    "clarification": "Summarize the current puzzle state and ask one focused textual clarification.",
}


def _contains_any_text(message: str, markers: tuple[str, ...]) -> bool:
    lower = (message or "").lower()
    return any(marker in lower for marker in markers)


def _answer_mode(
    message: str,
    evidence_assembly: Dict[str, Any] | None,
    *,
    correction_only_turn: bool,
    resolved_query: str | None = None,
) -> str:
    # P-4: utterance-about-the-conversation modes (correction, clarification,
    # phrase_evaluation, diagnostic_trace_answer) are decided ONLY from the raw
    # message — they describe what the user literally said. Question-type modes
    # (novice_rewrite, expert_mechanism) may ALSO consider the resolved query so a
    # context-poor reply ("the second one") that resolves to a mechanism question
    # gets the right mode. resolved_query is passed only when the caller has it
    # AND memory.answer_mode_consider_resolved_query is enabled (default off →
    # behaviour unchanged unless deliberately turned on).
    if correction_only_turn:
        return "correction_acknowledgement"
    if _hold_generation_for_clarification(evidence_assembly):
        return "clarification"
    if _contains_any_text(
        message,
        (
            "can the chatbot phrase",
            "can i phrase",
            "could i phrase",
            "phrase the answer",
            "is this phrase",
            "is the phrase",
            "is this wording",
            "is the wording",
            "the statement",
            "this statement",
            "that statement",
        ),
    ):
        return "phrase_evaluation"
    if _contains_any_text(message, ("reward model", "evaluator", "trace evidence", "before changing code", "diagnostic", "debug")):
        return "diagnostic_trace_answer"
    # Question-type modes: consider the raw message and (if provided) the resolved
    # query, so a context-poor reply that resolves to a mechanism/novice request
    # is classified correctly.
    question_text = f"{message}\n{resolved_query}" if resolved_query else message
    if _contains_any_text(
        question_text,
        (
            "novice",
            "one paragraph",
            "one-paragraph",
            "rewrite",
            "summarize",
            "summary",
            "concise answer",
            "essential caveat",
            "must not disappear",
        ),
    ):
        return "novice_rewrite"
    if _contains_any_text(question_text, ("mechanism", "mechanistic", "pathway", "explain how", "why does")):
        return "expert_mechanism"
    return "direct_answer"


def _prompt_hash(*parts: str) -> str:
    return hashlib.sha256("\n\n".join(parts).encode("utf-8")).hexdigest()[:24]


def _answer_mode_prompt(answer_mode: str, evidence_assembly: Dict[str, Any] | None) -> str:
    contract = ANSWER_MODE_CONTRACTS.get(answer_mode, ANSWER_MODE_CONTRACTS["direct_answer"])
    puzzle = (evidence_assembly or {}).get("evidence_puzzle") or {}
    return (
        "Answer mode contract:\n"
        f"- mode: {answer_mode}\n"
        f"- contract: {contract}\n"
        f"- puzzle_edge_support_status: {puzzle.get('edge_support_status', 'unknown')}\n"
        f"- puzzle_relation_evidence_count: {puzzle.get('relation_evidence_count', 0)}\n"
        f"- puzzle_covered_nodes: {puzzle.get('covered_nodes', [])[:8]}\n"
        f"- puzzle_missing_nodes: {puzzle.get('missing_nodes', [])[:8]}\n"
    )


def _needs_post_generation_guard(answer_mode: str, evidence_assembly: Dict[str, Any] | None) -> bool:
    puzzle = (evidence_assembly or {}).get("evidence_puzzle") or {}
    edge_status = str(puzzle.get("edge_support_status") or "").lower()
    relation_count = int(puzzle.get("relation_evidence_count") or 0)
    return answer_mode in {"novice_rewrite", "clarification"} and (
        edge_status in {"missing", "partial"} or relation_count <= 0
    )


def _post_generation_expansion_guard(
    answer: str,
    *,
    answer_mode: str,
    evidence_assembly: Dict[str, Any] | None,
    source_snippets: List[Dict[str, Any]] | None = None,
) -> tuple[str, dict[str, Any]]:
    assembly = evidence_assembly or {}
    puzzle = assembly.get("evidence_puzzle") or {}
    if not _needs_post_generation_guard(answer_mode, assembly):
        return answer, {"applied": False, "accepted_claims": [], "repaired_claims": [], "blocked_claims": []}

    covered = _display_terms(_filter_display_nodes(puzzle.get("covered_nodes")), limit=6)
    missing = _display_terms(_filter_display_nodes(puzzle.get("missing_nodes")), limit=6)
    edge_status = str(puzzle.get("edge_support_status") or "uncertain")
    relation_count = int(puzzle.get("relation_evidence_count") or 0)
    blocked_claims = [
        line.strip(" -*\t")
        for line in re.split(r"[\n.;]", answer or "")
        if line.strip() and len(line.strip()) > 24
    ][:8]

    source_points = _source_bound_points(source_snippets or [])
    if source_points:
        repair = (
            "For a novice: keep the explanation at the level supported by the retrieved source sentences. "
            f"The supported points are: {'; '.join(source_points)}. "
        )
    else:
        subject = covered or "the retrieved concepts"
        repair = (
            f"For a novice: the available evidence supports only the broader direction involving {subject}. "
            f"The current evidence puzzle has {edge_status} edge support"
            f" and {relation_count} validated relation-evidence link{'s' if relation_count != 1 else ''}. "
        )
    if missing:
        repair += f"Do not add a detailed mechanism across {missing} unless specific source sentences support those edges. "
    repair += "Keep unverified causes, mediators, and outcomes explicitly caveated."

    return repair, {
        "applied": True,
        "answer_mode": answer_mode,
        "edge_support_status": edge_status,
        "relation_evidence_count": relation_count,
        "accepted_claims": [repair],
        "repaired_claims": [
            {
                "reason": "answer exceeded supported puzzle boundary",
                "repair": "replaced generated answer with evidence-boundary novice wording",
            }
        ],
        "blocked_claims": blocked_claims,
    }


def _filter_display_nodes(nodes: Any) -> list[str]:
    blocked = {
        "give", "one", "one paragraph", "paragraph", "version", "novice", "user", "but",
        "keep", "biomedical", "direction", "answer", "again", "after", "correction",
    }
    kept: list[str] = []
    for node in nodes or []:
        text = str(node).strip().lower()
        if not text or text in blocked or len(text) < 3:
            continue
        kept.append(str(node).strip())
    return kept[:10]


def _source_bound_points(snippets: List[Dict[str, Any]]) -> list[str]:
    points: list[str] = []
    for idx, snippet in enumerate(snippets[:2], start=1):
        text = str(snippet.get("text") or snippet.get("title") or "").strip()
        if not text:
            continue
        text = re.sub(r"\[[^\]]+\]", "", text)
        text = re.sub(r"\s+", " ", text).strip(" .")
        if len(text) > 220:
            text = text[:217].rsplit(" ", 1)[0] + "..."
        points.append(f"[{idx}] {text}")
    return points


def _is_scope_or_memory_correction_only(message: str) -> bool:
    lower = (message or "").strip().lower()
    if not lower or "?" in lower:
        return False
    correction_markers = (
        "from now on",
        "going forward",
        "for the rest of",
        "remember that",
        "please remember",
        "stay only",
        "focus only",
        "keep the scope",
        "do not drift",
        "don't drift",
    )
    return any(marker in lower for marker in correction_markers)


def _correction_acknowledgement(message: str) -> str:
    correction = re.sub(r"\s+", " ", (message or "").strip())
    if len(correction) > 500:
        correction = correction[:497].rstrip() + "..."
    return (
        "Understood. I will treat this as a session scope correction: "
        f"{correction}. I will use it to constrain later retrieval and answers unless you explicitly ask to revisit it."
    )


@router.post("/memory/correction")
async def save_memory_correction(req: Request, body: ChatCorrectionRequest):
    """
    Persist a user-confirmed correction after a consistency warning.

    This keeps the workflow controlled: the model warns, the UI can ask the user
    what should be treated as true, and the answer becomes a high-importance
    session landmark for later turns.
    """
    tenant: str = getattr(req.state, "tenant_id", "default")
    store = MemoryStore(tenant)
    await store.add_user_correction(
        session_id=body.session_id,
        correction=body.correction,
        conflicting_claim=body.conflicting_claim,
        authoritative_fact=body.authoritative_fact,
        turn_index=body.turn_index,
    )
    return {"ok": True, "session_id": body.session_id}


@router.get("/memory/provider-metrics")
async def provider_metrics(req: Request):
    """
    Return in-process hosted-provider counters for diagnostics.

    This is intentionally compact: no prompts, responses, or secrets are
    exposed. Counters reset when the API process restarts.
    """
    return {
        "tenant": getattr(req.state, "tenant_id", "default"),
        "metrics": snapshot_provider_metrics(),
    }


@router.get("/models")
async def chat_models(req: Request):
    """
    Return chat/context model choices for the UI.

    Discovered provider models are marked available=true. Presets remain
    selectable when the provider catalog is unreachable from the API process.
    """
    return {
        "tenant": getattr(req.state, "tenant_id", "default"),
        "models": await LLMClient().model_catalog(),
    }


@router.get("/memory/ideas")
async def memory_ideas(req: Request, session_id: Optional[str] = None, limit: int = 20):
    """
    Return compact idea-index diagnostics for the tenant.

    This is read-only debug output for tracking memory behavior across turns.
    """
    tenant: str = getattr(req.state, "tenant_id", "default")
    bounded_limit = max(1, min(limit, 100))
    items = await MemoryStore(tenant).debug_ideas(session_id=session_id, limit=bounded_limit)
    return {
        "tenant": tenant,
        "session_id": session_id,
        "count": len(items),
        "items": items,
    }


@router.get("/memory/action-values")
async def memory_action_values(
    req: Request,
    session_id: Optional[str] = None,
    state_key: Optional[str] = None,
    limit: int = 20,
):
    """
    Return compact Q-like action-value telemetry for diagnostics.
    """
    tenant: str = getattr(req.state, "tenant_id", "default")
    bounded_limit = max(1, min(limit, 100))
    items = await MemoryStore(tenant).debug_action_values(
        session_id=session_id,
        state_key=state_key,
        limit=bounded_limit,
    )
    return {
        "tenant": tenant,
        "session_id": session_id,
        "state_key": state_key,
        "count": len(items),
        "items": items,
    }


@router.get("/memory/evidence-tables")
async def memory_evidence_tables(req: Request, session_id: Optional[str] = None, limit: int = 10):
    """
    Return stored evidence-table diagnostics for recent turns.
    """
    tenant: str = getattr(req.state, "tenant_id", "default")
    bounded_limit = max(1, min(limit, 50))
    items = await MemoryStore(tenant).evidence_tables(session_id=session_id, limit=bounded_limit)
    return {
        "tenant": tenant,
        "session_id": session_id,
        "count": len(items),
        "items": items,
    }


@router.get("/memory/search-notes")
async def memory_search_notes(req: Request, session_id: str, limit: int = 10):
    """
    Return recent auto-context search policy notes for a session.
    """
    tenant: str = getattr(req.state, "tenant_id", "default")
    bounded_limit = max(1, min(limit, 50))
    items = await MemoryStore(tenant).search_policy_notes(session_id=session_id, limit=bounded_limit)
    return {
        "tenant": tenant,
        "session_id": session_id,
        "count": len(items),
        "items": items,
    }


@router.post("/", response_class=StreamingResponse)
async def chat(req: Request, body: ChatRequest):
    """
    Chat endpoint with pinned context.
    Streams SSE events:
      - {"type": "token", "data": "<text>"}
      - {"type": "citations", "data": {...}}
      - {"type": "final", "data": {"done": true, "session_id": "..."}}
    """
    tenant: str = getattr(req.state, "tenant_id", "default")
    
    # Session handling
    session_id = body.session_id or str(uuid.uuid4())
    started_at = time.monotonic()
    correction_only_turn = _is_scope_or_memory_correction_only(body.message)
    
    preliminary_answer_mode = _answer_mode(body.message, {}, correction_only_turn=correction_only_turn)
    diagnostic_or_correction_mode = preliminary_answer_mode in {"diagnostic_trace_answer", "correction_acknowledgement"}

    # Resolve pinned selections to exact snippets
    pinned_items = []
    if body.items:
        # Convert Pydantic models to dicts
        items_as_dicts = [item.model_dump() for item in body.items]
        
        if fetch_pinned_snippets:
            pinned_items = await fetch_pinned_snippets(tenant, items_as_dicts)
        else:
            # Fallback: use items as-is
            pinned_items = items_as_dicts

    auto_context_snippets: List[Dict[str, Any]] = []
    auto_context_plan: Dict[str, Any] = {}
    if (
        not pinned_items
        and settings.memory.enabled
        and body.options.allow_auto_context
        and settings.memory.auto_context_enabled
        and not correction_only_turn
        and not diagnostic_or_correction_mode
    ):
        try:
            auto_payload = await build_auto_context(
                tenant=tenant,
                session_id=session_id,
                message=body.message,
                store=MemoryStore(tenant),
                selected_context_count=0,
                confidence_min=body.options.confidence_min,
                llm_provider=body.options.context_provider,
                llm_model=body.options.context_model,
                llm_api_format=body.options.context_api_format,
            )
            auto_context_snippets = auto_payload.get("snippets", []) or []
            auto_context_plan = auto_payload.get("plan", {}) or {}
            pinned_items = auto_context_snippets
        except Exception as e:
            print(f"[WARN] auto-context build failed: {e}")
    
    # Build prompt and citations
    citations = {}
    
    if build_prompt_and_citations:
        # Use advanced builder - pass options as a dict
        options_dict = {
            "allow_extra_retrieval": bool(body.options.allow_extra_retrieval and not auto_context_snippets and not diagnostic_or_correction_mode),
            "token_budget_ratio": settings.memory.token_budget_ratio,
            "include_spo_when_available": True,
        }
        prompt, citations, meta = await build_prompt_and_citations(
            tenant=tenant,
            message=body.message,
            pinned=pinned_items,
            options=options_dict,
        )
    elif build_prompt:
        # Use simple builder
        prompt = await build_prompt(
            message=body.message,
            snippets=pinned_items,
        )
        citations = {"snippets": pinned_items}
    else:
        # Minimal fallback
        context = "\n".join([s.get("text", "") for s in pinned_items])
        prompt = f"Context:\n{context}\n\nQuestion: {body.message}\n\nAnswer:"
        citations = {"snippets": pinned_items}
    evidence_assembly_context = str(
        (auto_context_plan.get("evidence_assembly") or {}).get("prompt_context") or ""
    ).strip()
    if evidence_assembly_context:
        prompt = f"{evidence_assembly_context}\n\nGrounded task prompt:\n{prompt}"
    active_evidence_assembly = (auto_context_plan.get("evidence_assembly") or {}) if auto_context_plan else {}
    # P-4 (default off): let question-type answer modes reconsider the resolved
    # query for context-poor replies. No-op unless answer_mode_consider_resolved_query.
    _resolved_query = None
    if settings.memory.answer_mode_consider_resolved_query:
        _resolved_query = ((auto_context_plan or {}).get("intent_resolution") or {}).get("effective_query") or None
    answer_mode = preliminary_answer_mode if diagnostic_or_correction_mode else _answer_mode(body.message, active_evidence_assembly, correction_only_turn=correction_only_turn, resolved_query=_resolved_query)
    answer_mode_contract = ANSWER_MODE_CONTRACTS.get(answer_mode, ANSWER_MODE_CONTRACTS["direct_answer"])
    answer_mode_context = _answer_mode_prompt(answer_mode, active_evidence_assembly)
    if not correction_only_turn:
        prompt = f"{answer_mode_context}\n\nGrounded task prompt:\n{prompt}"

    # Inference-time memory policy. This is not weight training; it selects a
    # working set, tracks rewards, and stores trajectories for later learning.
    context_plan = None
    if settings.memory.enabled and body.options.allow_memory:
        allow_web = (
            body.options.allow_web_search
            if body.options.allow_web_search is not None
            else settings.memory.allow_web_search_default
        )
        policy = ContextPolicy(tenant)
        context_plan = await policy.plan(
            session_id=session_id,
            message=body.message,
            allow_web_search=bool(allow_web and not diagnostic_or_correction_mode),
            confidence_min=body.options.confidence_min,
            evidence_assembly=active_evidence_assembly,
            gap_spec=(auto_context_plan or {}).get("gap_spec"),  # WP-B: pass GapSpec to steer retries
        )
        if context_plan.context_prefix:
            prompt = f"{context_plan.context_prefix}\n\nGrounded task prompt:\n{prompt}"
        if context_plan.warnings:
            warning_block = "\n".join(f"- {w}" for w in context_plan.warnings)
            prompt = f"Potential consistency warnings:\n{warning_block}\n\n{prompt}"
    external_grounding_covers_puzzle = _external_grounding_covers_puzzle(active_evidence_assembly, context_plan.web_results if context_plan else [])
    if external_grounding_covers_puzzle and answer_mode == "clarification":
        answer_mode = "direct_answer"
        answer_mode_contract = ANSWER_MODE_CONTRACTS[answer_mode]
        prompt = (
            "Answer-mode override: privacy-filtered external biomedical grounding covers the missing local evidence puzzle nodes. "
            "Answer directly from the combined local and external context, keep provenance explicit, and do not ask for clarification solely because local snippets were sparse.\n\n"
            f"Grounded task prompt:\n{prompt}"
        )
    
    # Initialize LLM client
    llm = LLMClient()
    
    async def gen():
        """
        Stream tokens as SSE events.
        Events:
          - {"type": "token", "data": "<partial text>"}
          - {"type": "citations", "data": {...}}
          - {"type": "final", "data": {"done": true, "session_id": "..."}}
        """
        nonlocal citations
        messages = [
            {
                "role": "system",
                "content": answer_system_prompt(answer_mode),
            },
        ]
        native_history = _native_history_messages(
            context_plan,
            limit=max(0, settings.memory.working_buffer_turns * 2),
        )
        messages.extend(native_history)
        messages.append({"role": "user", "content": prompt})
        prompt_context_hash = _prompt_hash(evidence_assembly_context)
        prompt_snapshot_hash = _prompt_hash(messages[0]["content"], prompt, answer_mode)
        prompt_token_estimate = max(1, (len(messages[0]["content"]) + len(prompt)) // 4)
        generation_telemetry = {
            "answer_mode": answer_mode,
            "answer_mode_contract": answer_mode_contract,
            "prompt_hash": prompt_snapshot_hash,
            "prompt_context_hash": prompt_context_hash,
            "context_window": {
                "max_input_tokens": settings.llm.max_input_tokens,
                "context_token_budget": max(256, int(settings.llm.max_input_tokens * settings.memory.token_budget_ratio)),
                "memory_working_buffer_token_budget": settings.memory.working_buffer_token_budget,
                "memory_token_budget_ratio": settings.memory.token_budget_ratio,
                "prompt_token_estimate": prompt_token_estimate,
                "nvidia_max_output_tokens": settings.llm.nvidia_max_tokens,
                "provider_timeout_sec": settings.llm.provider_timeout_sec,
            },
            "external_grounding_covers_puzzle": external_grounding_covers_puzzle,
            "puzzle_state": {
                **(((active_evidence_assembly or {}).get("evidence_puzzle")) or {}),
                "clarification_recommended": bool((active_evidence_assembly or {}).get("clarification_recommended")),
            },
        }

        if context_plan:
            context_plan.meta["native_history_message_count"] = len(native_history)
            for warning in context_plan.warnings:
                yield {"type": "warning", "data": {"message": warning}}
            if body.options.expose_memory_debug:
                yield {"type": "memory_debug", "data": context_plan.meta}
        
        answer_parts: List[str] = []
        hold_for_clarification = (not external_grounding_covers_puzzle) and _hold_generation_for_clarification(
            (auto_context_plan.get("evidence_assembly") or {}) if auto_context_plan else {}
        )
        opening_prefix = "" if not hold_for_clarification else _opening_clarification_prefix(
            (auto_context_plan.get("evidence_assembly") or {}) if auto_context_plan else {}
        )
        if opening_prefix:
            answer_parts.append(opening_prefix)
            yield {"type": "token", "data": opening_prefix}
        guard_generation = _needs_post_generation_guard(answer_mode, active_evidence_assembly)
        post_generation_trace: Dict[str, Any] = {
            "applied": False,
            "accepted_claims": [],
            "repaired_claims": [],
            "blocked_claims": [],
        }
        # Stream model tokens - parse OpenAI JSON chunks
        if correction_only_turn:
            correction_ack = _correction_acknowledgement(body.message)
            answer_parts.append(correction_ack)
            yield {"type": "token", "data": correction_ack}
        elif not hold_for_clarification:
            chat_kwargs = {
                key: value
                for key, value in {
                    "provider": body.options.chat_provider,
                    "model": body.options.chat_model,
                    "api_format": body.options.chat_api_format,
                }.items()
                if value
            }
            provider_error: Dict[str, Any] | None = None
            try:
                async with asyncio.timeout(settings.llm.provider_timeout_sec):
                    async for chunk in llm.chat_stream(messages, agent="answer", **chat_kwargs):
                        # Parse OpenAI-format JSON chunk
                        if chunk == "[DONE]":
                            break
                        
                        try:
                            data = json.loads(chunk)
                            
                            # Extract text content from OpenAI format
                            if "choices" in data and len(data["choices"]) > 0:
                                delta = data["choices"][0].get("delta", {})
                                content = delta.get("content", "")
                                if content:
                                    answer_parts.append(content)
                                    if not guard_generation:
                                        yield {"type": "token", "data": content}
                        except json.JSONDecodeError:
                            # Skip malformed chunks
                            continue
            except Exception as exc:
                provider_error = _provider_error_payload(exc)
                generation_telemetry["provider_error"] = provider_error
                yield {"type": "warning", "data": provider_error}
                if not "".join(answer_parts).strip():
                    fallback = _evidence_only_fallback_answer(
                        citations=(citations if isinstance(citations, dict) else {}),
                        context_plan=context_plan,
                        auto_context_plan=auto_context_plan,
                    )
                    answer_parts = [fallback]
                    yield {"type": "token", "data": fallback}
            if not provider_error and not "".join(answer_parts).strip():
                generation_telemetry["empty_stream_fallback"] = {
                    "message": "Hosted chat stream completed without textual token content; emitted evidence-only fallback.",
                    "answer_mode": answer_mode,
                }
                fallback = _evidence_only_fallback_answer(
                    citations=(citations if isinstance(citations, dict) else {}),
                    context_plan=context_plan,
                    auto_context_plan=auto_context_plan,
                )
                answer_parts = [fallback]
                if not guard_generation:
                    yield {"type": "token", "data": fallback}
            if guard_generation:
                repaired, post_generation_trace = _post_generation_expansion_guard(
                    "".join(answer_parts),
                    answer_mode=answer_mode,
                    evidence_assembly=active_evidence_assembly,
                    source_snippets=(citations.get("snippets", []) if isinstance(citations, dict) else []),
                )
                if post_generation_trace.get("applied"):
                    answer_parts = [repaired]
                    yield {"type": "semantic_drift_trace", "data": post_generation_trace}
                    yield {"type": "token", "data": repaired}
        
        # Send citations after completion
        if context_plan:
            citations_payload = citations if isinstance(citations, dict) else {"snippets": citations}
            citations_payload["memory"] = {
                "selected_context_count": len(context_plan.selected_context),
                "idea_count": context_plan.meta.get("idea_count", 0),
                "triplet_count": len(context_plan.retrieved_triplets),
                "web_result_count": len(context_plan.web_results),
                "auto_context_result_count": len(auto_context_snippets),
                "auto_context_query_count": len(auto_context_plan.get("variants", []) or []),
                "auto_context_used_llm": bool(auto_context_plan.get("used_llm", False)),
                "warnings": context_plan.warnings,
            }
            generation_telemetry["post_generation_guard"] = post_generation_trace
            citations_payload["generation_telemetry"] = generation_telemetry
            if context_plan.web_results:
                citations_payload["web_context"] = context_plan.web_results
            if auto_context_plan:
                citations_payload["auto_context"] = {
                    "state_key": auto_context_plan.get("state_key"),
                    "action_key": auto_context_plan.get("action_key"),
                    "strategy": auto_context_plan.get("strategy"),
                    "search_frame": auto_context_plan.get("search_frame", {}),
                    "result_count": auto_context_plan.get("result_count", 0),
                    "skipped_off_topic_count": auto_context_plan.get("skipped_off_topic_count", 0),
                    "levels": auto_context_plan.get("levels", []),
                    "level_reports": auto_context_plan.get("level_reports", []),
                    "retrieval_records": auto_context_plan.get("retrieval_records", []),
                    "evidence_assembly": {
                        key: value
                        for key, value in (auto_context_plan.get("evidence_assembly") or {}).items()
                        if key != "prompt_context"
                    },
                    "answer_mode": answer_mode,
                    "prompt_hash": prompt_snapshot_hash,
                    "prompt_context_hash": prompt_context_hash,
                    "query_labels": auto_context_plan.get("query_labels", []),
                    "used_llm": auto_context_plan.get("used_llm", False),
                    # WP-B/D: GapSpec and per-step rewards
                    "gap_spec": auto_context_plan.get("gap_spec", {}),
                    "step_rewards": auto_context_plan.get("step_rewards", []),
                }
            citations_payload["retrieval_pipeline"] = _retrieval_pipeline_trace(
                auto_context_plan=auto_context_plan,
                context_plan=context_plan,
                answer_mode=answer_mode,
                external_grounding_covers_puzzle=external_grounding_covers_puzzle,
            )
            citations = citations_payload

        yield {"type": "citations", "data": citations}

        if context_plan:
            answer_text = "".join(answer_parts)
            observed_context = list(context_plan.selected_context)
            observed_context.extend({"source": "auto_context", **item} for item in auto_context_snippets)
            _observe_kwargs = dict(
                session_id=session_id,
                turn_index=context_plan.turn_index,
                question=body.message,
                answer=answer_text,
                selected_context=observed_context,
                retrieved_triplets=context_plan.retrieved_triplets,
                pinned_snippets=pinned_items,
                source_sentences=(citations.get("snippets", []) if isinstance(citations, dict) else []),
                search_plan={**(auto_context_plan or {}), "answer_mode": answer_mode, "prompt_hash": prompt_snapshot_hash, "prompt_context_hash": prompt_context_hash},
                started_at=started_at,
                token_budget=body.options.token_budget,
            )
            # Move observe_turn off the SSE critical path so HF NLI cold-starts
            # (up to 45 s) do not block the stream between citations and final.
            # Consistency warnings from this turn will surface in the next turn
            # via memory retrieval, or can be fetched from /chat/memory/evidence-tables.
            asyncio.create_task(policy.observe_turn(**_observe_kwargs))

        # Final event ? emitted immediately after citations
        yield {"type": "final", "data": {"done": True, "session_id": session_id}}
    
    # Use SSE helper
    headers = {
        "X-Chat-Session-Id": session_id,
        "Cache-Control": "no-store",
    }
    
    return await sse_stream(gen(), heartbeat=settings.app.sse_heartbeat_seconds, headers=headers)
