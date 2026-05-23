# services/api/app/routers/chat.py
import uuid
import json
import time
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Request
from starlette.responses import StreamingResponse
from pydantic import BaseModel, Field

# Config
from app.config import settings

# LLM client
from app.clients.llm import LLMClient
from app.memory.policy import ContextPolicy
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
    return (
        "I will assemble the supported evidence pieces first and keep missing links explicit. "
        f"To refine the explanation, which interpretation should lead: {frame_text}?\n\n"
    )


def _hold_generation_for_clarification(evidence_assembly: Dict[str, Any] | None) -> bool:
    assembly = evidence_assembly or {}
    if not assembly.get("clarification_recommended"):
        return False
    edge_status = ((assembly.get("evidence_puzzle") or {}).get("edge_support_status") or "").lower()
    return edge_status in {"missing", "partial"}


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
    ):
        try:
            auto_payload = await build_auto_context(
                tenant=tenant,
                session_id=session_id,
                message=body.message,
                store=MemoryStore(tenant),
                selected_context_count=0,
                confidence_min=body.options.confidence_min,
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
            "allow_extra_retrieval": bool(body.options.allow_extra_retrieval and not auto_context_snippets),
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
            allow_web_search=bool(allow_web),
            confidence_min=body.options.confidence_min,
        )
        if context_plan.context_prefix:
            prompt = f"{context_plan.context_prefix}\n\nGrounded task prompt:\n{prompt}"
        if context_plan.warnings:
            warning_block = "\n".join(f"- {w}" for w in context_plan.warnings)
            prompt = f"Potential consistency warnings:\n{warning_block}\n\n{prompt}"
    
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
            {"role": "system", "content": "You are a helpful research assistant. Answer based on the provided context."},
        ]
        native_history = _native_history_messages(
            context_plan,
            limit=max(0, settings.memory.working_buffer_turns * 2),
        )
        messages.extend(native_history)
        messages.append({"role": "user", "content": prompt})

        if context_plan:
            context_plan.meta["native_history_message_count"] = len(native_history)
            for warning in context_plan.warnings:
                yield {"type": "warning", "data": {"message": warning}}
            if body.options.expose_memory_debug:
                yield {"type": "memory_debug", "data": context_plan.meta}
        
        answer_parts: List[str] = []
        opening_prefix = _opening_clarification_prefix(
            (auto_context_plan.get("evidence_assembly") or {}) if auto_context_plan else {}
        )
        if opening_prefix:
            answer_parts.append(opening_prefix)
            yield {"type": "token", "data": opening_prefix}
        hold_for_clarification = _hold_generation_for_clarification(
            (auto_context_plan.get("evidence_assembly") or {}) if auto_context_plan else {}
        )
        # Stream model tokens - parse OpenAI JSON chunks
        if not hold_for_clarification:
            async for chunk in llm.chat_stream(messages):
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
                            yield {"type": "token", "data": content}
                except json.JSONDecodeError:
                    # Skip malformed chunks
                    continue
        
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
                    "evidence_assembly": {
                        key: value
                        for key, value in (auto_context_plan.get("evidence_assembly") or {}).items()
                        if key != "prompt_context"
                    },
                    "query_labels": auto_context_plan.get("query_labels", []),
                    "used_llm": auto_context_plan.get("used_llm", False),
                }
            citations = citations_payload

        yield {"type": "citations", "data": citations}

        if context_plan:
            answer_text = "".join(answer_parts)
            try:
                observed_context = list(context_plan.selected_context)
                observed_context.extend({"source": "auto_context", **item} for item in auto_context_snippets)
                trace = await ContextPolicy(tenant).observe_turn(
                    session_id=session_id,
                    turn_index=context_plan.turn_index,
                    question=body.message,
                    answer=answer_text,
                    selected_context=observed_context,
                    retrieved_triplets=context_plan.retrieved_triplets,
                    pinned_snippets=pinned_items,
                    source_sentences=(citations.get("snippets", []) if isinstance(citations, dict) else []),
                    search_plan=auto_context_plan,
                    started_at=started_at,
                    token_budget=body.options.token_budget,
                )
                if trace.get("conflicts"):
                    yield {
                        "type": "consistency_warning",
                        "data": {
                            "message": "The generated answer may conflict with retrieved triplet memory. Please confirm which fact should be treated as authoritative.",
                            "conflicts": trace.get("conflicts", [])[:2],
                        },
                    }
                nli_evidence = trace.get("nli_evidence", [])
                nli_contradictions = [
                    e for e in nli_evidence
                    if float(e.get("contradiction", 0.0) or 0.0) >= settings.memory.nli_contradiction_threshold
                ]
                if nli_contradictions:
                    yield {
                        "type": "consistency_warning",
                        "data": {
                            "message": "Biomedical NLI evidence suggests that one or more answer claims may contradict their source sentences.",
                            "nli_evidence": nli_contradictions[:2],
                        },
                    }
                claim_support = trace.get("claim_support", [])
                claim_contradictions = [
                    c for c in claim_support
                    if c.get("status") == "contradicted" or c.get("needs_user_confirmation")
                ]
                if claim_contradictions:
                    yield {
                        "type": "consistency_warning",
                        "data": {
                            "message": "Source-sentence factuality checks suggest that one or more answer claims may contradict available evidence. Please confirm which fact should be treated as authoritative.",
                            "claims": claim_contradictions[:2],
                        },
                    }
                longitudinal_warnings = (trace.get("longitudinal_consistency", {}) or {}).get("warnings", [])
                if longitudinal_warnings:
                    yield {
                        "type": "consistency_warning",
                        "data": {
                            "message": "Cross-turn memory consistency checks found possible drift or conflicts with prior evidence-supported conversation memory.",
                            "warnings": longitudinal_warnings[:3],
                        },
                    }
                if body.options.expose_memory_debug:
                    yield {"type": "reward", "data": trace.get("reward", {})}
                    yield {"type": "evidence_table", "data": trace.get("evidence_table", {})}
                    yield {"type": "conversation_frame", "data": trace.get("conversation_frame", {})}
            except Exception as e:
                yield {"type": "warning", "data": {"message": f"Memory trace update failed: {e}"}}
        
        # Final event
        yield {"type": "final", "data": {"done": True, "session_id": session_id}}
    
    # Use SSE helper
    headers = {
        "X-Chat-Session-Id": session_id,
        "Cache-Control": "no-store",
    }
    
    return await sse_stream(gen(), heartbeat=settings.app.sse_heartbeat_seconds, headers=headers)
