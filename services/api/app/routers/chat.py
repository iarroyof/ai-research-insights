# services/api/app/routers/chat.py
from __future__ import annotations

import uuid
from typing import List, Optional, Dict, Any

from fastapi import APIRouter, Request
from starlette.responses import StreamingResponse

from pydantic import BaseModel, Field

# Config
from app.config import settings

# LLM client (your existing wrapper)
from app.clients.llm import LLMClient

# SSE helper (your existing helper that builds a StreamingResponse)
from app.utils.sse import sse_stream

# Pinned-context resolution + prompt/citation builders.
# We try newer helpers first; fall back to your existing simple builder.
try:
    # Preferred (newer) helper locations
    from app.rag.context import fetch_pinned_snippets  # resolves (paper_id, sent_id) -> text + meta
except Exception:  # pragma: no cover
    fetch_pinned_snippets = None  # will use a local fallback

try:
    from app.rag.citations import build_prompt_and_citations  # returns (prompt, citations, meta)
except Exception:  # pragma: no cover
    build_prompt_and_citations = None

try:
    # legacy simple builder (already present in your repo)
    from app.rag.citations import build_prompt  # type: ignore
except Exception:  # pragma: no cover
    build_prompt = None


router = APIRouter(prefix="/chat", tags=["chat"])


# ---------- Request/response models (local to this router for now) ----------
class ChatItem(BaseModel):
    paper_id: str
    sent_id: Optional[int] = None


class ChatRequest(BaseModel):
    session_id: Optional[str] = Field(default=None, description="Provide to continue a session; omit to start new.")
    message: str = Field(min_length=1, max_length=8000)
    items: List[ChatItem] = Field(default_factory=list, description="Pinned context from Search selections.")
    options: Dict[str, Any] = Field(
        default_factory=dict,
        description="Optional flags, e.g. {'allow_extra_retrieval': true}"
    )


# ---------- Local fallbacks (safe defaults if newer helpers aren’t present) ----------
async def _fallback_fetch_pinned_snippets(tenant_id: str, items: List[ChatItem]) -> List[Dict[str, Any]]:
    """
    Fallback: returns minimal placeholder snippets.
    In production, replace by fetching (paper_id, sent_id) from your DB/OpenSearch to get:
      - text, pmid/pmcid, page, char spans, bbox, etc.
    """
    out: List[Dict[str, Any]] = []
    for it in items:
        out.append(
            {
                "paper_id": it.paper_id,
                "sent_id": it.sent_id,
                "text": "...snippet...",   # TODO: real sentence text
                "pmid": None,
                "pmcid": None,
                "page": None,
                "bbox": None,
            }
        )
    return out


async def _fallback_build_prompt_and_citations(
    tenant: str,
    message: str,
    pinned: List[Dict[str, Any]],
    options: Dict[str, Any],
):
    """
    Fallback: builds a simple prompt and empty citations block.
    """
    # If legacy build_prompt exists, use it so behavior matches your previous code.
    if build_prompt is not None:
        prompt = build_prompt(message, pinned)
    else:
        # Minimal prompt if nothing else is available
        preface = "You are a helpful and concise assistant.\n"
        ctx = "\n\n".join([f"- {s.get('text', '')}" for s in pinned]) or "(no pinned context)"
        prompt = f"{preface}\nContext:\n{ctx}\n\nUser: {message}\nAnswer:"
    citations: Dict[str, Any] = {
        "snippets": pinned,  # echo what we used so the client can render sources
        "meta": {"tenant": tenant},
    }
    meta: Dict[str, Any] = {}
    return prompt, citations, meta


# ---------- Chat endpoint (SSE streaming) ----------
@router.post("/", response_class=StreamingResponse)
async def chat(req: Request, body: ChatRequest):
    tenant: str = getattr(req.state, "tenant_id", "default")

    # Session handling: reuse incoming or generate a fresh one
    session_id = body.session_id or str(uuid.uuid4())

    # Resolve pinned selections to exact snippets (text + metadata)
    if fetch_pinned_snippets is not None:
        snippets = await fetch_pinned_snippets(tenant, body.items)
    else:
        snippets = await _fallback_fetch_pinned_snippets(tenant, body.items)

    # Build prompt (and capture citations for the client to render after the stream)
    if build_prompt_and_citations is not None:
        prompt, citations, _meta = await build_prompt_and_citations(
            tenant=tenant, message=body.message, pinned=snippets, options=body.options
        )
    else:
        prompt, citations, _meta = await _fallback_build_prompt_and_citations(
            tenant=tenant, message=body.message, pinned=snippets, options=body.options
        )

    # Prepare LLM client
    llm = LLMClient()

    async def gen():
        """
        Stream tokens as SSE events.
        Events:
          - {"type": "token", "data": "<partial text>"}
          - {"type": "citations", "data": {...}}    # at the end, sources & metadata
          - {"type": "final", "data": {"done": true}}
        """
        messages = [
            {"role": "system", "content": "You are helpful and concise."},
            {"role": "user", "content": prompt},
        ]

        # Stream model tokens
        async for chunk in llm.chat_stream(messages):
            # `chunk` should be a string token delta; adapt if your client yields dicts
            yield {"type": "token", "data": chunk}

        # Send citations after completion
        yield {"type": "citations", "data": citations}

        # Final event
        yield {"type": "final", "data": {"done": True, "session_id": session_id}}

    # Use your SSE helper; include session header for the Streamlit client to capture
    headers = {
        "X-Chat-Session-Id": session_id,
        "Cache-Control": "no-store",
    }
    return await sse_stream(gen(), heartbeat=settings.app.sse_heartbeat_seconds, headers=headers)

