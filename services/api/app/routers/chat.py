# services/api/app/routers/chat.py
import uuid
import json
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Request
from starlette.responses import StreamingResponse
from pydantic import BaseModel, Field

# Config
from app.config import settings

# LLM client
from app.clients.llm import LLMClient

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
    token_budget: int = Field(default=2000)
    confidence_min: float = Field(default=0.5)


class ChatRequest(BaseModel):
    message: str
    items: List[ChatItem] = Field(default_factory=list)
    session_id: Optional[str] = None
    options: ChatOptions = Field(default_factory=ChatOptions)


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
    
    # Build prompt and citations
    citations = {}
    
    if build_prompt_and_citations:
        # Use advanced builder - pass options as a dict
        options_dict = {
            "allow_extra_retrieval": body.options.allow_extra_retrieval,
            "token_budget_ratio": 0.6,  # or calculate from body.options.token_budget
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
        messages = [
            {"role": "system", "content": "You are a helpful research assistant. Answer based on the provided context."},
            {"role": "user", "content": prompt},
        ]
        
        # Stream model tokens - parse OpenAI JSON chunks
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
                        yield {"type": "token", "data": content}
            except json.JSONDecodeError:
                # Skip malformed chunks
                continue
        
        # Send citations after completion
        yield {"type": "citations", "data": citations}
        
        # Final event
        yield {"type": "final", "data": {"done": True, "session_id": session_id}}
    
    # Use SSE helper
    headers = {
        "X-Chat-Session-Id": session_id,
        "Cache-Control": "no-store",
    }
    
    return await sse_stream(gen(), heartbeat=settings.app.sse_heartbeat_seconds, headers=headers)
