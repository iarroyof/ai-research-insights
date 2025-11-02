from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from app.search.hybrid import hybrid_search_generic  # reuse your RRF util, parametrize index/fields

router = APIRouter(prefix="/chat/history", tags=["chat-history"])

class ChatSearchReq(BaseModel):
    q: str = Field(min_length=2, max_length=2000)
    session_id: Optional[str] = None
    role: Optional[str] = Field(default=None, pattern="^(user|assistant)$")
    k: int = Field(default=20, ge=1, le=200)

class ChatHit(BaseModel):
    session_id: str
    message_id: str
    anchor_id: str
    turn_index: int
    role: str
    snippet: str
    score: float
    created_at: Optional[str] = None
    session_title: Optional[str] = None

class ChatSearchResp(BaseModel):
    hits: List[ChatHit]

@router.post("/search", response_model=ChatSearchResp)
async def search_history(req: Request, body: ChatSearchReq):
    tenant = getattr(getattr(req, "state", None), "tenant_id", None) or "default"
    idx = f"{settings.search.index_prefix}{tenant}_chats"  # e.g., t_default_chats

    filters: Dict[str, Any] = {}
    if body.session_id:
        filters["session_id"] = body.session_id
    if body.role:
        filters["role"] = body.role

    # hybrid_search_generic should accept: index, query, filters, fields, k
    fields = {"text": "content", "vector": "vec"}  # map abstraction for the chats index
    hits = await hybrid_search_generic(index=idx, query=body.q, filters=filters, fields=fields, k=body.k)

    out: List[ChatHit] = []
    for h in hits:
        src = h.get("_source", h)
        out.append(ChatHit(
            session_id=src.get("session_id"),
            message_id=src.get("message_id"),
            anchor_id=src.get("anchor_id"),
            turn_index=src.get("turn_index", 0),
            role=src.get("role", "assistant"),
            snippet=src.get("content", "")[:500],
            score=h.get("_score", 0.0),
            created_at=src.get("created_at"),
            session_title=src.get("session_title"),
        ))
    return ChatSearchResp(hits=out)

