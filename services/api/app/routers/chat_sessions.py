from fastapi import APIRouter, Request, HTTPException, Query
from typing import Optional
from app.db.chat_repo import fetch_messages_window  # your repo function

router = APIRouter(prefix="/chat/session", tags=["chat-history"])

@router.get("/{session_id}/messages")
async def get_messages(req: Request, session_id: str, focus: Optional[str] = None, window: int = Query(40, ge=10, le=200)):
    tenant = getattr(getattr(req, "state", None), "tenant_id", None) or "default"
    # repo enforces tenant/user ownership
    data = await fetch_messages_window(tenant, session_id, focus_anchor_id=focus, window=window)
    return {"focus_anchor_id": focus, "messages": data}

