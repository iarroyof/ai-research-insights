# services/api/app/routers/search.py
from __future__ import annotations

from typing import Optional, List, Any, Dict, Callable, Awaitable
from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel, Field
import anyio

router = APIRouter(prefix="/search", tags=["search"], include_in_schema=True, redirect_slashes=False)

# --- Try the new search backend first; fall back to legacy ---
_search_new: Optional[Callable[..., Awaitable[List[Dict[str, Any]]]]] = None
_search_legacy: Optional[Callable[..., List[Dict[str, Any]]]] = None

try:
    # Preferred modern module path
    from app.search.hybrid import hybrid_search_sentences as _modern_search  # type: ignore
    async def _call_new(tenant: str, query: str, filters: Dict[str, Any], k: int) -> List[Dict[str, Any]]:
        return await _modern_search(tenant, query, filters, k)
    _search_new = _call_new
except Exception:
    _search_new = None

if _search_new is None:
    try:
        # Legacy implementation already in your repo
        from app.rag.retrieval import hybrid_search as _legacy_search  # type: ignore

        async def _call_legacy(tenant: str, query: str, target: str, k: int) -> List[Dict[str, Any]]:
            # legacy signature: (tenant, query, target) -> ES hits (list of dicts)
            # Run sync func in a worker thread to keep endpoint non-blocking
            return await anyio.to_thread.run_sync(_legacy_search, tenant, query, target)
        _search_legacy = _call_legacy
    except Exception:
        _search_legacy = None


# ---------- Models (keep "items" key for backwards-compat) ----------

class SearchRequest(BaseModel):
    query: str = Field(min_length=2, max_length=2000)
    target: str = Field(default="all", pattern="^(papers|triplets|chats|all)$")
    filters: Dict[str, Any] = Field(default_factory=dict)
    k: int = Field(default=20, ge=1, le=200)


class SearchHit(BaseModel):
    paper_id: Optional[str] = None
    title: Optional[str] = None
    pmid: Optional[str] = None
    pmcid: Optional[str] = None
    page: Optional[int] = None
    sent_id: Optional[str] = None  # ✅ FIXED: Changed from int to str for hash IDs
    score: Optional[float] = None
    text: Optional[str] = None
    # ✅ ADDED: Triplet-specific fields
    subject: Optional[str] = None
    relation: Optional[str] = None
    object: Optional[str] = None
    confidence: Optional[float] = None


class SearchResponse(BaseModel):
    # Streamlit expects key "items"; we keep that alias
    hits: List[SearchHit] = Field(default_factory=list, serialization_alias="items")

# ---------- Add Triplet-based fallback backend if none available ----------

if _search_new is None and _search_legacy is None:
    try:
        from app.routers.triplets import triplets_search

        async def _call_triplets(tenant: str, query: str, filters: Dict[str, Any], k: int):
            # Delegate to triplet index search
            results = await triplets_search(
                tenant=tenant,
                q=query,
                confidence_min=filters.get("confidence_min", 0.0),
                entity_type=filters.get("entity_type"),
                pmid=filters.get("pmid"),
            )
            # Normalize to legacy format
            hits = []
            for t in results:
                hits.append({
                    "_source": {
                        "paper_id": t.get("article_id"),
                        "title": t.get("sentence_text", "")[:120],
                        "pmid": t.get("pmid"),
                        "pmcid": None,
                        "page": None,
                        "sent_id": t.get("triple_id"),
                        "score": t.get("confidence", 0),
                        "text": t.get("sentence_text", ""),
                    },
                    "_score": t.get("confidence", 0),
                })
            return hits

        _search_new = _call_triplets
        print("[INFO] Using triplet-based fallback search backend")
    except Exception as e:
        print(f"[WARN] Could not enable triplet fallback backend: {e}")

# ----------------------------- Route ------------------------------

@router.post("/", response_model=SearchResponse)
async def search(req: Request, body: SearchRequest) -> SearchResponse:
    tenant = getattr(getattr(req, "state", None), "tenant_id", None) or "default"

    if _search_new is None and _search_legacy is None:
        raise HTTPException(status_code=500, detail="No search backend available")

    try:
        if _search_new:
            # New backend supports filters & k directly and returns normalized hits
            raw_hits = await _search_new(tenant, body.query, body.filters, body.k)
        else:
            # Legacy backend returns raw ES hits; we'll normalize below
            raw_hits = await _search_legacy(tenant, body.query, body.target, body.k)  # type: ignore

        hits: List[SearchHit] = []
        for h in raw_hits or []:
            # Accept both "normalized" and raw ES hit formats
            if "_source" in h:
                src = h.get("_source", {}) or {}
                score = h.get("_score")
            else:
                src = h or {}
                score = src.get("score")

            # Title field can be under different keys
            title = src.get("title") or src.get("paper_title") or src.get("doc_title")

            # Best-effort sentence text location
            text = (
                src.get("text")
                or src.get("sentence_text")
                or src.get("chunk")
                or src.get("content")
            )

            hits.append(
                SearchHit(
                    paper_id=src.get("paper_id"),
                    title=title,
                    pmid=src.get("pmid"),
                    pmcid=src.get("pmcid"),
                    page=src.get("page"),
                    sent_id=src.get("sent_id"),  # Now accepts string
                    score=score,
                    text=text,
                    # ✅ ADDED: Pass through triplet fields
                    subject=src.get("subject"),
                    relation=src.get("relation"),
                    object=src.get("object"),
                    confidence=src.get("confidence"),
                )
            )

        return SearchResponse(hits=hits)

    except HTTPException:
        raise
    except Exception as e:
        import traceback, sys
        print("[ERROR] Unified search traceback:", file=sys.stderr)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Search failed: {type(e).__name__}: {e}")
