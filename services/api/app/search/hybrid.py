# services/api/app/search/hybrid.py

from __future__ import annotations
from typing import List, Dict, Any
from app.search.os_client import os_hybrid_query, os_multilevel_query

async def hybrid_search_sentences(tenant: str, query: str, filters: Dict[str, Any], k: int) -> List[Dict[str, Any]]:
    """
    Unified entrypoint for hybrid search.
    Currently just delegates to OpenSearch BM25 search.
    """
    # os_hybrid_query is synchronous, so run in threadpool
    import asyncio
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, os_hybrid_query, tenant, query, filters, k)


async def hybrid_search_multilevel(tenant: str, level: str, query: str, filters: Dict[str, Any], k: int) -> List[Dict[str, Any]]:
    """
    Structured retrieval entrypoint for auto-context search.

    BM25 remains primary at every level. The lower-level OpenSearch client may
    append vector fallback hits only when the caller supplies query vectors and
    BM25 is sparse/low-confidence.
    """
    import asyncio
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, os_multilevel_query, tenant, level, query, filters, k)
