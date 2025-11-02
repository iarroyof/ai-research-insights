# services/api/app/search/hybrid.py

from __future__ import annotations
from typing import List, Dict, Any
from app.search.os_client import os_hybrid_query

async def hybrid_search_sentences(tenant: str, query: str, filters: Dict[str, Any], k: int) -> List[Dict[str, Any]]:
    """
    Unified entrypoint for hybrid search.
    Currently just delegates to OpenSearch BM25 search.
    """
    # os_hybrid_query is synchronous, so run in threadpool
    import asyncio
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, os_hybrid_query, tenant, query, filters, k)

