# services/api/app/search/os_client.py

from __future__ import annotations
from typing import List, Dict, Any
from opensearchpy import OpenSearch
from app.config import settings

_client = None

def os_client() -> OpenSearch:
    global _client
    if _client is None:
        _client = OpenSearch(
            hosts=[settings.opensearch.endpoint],
            verify_certs=False
        )
    return _client

def os_hybrid_query(tenant: str, query: str, filters: Dict[str, Any], k: int) -> List[Dict[str, Any]]:
    """
    Performs BM25 text search in OpenSearch. (Vector & RRF hooks to come later.)
    """
    idx = f"{settings.opensearch.index_prefix}{tenant}_chunks"
    body = {
        "size": k,
        "query": {"match": {"text": {"query": query}}},
        "_source": ["paper_id", "title", "pmid", "pmcid", "page", "sent_id", "text"]
    }
    res = os_client().search(index=idx, body=body)
    hits = []
    for h in res["hits"]["hits"]:
        src = h["_source"]
        hits.append({
            "paper_id": src.get("paper_id"),
            "title": src.get("title"),
            "pmid": src.get("pmid"),
            "pmcid": src.get("pmcid"),
            "page": src.get("page"),
            "sent_id": src.get("sent_id"),
            "text": src.get("text"),
            "score": h["_score"]
        })
    return hits
