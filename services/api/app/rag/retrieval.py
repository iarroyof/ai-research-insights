from typing import Dict, List
from app.clients.opensearch import _os, CHUNKS, TRIPLETS
from app.config import settings
from .fusion import rrf_fuse

def keyword_search(tenant: str, q: str, target: str, size: int = 50):
    index = CHUNKS(tenant) if target in ("papers","chunks","all") else TRIPLETS(tenant)
    body = {
        "query": {"multi_match": {"query": q, "fields": [
            "text^2", "sentence_text", "subject", "predicate", "object", "title"
        ]}},
        "size": size
    }
    return _os.search(index=index, body=body)["hits"]["hits"]

def vector_search_disabled(*args, **kwargs):
    # Vector path can be enabled once vectors are indexed.
    return []

def hybrid_search(tenant: str, q: str, target: str = "all", size: int = 50):
    bm25 = keyword_search(tenant, q, target, size)
    vec = vector_search_disabled()
    fused = rrf_fuse(bm25, vec, k=size) if vec else bm25
    return fused
