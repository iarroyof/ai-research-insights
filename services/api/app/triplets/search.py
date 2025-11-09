# services/api/app/triplets/search.py
from __future__ import annotations
import json  # ADDED: Missing import
from app.search.os_client import os_client
from app.config import settings

async def search_triplets(  # FIXED: Function name matches import
    tenant: str,
    q: str,
    confidence_min: float = 0.0,  # ADDED: Default value
    entity_type: str | None = None,  # ADDED: Default value
    pmid: str | None = None  # ADDED: Default value
):
    """
    Search triplets across possible index variants.
    Returns list of triplet documents.
    """
    client = os_client()

    # Build possible index names (try both with and without prefix)
    prefix = getattr(settings.opensearch, "index_prefix", "")
    
    # Generate all possible index name patterns
    possible_indices = []
    if prefix:
        possible_indices.append(f"{prefix}{tenant}_triplets")
    possible_indices.append(f"{tenant}_triplets")
    possible_indices.append("triplets_default")  # Legacy fallback
    
    # Multi-field search across triplet fields
    must = (
        [{"multi_match": {
            "query": q,
            "fields": ["subject", "relation", "predicate", "object", "sentence_text"],
            "type": "best_fields"
        }}]
        if q else [{"match_all": {}}]
    )

    # Build filters
    filterq = []
    if confidence_min and confidence_min > 0:
        filterq.append({"range": {"confidence": {"gte": confidence_min}}})
    if entity_type:
        filterq.append({"term": {"subject_entity_type": entity_type}})
    if pmid:
        filterq.append({"term": {"pmid": pmid}})

    body = {
        "size": 50,
        "query": {"bool": {"must": must, "filter": filterq}},
    }

    # Try each possible index until one works
    for idx in possible_indices:
        try:
            print(f"[DEBUG] Attempting search on index: {idx}")
            print(f"[DEBUG] Query body: {json.dumps(body, indent=2)}")

            res = client.search(index=idx, body=body)
            hits = res.get("hits", {}).get("hits", [])
            
            if hits:
                print(f"[INFO] Found {len(hits)} results in index: {idx}")
                return [h["_source"] for h in hits]
            else:
                print(f"[INFO] Index {idx} exists but returned no results")
                
        except Exception as e:
            print(f"[WARN] Index '{idx}' not accessible: {e}")
            continue

    print("[WARN] No results found in any triplet index variant.")
    return []
