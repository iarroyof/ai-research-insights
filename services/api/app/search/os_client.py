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
    Performs BM25 text search in OpenSearch.
    
    UPDATED: Now searches triplet indices instead of chunk indices.
    Falls back through multiple possible index names.
    """
    # Build list of possible index names to try
    # NOTE: /triplets/ingest_csv creates "triplets_{tenant}" format
    # NOTE: /triplets/csv creates "{prefix}{tenant}_triplets" format
    prefix = getattr(settings.opensearch, "index_prefix", "")
    possible_indices = []
    
    # Try all possible naming patterns
    possible_indices.extend([
        f"triplets_{tenant}",      # Created by /triplets/ingest_csv
        f"{tenant}_triplets",      # Alternative pattern
        "triplets_default",        # Legacy fallback
    ])
    
    if prefix:
        possible_indices.append(f"{prefix}{tenant}_triplets")  # With prefix
    
    # Build query for triplet fields
    body = {
        "size": k,
        "query": {
            "multi_match": {
                "query": query,
                "fields": [
                    "subject^2",           # Boost subject matches
                    "object^2",            # Boost object matches
                    "sentence_text",       # Full sentence context
                    "predicate",           # Relation type
                    "relation"             # Alternative field name
                ],
                "type": "best_fields"
            }
        },
        "_source": [
            "paper_id", "pmid", "pmcid", 
            "subject", "predicate", "relation", "object",
            "sentence_text", "confidence",
            "subject_entity_type", "object_entity_type"
        ]
    }
    
    # Add confidence filter if provided
    confidence_min = filters.get("confidence_min")
    if confidence_min:
        body["query"] = {
            "bool": {
                "must": [body["query"]],
                "filter": [{"range": {"confidence": {"gte": confidence_min}}}]
            }
        }
    
    # Try each index until one works
    client = os_client()
    last_error = None
    
    for idx in possible_indices:
        try:
            print(f"[DEBUG] Searching index: {idx}")
            res = client.search(index=idx, body=body)
            hits = res.get("hits", {}).get("hits", [])
            
            if hits:
                print(f"[INFO] Found {len(hits)} results in index: {idx}")
                
                # Normalize triplet results to look like chunk results
                normalized = []
                for h in hits:
                    src = h["_source"]
                    
                    # Build a readable text from triplet components
                    triplet_text = f"{src.get('subject', '')} - {src.get('predicate') or src.get('relation', '')} - {src.get('object', '')}"
                    full_text = src.get('sentence_text', triplet_text)
                    
                    normalized.append({
                        "paper_id": src.get("paper_id"),
                        "title": full_text[:100] + "..." if len(full_text) > 100 else full_text,
                        "pmid": src.get("pmid"),
                        "pmcid": src.get("pmcid"),
                        "page": None,
                        "sent_id": None,
                        "text": full_text,
                        "score": h["_score"],
                        # Include triplet-specific fields for frontend
                        "subject": src.get("subject"),
                        "predicate": src.get("predicate") or src.get("relation"),
                        "object": src.get("object"),
                        "confidence": src.get("confidence"),
                    })
                
                return normalized
            else:
                print(f"[INFO] Index {idx} exists but returned 0 results")
                
        except Exception as e:
            print(f"[WARN] Failed to search index '{idx}': {e}")
            last_error = e
            continue
    
    # If we get here, all indices failed
    if last_error:
        raise last_error
    
    # No error but no results
    print("[WARN] No triplet indices found or all returned empty results")
    return []
