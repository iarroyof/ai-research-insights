# services/api/app/triplets/graph.py
from __future__ import annotations
from typing import List, Dict, Any
from app.search.os_client import os_client
from app.config import settings


def _triplet_index_candidates(tenant: str) -> List[str]:
    """
    Return all possible index names for triplets, in priority order.
    """
    prefix = getattr(getattr(settings, "opensearch", None), "index_prefix", "") or ""
    candidates: List[str] = []
    if prefix:
        candidates.append(f"{prefix}{tenant}_triplets")
    candidates.append(f"{tenant}_triplets")
    candidates.append("triplets_default")  # Legacy / bulk-import index
    return candidates


def _compute_confidence(src: Dict[str, Any]) -> float:
    """
    Compute confidence from probability fields.
    """
    sp_ebio = float(src.get("subject_probably_EBio", 0.0))
    sp_ngen = float(src.get("subject_probably_NGen", 0.0))
    sp_other = float(src.get("subject_probably_otro", 0.0))
    op_ebio = float(src.get("object_probably_EBio", 0.0))
    op_ngen = float(src.get("object_probably_NGen", 0.0))
    op_other = float(src.get("object_probably_otro", 0.0))
    
    s_max = max(sp_ebio, sp_ngen, sp_other)
    o_max = max(op_ebio, op_ngen, op_other)
    return min(s_max, o_max)


async def subgraph_by_ids(tenant: str, ids: List[str], confidence_min: float) -> Dict[str, Any]:
    """
    Build a simple graph from triplets identified by OpenSearch _id.
    Nodes: unique subjects/objects
    Edges: subject --(relation)--> object, annotated with triple_id + confidence
    """
    if not ids:
        return {"nodes": [], "edges": []}
    
    client = os_client()
    indices = _triplet_index_candidates(tenant)
    
    nodes: Dict[str, Dict[str, Any]] = {}
    edges: List[Dict[str, Any]] = []
    
    # Try each index candidate until we find one that works
    for index in indices:
        try:
            body = {"ids": ids}
            res = client.mget(index=index, body=body)
            
            found_count = sum(1 for doc in res.get("docs", []) if doc.get("found"))
            
            if found_count > 0:
                print(f"[INFO] subgraph_by_ids: found {found_count}/{len(ids)} documents in index '{index}'")
                
                for doc in res.get("docs", []):
                    if not doc.get("found"):
                        continue
                    
                    src = doc.get("_source", {})
                    conf = _compute_confidence(src)
                    
                    if conf < confidence_min:
                        continue
                    
                    subj = src.get("subject")
                    obj = src.get("object")
                    rel = src.get("relation")
                    tid = doc.get("_id")
                    
                    if not subj or not obj:
                        continue
                    
                    # Add nodes
                    if subj not in nodes:
                        nodes[subj] = {"id": subj, "label": subj, "type": "entity"}
                    if obj not in nodes:
                        nodes[obj] = {"id": obj, "label": obj, "type": "entity"}
                    
                    # Add edge
                    edges.append({
                        "id": tid,
                        "source": subj,
                        "target": obj,
                        "label": rel or "related_to",
                        "triple_id": tid,
                        "confidence": conf,
                    })
                
                # If we found documents in this index, return the results
                if nodes:
                    return {"nodes": list(nodes.values()), "edges": edges}
                    
        except Exception as e:
            print(f"[WARN] subgraph_by_ids: index '{index}' not accessible: {e}")
            continue
    
    # If we get here, no index had the documents
    print(f"[WARN] subgraph_by_ids: no documents found in any index for {len(ids)} IDs")
    return {"nodes": [], "edges": []}
