# services/api/app/triplets/filters.py
"""
Quality filters and semantic search for triplet graphs.
Combines: stopword filtering, deduplication, biomedical priority, semantic search.
"""
from __future__ import annotations
from typing import List, Dict, Any, Set, Optional
import re

from app.search.os_client import os_client
from app.config import settings

# Stopwords: pronouns, generic terms, non-informative subjects/objects
STOPWORD_ENTITIES = {
    # Pronouns
    "we", "they", "it", "he", "she", "i", "you", "them", "us", "our", "their",
    "this", "that", "these", "those", "which", "what", "who", "whom",
    # Generic terms
    "study", "studies", "result", "results", "data", "analysis", "method", "methods",
    "approach", "paper", "article", "research", "work", "finding", "findings",
    "experiment", "experiments", "observation", "observations",
    "factor", "factors", "phenomenon", "part", "parts", "case", "cases",
    "example", "examples", "type", "types", "form", "forms", "kind", "group",
    "patient", "patients",  # Too generic in biomedical context
    "effect", "effects", "level", "levels", "rate", "rates",
    # Non-specific
    "one", "two", "three", "many", "several", "some", "all", "most", "both",
    "other", "others", "another", "same", "different", "various", "multiple",
    "first", "second", "third", "new", "recent", "previous", "current",
    # Actions as subjects (usually extraction errors)
    "use", "using", "used", "show", "shown", "found", "reported",
}

MIN_ENTITY_LENGTH = 3


def is_informative_entity(entity: str) -> bool:
    """Check if an entity is informative."""
    if not entity:
        return False
    
    clean = entity.lower().strip()
    
    if len(clean) < MIN_ENTITY_LENGTH:
        return False
    
    if clean in STOPWORD_ENTITIES:
        return False
    
    # Starts with stopword
    first_word = clean.split()[0] if clean.split() else ""
    if first_word in STOPWORD_ENTITIES:
        return False
    
    # All digits or single character repeated
    if clean.isdigit() or len(set(clean.replace(" ", ""))) < 2:
        return False
    
    return True


def normalize_entity(entity: str) -> str:
    """Normalize entity for deduplication."""
    if not entity:
        return ""
    norm = entity.lower().strip()
    norm = re.sub(r'\s+', ' ', norm)
    norm = re.sub(r'[^\w\s]', '', norm)
    return norm


def triplet_signature(triple: Dict[str, Any]) -> str:
    """Create signature for deduplication."""
    subj = normalize_entity(triple.get("subject", ""))
    pred = normalize_entity(triple.get("predicate", "") or triple.get("relation", ""))
    obj = normalize_entity(triple.get("object", ""))
    return f"{subj}|{pred}|{obj}"


def compute_quality_score(triple: Dict[str, Any], search_score: float = 0.0) -> float:
    """
    Compute combined quality score:
    - Biomedical relevance (EBio probabilities)
    - Confidence
    - Search relevance score
    """
    subj_ebio = triple.get("subject_probably_EBio", 0) or 0
    obj_ebio = triple.get("object_probably_EBio", 0) or 0
    conf = triple.get("confidence", 0) or 0
    
    # Weighted combination
    biomedical_score = (subj_ebio + obj_ebio) / 2
    
    # Normalize search_score (OpenSearch scores can be > 1)
    normalized_search = min(search_score / 20.0, 1.0) if search_score > 0 else 0
    
    return (biomedical_score * 0.3) + (conf * 0.3) + (normalized_search * 0.4)


def extract_key_terms(sentences: List[Dict[str, Any]]) -> str:
    """
    Extract key terms from selected sentences for semantic search.
    Uses subject/relation/object if available, otherwise sentence text.
    """
    terms = []
    
    for s in sentences:
        # Prefer structured triplet components
        if s.get("subject"):
            terms.append(s["subject"])
        if s.get("object"):
            terms.append(s["object"])
        if s.get("relation"):
            terms.append(s["relation"])
        
        # Fallback to text snippet
        if not terms and s.get("text"):
            # Extract first 100 chars as context
            terms.append(s["text"][:100])
    
    # Deduplicate and join
    unique_terms = list(dict.fromkeys(terms))[:20]  # Max 20 terms
    return " ".join(unique_terms)


async def search_triplets_semantic(
    tenant: str,
    query: str,
    paper_ids: Optional[List[str]] = None,
    confidence_min: float = 0.5,
    ebio_min: float = 0.3,
    top_k: int = 50,
) -> List[Dict[str, Any]]:
    """
    Semantic search for triplets with quality filtering.
    
    Args:
        tenant: Tenant ID
        query: Search query (extracted from selected sentences)
        paper_ids: Optional list of paper IDs to filter by
        confidence_min: Minimum confidence threshold
        ebio_min: Minimum EBio probability for at least one entity
        top_k: Number of results to return
    
    Returns:
        Filtered, deduplicated, ranked list of triplets
    """
    client = os_client()
    
    # Build possible index names
    prefix = getattr(settings.opensearch, "index_prefix", "")
    possible_indices = []
    if prefix:
        possible_indices.append(f"{prefix}{tenant}_triplets")
    possible_indices.append(f"{tenant}_triplets")
    possible_indices.append("triplets_default")
    
    # Build query
    must = []
    filter_clauses = []
    should = []
    
    # Main semantic search across triplet fields
    if query:
        must.append({
            "multi_match": {
                "query": query,
                "fields": [
                    "subject^2",      # Boost subject matches
                    "object^2",       # Boost object matches
                    "relation",
                    "sentence_text"
                ],
                "type": "best_fields",
                "fuzziness": "AUTO"
            }
        })
    else:
        must.append({"match_all": {}})
    
    # Filter by paper_ids if provided
    if paper_ids:
        filter_clauses.append({"terms": {"article_id": paper_ids}})
    
    # Biomedical preference: boost triplets with high EBio scores
    should.append({
        "range": {
            "subject_probably_EBio": {"gte": ebio_min, "boost": 2.0}
        }
    })
    should.append({
        "range": {
            "object_probably_EBio": {"gte": ebio_min, "boost": 2.0}
        }
    })
    
    body = {
        "size": top_k * 3,  # Fetch extra for post-filtering
        "query": {
            "bool": {
                "must": must,
                "filter": filter_clauses,
                "should": should,
                "minimum_should_match": 0
            }
        },
        "_source": True
    }
    
    # Try each possible index
    raw_results = []
    for idx in possible_indices:
        try:
            res = client.search(index=idx, body=body)
            hits = res.get("hits", {}).get("hits", [])
            
            for h in hits:
                src = h.get("_source", {})
                src["_search_score"] = h.get("_score", 0)
                src["_id"] = h.get("_id")
                raw_results.append(src)
            
            if raw_results:
                break  # Found results, stop trying other indices
                
        except Exception as e:
            print(f"[WARN] search_triplets_semantic: index '{idx}' error: {e}")
            continue
    
    # Apply quality filters
    filtered = filter_triplets_quality(
        raw_results,
        confidence_min=confidence_min,
        ebio_min=ebio_min,
        max_results=top_k
    )
    
    return filtered


def filter_triplets_quality(
    triplets: List[Dict[str, Any]],
    confidence_min: float = 0.5,
    ebio_min: float = 0.3,
    max_results: int = 50,
) -> List[Dict[str, Any]]:
    """
    Apply quality filters to triplets:
    1. Remove non-informative entities (stopwords)
    2. Require minimum biomedical relevance
    3. Deduplicate similar triplets
    4. Rank by quality score and limit results
    """
    filtered = []
    seen_signatures: Set[str] = set()
    
    for t in triplets:
        subj = t.get("subject", "")
        obj = t.get("object", "")
        
        # Skip non-informative entities
        if not is_informative_entity(subj) or not is_informative_entity(obj):
            continue
        
        # Compute confidence if not present
        if "confidence" not in t:
            subj_max = max(
                t.get("subject_probably_EBio", 0) or 0,
                t.get("subject_probably_NGen", 0) or 0,
                t.get("subject_probably_otro", 0) or 0
            )
            obj_max = max(
                t.get("object_probably_EBio", 0) or 0,
                t.get("object_probably_NGen", 0) or 0,
                t.get("object_probably_otro", 0) or 0
            )
            t["confidence"] = min(subj_max, obj_max)
        
        # Skip low confidence
        if t.get("confidence", 0) < confidence_min:
            continue
        
        # Check biomedical relevance (at least one entity should be biomedical)
        subj_ebio = t.get("subject_probably_EBio", 0) or 0
        obj_ebio = t.get("object_probably_EBio", 0) or 0
        if subj_ebio < ebio_min and obj_ebio < ebio_min:
            continue
        
        # Deduplication
        sig = triplet_signature(t)
        if sig in seen_signatures:
            continue
        seen_signatures.add(sig)
        
        # Compute quality score for ranking
        search_score = t.get("_search_score", 0)
        t["_quality_score"] = compute_quality_score(t, search_score)
        
        filtered.append(t)
    
    # Sort by quality score (descending) and limit
    filtered.sort(key=lambda x: x.get("_quality_score", 0), reverse=True)
    
    return filtered[:max_results]

