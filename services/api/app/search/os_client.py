# services/api/app/search/os_client.py

from __future__ import annotations
from typing import List, Dict, Any
import re
import hashlib
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


def extract_pmid_from_article_id(article_id: str) -> tuple[str, str]:
    """
    Extract PMID/PMCID from article_id patterns.
    
    Examples:
        "PMC10138687.txt" -> (None, "PMC10138687")
        "30349342" -> ("30349342", None)
    
    Returns:
        (pmid, pmcid) tuple
    """
    if not article_id:
        return (None, None)
    
    # Remove file extensions
    clean_id = re.sub(r'\.(txt|pdf|xml)$', '', article_id, flags=re.IGNORECASE)
    
    # Check for PMC pattern
    pmc_match = re.search(r'PMC(\d+)', clean_id, re.IGNORECASE)
    if pmc_match:
        return (None, f"PMC{pmc_match.group(1)}")
    
    # Check if it's a plain numeric PMID
    if clean_id.isdigit():
        return (clean_id, None)
    
    return (None, None)


def generate_sent_id(sentence_text: str, article_id: str = "") -> str:
    """
    Generate a deterministic sentence ID from text hash.
    """
    combined = f"{article_id}:{sentence_text}"
    return hashlib.md5(combined.encode('utf-8')).hexdigest()[:12]


def calculate_triplet_confidence(src: dict) -> float:
    """
    Calculate triplet confidence from probability scores.
    Confidence = min(max(subject_probs), max(object_probs))
    """
    subj_ebio = src.get("subject_probably_EBio", 0.0)
    subj_ngen = src.get("subject_probably_NGen", 0.0)
    subj_otro = src.get("subject_probably_otro", 0.0)
    
    obj_ebio = src.get("object_probably_EBio", 0.0)
    obj_ngen = src.get("object_probably_NGen", 0.0)
    obj_otro = src.get("object_probably_otro", 0.0)
    
    subj_max = max(subj_ebio, subj_ngen, subj_otro)
    obj_max = max(obj_ebio, obj_ngen, obj_otro)
    
    return min(subj_max, obj_max)


def _possible_triplet_indices(tenant: str) -> list[str]:
    prefix = getattr(settings.opensearch, "index_prefix", "")
    possible_indices = [
        f"triplets_{tenant}",
        f"{tenant}_triplets",
        "triplets_default",
    ]
    if prefix:
        possible_indices.append(f"{prefix}{tenant}_triplets")
    return possible_indices


def _possible_indices_for_level(tenant: str, level: str) -> list[str]:
    prefix = getattr(settings.opensearch, "index_prefix", "") or ""
    names: list[str]
    if level == "title":
        names = [f"{tenant}_papers", f"{tenant}_chunks", f"papers_{tenant}", f"chunks_{tenant}", f"triplets_{tenant}", f"{tenant}_triplets"]
    elif level == "paper":
        names = [f"{tenant}_chunks", f"{tenant}_papers", f"chunks_{tenant}", f"papers_{tenant}", f"triplets_{tenant}", f"{tenant}_triplets"]
    else:
        names = [f"triplets_{tenant}", f"{tenant}_triplets", f"{tenant}_chunks", f"chunks_{tenant}", "triplets_default"]
    if prefix:
        prefixed = [f"{prefix}{name}" for name in names if not name.startswith(prefix)]
        names.extend(prefixed)
    if level in {"title", "paper"}:
        names.extend(["papers_default", "chunks_default", "triplets_default"])
    return list(dict.fromkeys(names))


def _source_fields() -> list[str]:
    return [
        "article_id",
        "id",
        "paper_id",
        "title",
        "paper_title",
        "doc_title",
        "article_title",
        "abstract",
        "content",
        "chunk",
        "pmid",
        "pmcid",
        "page",
        "sent_id",
        "subject",
        "relation",
        "predicate",
        "object",
        "sentence_text",
        "text",
        "subject_probably_EBio",
        "subject_probably_NGen",
        "subject_probably_otro",
        "object_probably_EBio",
        "object_probably_NGen",
        "object_probably_otro",
    ]


def _has_triplet_confidence_fields(src: dict) -> bool:
    return any(
        key in src
        for key in (
            "subject_probably_EBio",
            "subject_probably_NGen",
            "subject_probably_otro",
            "object_probably_EBio",
            "object_probably_NGen",
            "object_probably_otro",
        )
    )


def _confidence_from_source(src: dict) -> float:
    if _has_triplet_confidence_fields(src):
        return calculate_triplet_confidence(src)
    try:
        return float(src.get("confidence", 1.0) or 1.0)
    except Exception:
        return 1.0


def _title_from_source(src: dict, sentence_text: str) -> str:
    return (
        src.get("title")
        or src.get("paper_title")
        or src.get("doc_title")
        or src.get("article_title")
        or (sentence_text[:100] + "..." if len(sentence_text) > 100 else sentence_text)
    )


def _text_from_source(src: dict, *, level: str) -> str:
    title = src.get("title") or src.get("paper_title") or src.get("doc_title") or src.get("article_title") or ""
    if level == "title":
        return title or src.get("sentence_text") or src.get("text") or src.get("abstract") or ""
    if level == "paper":
        return src.get("abstract") or src.get("text") or src.get("content") or src.get("chunk") or src.get("sentence_text") or title
    return src.get("sentence_text") or src.get("text") or src.get("content") or src.get("chunk") or title


def _normalize_search_hit(hit: dict, *, retrieval_mode: str, level: str) -> Dict[str, Any]:
    src = hit.get("_source", {}) or {}
    article_id = src.get("article_id") or src.get("paper_id") or src.get("id") or ""
    pmid, pmcid = extract_pmid_from_article_id(article_id)
    pmid = src.get("pmid") or pmid
    pmcid = src.get("pmcid") or pmcid
    sentence_text = _text_from_source(src, level=level)
    sent_id = src.get("sent_id") or generate_sent_id(sentence_text, article_id)
    confidence = _confidence_from_source(src)
    title = _title_from_source(src, sentence_text)
    return {
        "paper_id": article_id,
        "title": title,
        "pmid": pmid,
        "pmcid": pmcid,
        "page": src.get("page"),
        "sent_id": sent_id,
        "text": sentence_text,
        "score": hit.get("_score", 0.0),
        "retrieval_mode": retrieval_mode,
        "search_level": level,
        "subject": src.get("subject", ""),
        "relation": src.get("relation") or src.get("predicate", ""),
        "object": src.get("object", ""),
        "confidence": round(confidence, 4),
        "subject_probs": {
            "EBio": round(src.get("subject_probably_EBio", 0.0), 4),
            "NGen": round(src.get("subject_probably_NGen", 0.0), 4),
            "other": round(src.get("subject_probably_otro", 0.0), 4),
        },
        "object_probs": {
            "EBio": round(src.get("object_probably_EBio", 0.0), 4),
            "NGen": round(src.get("object_probably_NGen", 0.0), 4),
            "other": round(src.get("object_probably_otro", 0.0), 4),
        },
    }


def _normalize_hits(hits: list[dict], *, confidence_min: float | None, k: int, retrieval_mode: str, level: str = "sentence") -> list[dict]:
    normalized = []
    for hit in hits:
        item = _normalize_search_hit(hit, retrieval_mode=retrieval_mode, level=level)
        if confidence_min and item["confidence"] < confidence_min:
            continue
        normalized.append(item)
        if len(normalized) >= k:
            break
    return normalized


def _needs_vector_fallback(results: list[dict], *, k: int, filters: Dict[str, Any]) -> bool:
    if not filters.get("query_vector"):
        return False
    if not bool(filters.get("allow_vector_fallback", getattr(settings.opensearch, "use_vectors", False))):
        return False
    min_results = int(filters.get("fallback_min_results", min(3, max(1, k))) or 0)
    min_score = float(filters.get("fallback_min_score", 0.0) or 0.0)
    if len(results) < min_results:
        return True
    best_score = max((float(item.get("score", 0.0) or 0.0) for item in results), default=0.0)
    return bool(min_score and best_score < min_score)


def _vector_query_bodies(filters: Dict[str, Any], k: int) -> list[dict]:
    query_vector = filters.get("query_vector")
    if not isinstance(query_vector, list) or not query_vector:
        return []
    vector_fields = filters.get("vector_fields") or ["embedding", "vector", "vec", "sentence_vector"]
    out = []
    for field in vector_fields:
        out.append(
            {
                "size": k,
                "query": {
                    "knn": {
                        str(field): {
                            "vector": query_vector,
                            "k": k,
                        }
                    }
                },
                "_source": _source_fields(),
            }
        )
    return out


def _fields_for_level(level: str) -> list[str]:
    if level == "title":
        return [
            "title^8",
            "paper_title^8",
            "doc_title^8",
            "article_title^8",
            "abstract^2",
            "text",
            "sentence_text",
            "subject",
            "object",
        ]
    if level == "paper":
        return [
            "title^4",
            "paper_title^4",
            "doc_title^4",
            "article_title^4",
            "abstract^5",
            "text^3",
            "content^3",
            "chunk^3",
            "sentence_text^2",
            "subject",
            "object",
        ]
    return [
        "sentence_text^5",
        "text^4",
        "content^2",
        "chunk^2",
        "subject^2",
        "object^2",
        "relation",
        "predicate",
        "title",
        "paper_title",
    ]


def os_multilevel_query(tenant: str, level: str, query: str, filters: Dict[str, Any], k: int) -> List[Dict[str, Any]]:
    """
    BM25-first OpenSearch retrieval for one structured level.

    Levels:
      - title: paper-title-heavy search for candidate papers and vocabulary.
      - paper: paper/chunk/abstract-level search for broader background.
      - sentence: sentence/triplet-level search for exact evidence.
    """
    level = level if level in {"title", "paper", "sentence"} else "sentence"
    confidence_min = filters.get("confidence_min") if level == "sentence" else None
    body = {
        "size": min(max(k, 1) * (3 if confidence_min else 1), 200),
        "query": {
            "multi_match": {
                "query": query,
                "fields": _fields_for_level(level),
                "type": "best_fields",
            }
        },
        "_source": _source_fields(),
    }

    client = os_client()
    last_error = None
    for idx in _possible_indices_for_level(tenant, level):
        try:
            print(f"[DEBUG] Searching level={level} index: {idx}")
            res = client.search(index=idx, body=body)
            hits = res.get("hits", {}).get("hits", [])
            if hits:
                print(f"[INFO] Found {len(hits)} level={level} results in index: {idx}")
                normalized = _normalize_hits(hits, confidence_min=confidence_min, k=k, retrieval_mode="bm25", level=level)
                if level == "sentence" and _needs_vector_fallback(normalized, k=k, filters=filters):
                    normalized = _append_vector_fallback(client, idx, normalized, filters, confidence_min, k)
                return normalized[:k]
        except Exception as e:
            print(f"[WARN] Failed level={level} search index '{idx}': {e}")
            last_error = e
            continue
    if last_error:
        print(f"[WARN] No accessible level={level} index returned results: {last_error}")
    return []


def os_hybrid_query(tenant: str, query: str, filters: Dict[str, Any], k: int) -> List[Dict[str, Any]]:
    """
    Performs BM25 text search in OpenSearch.

    UPDATED: Now searches triplet indices and extracts actual fields.
    Falls back through multiple possible index names.
    """
    possible_indices = _possible_triplet_indices(tenant)

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
                    "relation"             # Relation type
                ],
                "type": "best_fields"
            }
        },
        "_source": _source_fields(),
    }

    # Add confidence filter if provided
    confidence_min = filters.get("confidence_min")
    if confidence_min:
        # Since confidence is calculated, we'll filter after retrieval
        # Increase k to compensate for filtering
        body["size"] = min(k * 3, 200)

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
                normalized = _normalize_hits(hits, confidence_min=confidence_min, k=k, retrieval_mode="bm25", level="sentence")
                if _needs_vector_fallback(normalized, k=k, filters=filters):
                    normalized = _append_vector_fallback(client, idx, normalized, filters, confidence_min, k)
                return normalized[:k]
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


def _append_vector_fallback(
    client: OpenSearch,
    index: str,
    bm25_results: list[dict],
    filters: Dict[str, Any],
    confidence_min: float | None,
    k: int,
) -> list[dict]:
    seen = {(item.get("paper_id"), item.get("sent_id"), item.get("text")) for item in bm25_results}
    merged = list(bm25_results)
    for body in _vector_query_bodies(filters, max(k, int(filters.get("vec_k", k) or k))):
        try:
            res = client.search(index=index, body=body)
        except Exception as e:
            print(f"[WARN] Vector fallback failed for index '{index}': {e}")
            continue
        hits = res.get("hits", {}).get("hits", [])
        for item in _normalize_hits(hits, confidence_min=confidence_min, k=k, retrieval_mode="vector_fallback", level="sentence"):
            key = (item.get("paper_id"), item.get("sent_id"), item.get("text"))
            if key in seen:
                continue
            seen.add(key)
            merged.append(item)
            if len(merged) >= k:
                return merged
    return merged
