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


def os_hybrid_query(tenant: str, query: str, filters: Dict[str, Any], k: int) -> List[Dict[str, Any]]:
    """
    Performs BM25 text search in OpenSearch.

    UPDATED: Now searches triplet indices and extracts actual fields.
    Falls back through multiple possible index names.
    """
    # Build list of possible index names to try
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
                    "relation"             # Relation type
                ],
                "type": "best_fields"
            }
        },
        "_source": [
            "article_id",              # ✅ ACTUAL field in index
            "subject", "relation", "object",
            "sentence_text",
            "subject_probably_EBio", "subject_probably_NGen", "subject_probably_otro",
            "object_probably_EBio", "object_probably_NGen", "object_probably_otro"
        ]
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

                # Normalize triplet results to expected format
                normalized = []
                for h in hits:
                    src = h["_source"]
                    
                    # Extract article_id and parse PMID/PMCID
                    article_id = src.get("article_id", "")
                    pmid, pmcid = extract_pmid_from_article_id(article_id)
                    
                    # Generate synthetic sent_id
                    sentence_text = src.get("sentence_text", "")
                    sent_id = generate_sent_id(sentence_text, article_id)
                    
                    # Calculate confidence
                    confidence = calculate_triplet_confidence(src)
                    
                    # Apply confidence filter if specified
                    if confidence_min and confidence < confidence_min:
                        continue
                    
                    # Build a readable title from triplet
                    subject = src.get("subject", "")
                    relation = src.get("relation", "")
                    obj = src.get("object", "")
                    triplet_summary = f"{subject} - {relation} - {obj}"
                    
                    # Truncate sentence for title
                    title = sentence_text[:100] + "..." if len(sentence_text) > 100 else sentence_text

                    normalized.append({
                        "paper_id": article_id,      # Use article_id as paper_id
                        "title": title,
                        "pmid": pmid,
                        "pmcid": pmcid,
                        "page": None,                # Not available in CSV data
                        "sent_id": sent_id,          # Synthetic ID from hash
                        "text": sentence_text,
                        "score": h["_score"],
                        # Include triplet-specific fields
                        "subject": subject,
                        "relation": relation,
                        "object": obj,
                        "confidence": round(confidence, 4),
                        "subject_probs": {
                            "EBio": round(src.get("subject_probably_EBio", 0.0), 4),
                            "NGen": round(src.get("subject_probably_NGen", 0.0), 4),
                            "other": round(src.get("subject_probably_otro", 0.0), 4)
                        },
                        "object_probs": {
                            "EBio": round(src.get("object_probably_EBio", 0.0), 4),
                            "NGen": round(src.get("object_probably_NGen", 0.0), 4),
                            "other": round(src.get("object_probably_otro", 0.0), 4)
                        }
                    })
                    
                    # Stop if we've collected enough results after filtering
                    if len(normalized) >= k:
                        break

                # Return top k results
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
