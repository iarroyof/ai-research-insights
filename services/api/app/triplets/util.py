# services/api/app/triplets/util.py
from __future__ import annotations

from typing import List, Dict, Any

from app.search.os_client import os_client
from app.config import settings


def _triplets_index_candidates(tenant: str) -> List[str]:
    """
    Possible index names where triplets may live.
    """
    prefix = getattr(getattr(settings, "opensearch", None), "index_prefix", "") or ""
    candidates: List[str] = []
    if prefix:
        candidates.append(f"{prefix}{tenant}_triplets")
    candidates.append(f"{tenant}_triplets")
    candidates.append("triplets_default")  # legacy / bulk-import index
    return candidates


def _compute_confidence(doc: Dict[str, Any]) -> float:
    """
    Reconstruct a confidence score from prob fields in triplets_default.
    confidence = min( max(subject_probs), max(object_probs) ).
    Falls back to 0.0 if fields are missing.
    """
    sp_ebio = float(doc.get("subject_probably_EBio", 0.0) or 0.0)
    sp_ngen = float(doc.get("subject_probably_NGen", 0.0) or 0.0)
    sp_other = float(doc.get("subject_probably_otro", 0.0) or 0.0)

    op_ebio = float(doc.get("object_probably_EBio", 0.0) or 0.0)
    op_ngen = float(doc.get("object_probably_NGen", 0.0) or 0.0)
    op_other = float(doc.get("object_probably_otro", 0.0) or 0.0)

    s_max = max(sp_ebio, sp_ngen, sp_other)
    o_max = max(op_ebio, op_ngen, op_other)

    return min(s_max, o_max)


async def triples_for_sentences(
    tenant: str,
    sentences: List[dict],
    confidence_min: float,
) -> List[Dict[str, Any]]:
    """
    Legacy helper: given a list of sentence dicts with paper_id / pmcid,
    fetch existing triplets for those papers and filter by confidence.
    NOTE: triplets_default does NOT store sent_id, so we can only filter by article_id.
    """
    # Derive paper_ids (article_id in OS) from either paper_id or pmcid
    paper_ids = sorted(
        {
            s.get("paper_id")
            or (f"{s['pmcid']}.txt" if s.get("pmcid") and not str(s.get("paper_id", "")).endswith(".txt") else None)
            for s in sentences
            if s.get("paper_id") or s.get("pmcid")
        }
    )
    paper_ids = [pid for pid in paper_ids if pid]

    return await triples_for_papers(tenant, paper_ids, confidence_min)


async def triples_for_papers(
    tenant: str,
    paper_ids: List[str],
    confidence_min: float,
) -> List[Dict[str, Any]]:
    """
    Fetch all triplets for the given paper_ids (article_id) from OpenSearch,
    across possible index variants, and filter by reconstructed confidence.
    Returns a normalized list of {triple_id, paper_id, sentence_text, subject, predicate, object, confidence}.
    """
    if not paper_ids:
        return []

    client = os_client()
    indices = _triplets_index_candidates(tenant)

    results: List[Dict[str, Any]] = []

    for idx in indices:
        try:
            # We keep size reasonably large but bounded; if any paper has >10k triplets,
            # we can later switch to scroll/point-in-time search.
            body = {
                "size": 10000,
                "query": {
                    "terms": {
                        "article_id": paper_ids,
                    }
                },
            }

            res = client.search(index=idx, body=body)
            hits = res.get("hits", {}).get("hits", [])

        except Exception as e:
            # Index not present / not accessible – just skip
            print(f"[WARN] triples_for_papers: index '{idx}' not accessible: {e}")
            continue

        for h in hits:
            src = h.get("_source", {})
            conf = _compute_confidence(src)

            if conf < confidence_min:
                continue

            results.append(
                {
                    "triple_id": h.get("_id"),
                    "paper_id": src.get("article_id"),
                    "sentence_text": src.get("sentence_text"),
                    "subject": src.get("subject"),
                    "predicate": src.get("relation"),
                    "object": src.get("object"),
                    "confidence": conf,
                }
            )

    return results


async def upsert_triples_batch(tenant: str, triples: List[dict]):
    """
    Persist triples to Neo4j + OpenSearch (batched).
    For now, this is a no-op for your existing precomputed triplets_default index;
    the REBEL path can later be wired to the same writer as CSV ingestion.
    """
    # TODO: wire this to your CSV ingestion writer when you want REBEL triplets persisted.
    # For now, we do nothing to keep precomputed triplets_default read-only.
    return True

