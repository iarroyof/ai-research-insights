from __future__ import annotations
from typing import List

async def triples_for_sentences(tenant: str, sentences: List[dict], confidence_min: float):
    """
    Query OpenSearch (triplets index) by (paper_id, sent_id) and confidence >= threshold.
    Return list of {triple_id, sent_id, ...}
    """
    # TODO: implement OS query by paper_id + sent_id
    return []

async def upsert_triples_batch(tenant: str, triples: List[dict]):
    """
    Persist triples to Neo4j + OpenSearch (batched). Reuse logic from CSV ingestion writer.
    """
    # TODO: call the same writers as CSV path, adapting payload fields
    return True

