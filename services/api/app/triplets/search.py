from __future__ import annotations
from app.search.os_client import os_client
from app.config import settings


async def search_triplets(
    tenant: str,
    q: str,
    confidence_min: float,
    entity_type: str | None,
    pmid: str | None
):
    idx = f"{settings.os.index_prefix}{tenant}_triplets"
    must = [{"match": {"sentence_text": {"query": q}}}] if q else [{"match_all": {}}]
    filterq = [{"range": {"confidence": {"gte": confidence_min}}}]
    if entity_type:
        filterq.append({"term": {"subject_entity_type": entity_type}})
    if pmid:
        filterq.append({"term": {"pmid": pmid}})
    body = {"size": 50, "query": {"bool": {"must": must, "filter": filterq}}}
    res = os_client().search(index=idx, body=body)
    return [h["_source"] for h in res["hits"]["hits"]]
