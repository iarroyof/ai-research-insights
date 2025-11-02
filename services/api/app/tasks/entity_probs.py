# services/api/app/tasks/entity_probs.py
from __future__ import annotations
from typing import List, Dict, Any
from celery import group, shared_task

from app.services.zero_shot import score_labels
from app.core.settings import settings

# Example stubs – replace with your real storage queries/updates
def iter_triples_missing_probs(limit: int, offset: int) -> List[Dict[str, Any]]:
    """
    Return a page of triples missing any of the {EBio,NGen,otro} fields.
    Must include keys: id, sentence_text, subject, object
    """
    # TODO: query Postgres or OS. This is a placeholder.
    return []

def save_probs(triple_id: str, subj: Dict[str,float], obj: Dict[str,float]):
    """
    Persist probabilities back to DB / index.
    """
    # TODO: implement persistence.
    pass

@shared_task(ignore_result=False)
def compute_entity_probs_batch(triples: List[Dict[str, Any]], labels: List[str] | None = None):
    labs = labels or settings.classification.labels
    subj_texts = [t["subject"] for t in triples]
    obj_texts  = [t["object"]  for t in triples]
    sp = score_labels(subj_texts, labs)
    op = score_labels(obj_texts, labs)
    for t, sprob, oprob in zip(triples, sp, op):
        save_probs(t["id"], sprob, oprob)
    return len(triples)

@shared_task(ignore_result=False)
def backfill_missing_entity_probs(total_batches: int = 100, page_size: int = 500, labels: List[str] | None = None):
    """
    Fan out into parallel subjobs. Returns total scheduled.
    """
    batches = []
    for b in range(total_batches):
        offset = b * page_size
        page = iter_triples_missing_probs(page_size, offset)
        if not page:
            break
        batches.append(compute_entity_probs_batch.s(page, labels))
    job = group(batches)
    async_result = job.apply_async()
    return {"scheduled": len(batches), "task_ids": [r.id for r in async_result.results]}

