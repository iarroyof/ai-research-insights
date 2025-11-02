# services/api/app/routers/jobs.py
from __future__ import annotations
from typing import Optional, List
from fastapi import APIRouter
from pydantic import BaseModel
from app.tasks.entity_probs import backfill_missing_entity_probs

router = APIRouter(prefix="/jobs", tags=["jobs"])

class BackfillReq(BaseModel):
    total_batches: int = 100
    page_size: int = 500
    labels: Optional[List[str]] = None

@router.post("/backfill/probabilities")
def launch_backfill(req: BackfillReq):
    result = backfill_missing_entity_probs.delay(
        total_batches=req.total_batches,
        page_size=req.page_size,
        labels=req.labels,
    )
    return {"enqueued": True, "task_id": result.id}
