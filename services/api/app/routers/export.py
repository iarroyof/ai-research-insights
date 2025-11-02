# services/api/app/routers/export.py
from __future__ import annotations
from typing import List
from fastapi import APIRouter
from pydantic import BaseModel
import csv, io

from app.integrations.extraction_client import extract_rows_for_csv
from app.clients.minio import get_minio  # you already have this

router = APIRouter(prefix="/export", tags=["export"])

class ExportRequest(BaseModel):
    article_id: str
    sentences: List[str]
    labels: List[str] | None = None
    add_probs: bool = True

@router.post("/triples.csv")
async def export_triples_csv(req: ExportRequest):
    rows, header = await extract_rows_for_csv(
        req.article_id, req.sentences, labels=req.labels, add_probs=req.add_probs
    )
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(header)
    w.writerows(rows)
    data = buf.getvalue().encode()

    from app.core.settings import settings
    bucket = settings.export.triples_csv_bucket
    key = settings.export.triples_csv_object

    minio = get_minio()
    minio.put_object(
        bucket, key, io.BytesIO(data), len(data),
        content_type="text/csv",
    )
    return {"bucket": bucket, "object": key, "rows": len(rows)}

