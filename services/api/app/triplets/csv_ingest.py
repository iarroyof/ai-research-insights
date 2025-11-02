from __future__ import annotations
import csv, io, uuid
from typing import Tuple
from app.config import settings
from app.tasks.triplets import enqueue_triplets_csv


async def ingest_csv_triplets(tenant: str, file) -> Tuple[str, dict]:
    content = await file.read()
    try:
        _ = content.decode("utf-8")
    except UnicodeDecodeError:
        raise ValueError("CSV must be UTF-8")

    job_id = str(uuid.uuid4())
    enqueue_triplets_csv.delay(tenant, job_id, content)
    return job_id, {"ingested": 0, "rejected": 0}
