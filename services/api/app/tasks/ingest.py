# services/api/app/tasks/ingest.py
from __future__ import annotations

import uuid
import io
from typing import List, Tuple

from celery import shared_task
from minio import Minio
from datetime import datetime, timezone

from app.config import settings


def _minio_client() -> Minio:
    # new layout: settings.storage["minio"]["endpoint"], etc.
    try:
        ep = settings.storage["minio"]["endpoint"]
        ak = settings.storage["minio"]["access_key"]
        sk = settings.storage["minio"]["secret_key"]
        bucket = settings.storage["minio"]["bucket"]
    except Exception:
        ep = settings.minio_endpoint
        ak = settings.minio_access_key
        sk = settings.minio_secret_key
        bucket = settings.minio_bucket
    return Minio(
        endpoint=ep.replace("http://", "").replace("https://", ""),
        access_key=ak,
        secret_key=sk,
        secure=ep.startswith("https://"),
    )


def _bucket_name() -> str:
    try:
        return settings.storage["minio"]["bucket"]
    except Exception:
        return settings.minio_bucket


@shared_task(name="ingest.upload_and_enqueue")
def enqueue_ingest_files(tenant: str, files_meta: List[dict]) -> str:
    """
    files_meta: [{filename:str, content:bytes, content_type:str}]
    This task persists files to MinIO and (in real pipeline) would trigger:
      pdf->text->chunk->embed->index->RE->graph
    """
    job_id = str(uuid.uuid4())
    cli = _minio_client()
    bucket = _bucket_name()

    # Ensure bucket exists
    found = cli.bucket_exists(bucket)
    if not found:
        cli.make_bucket(bucket)

    # Store files under tenant/job_id/
    now = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base_prefix = f"ingest/{tenant}/{now}_{job_id}/"

    for f in files_meta:
        name = f["filename"]
        content = f["content"]
        ctype = f.get("content_type") or "application/octet-stream"
        obj_key = base_prefix + name
        cli.put_object(
            bucket_name=bucket,
            object_name=obj_key,
            data=io.BytesIO(content),
            length=len(content),
            content_type=ctype,
        )

    # TODO: chain Celery tasks: extract_text.s(...)->chunk.s(...)->embed.s(...)->index.s(...)->extract_re.s(...)
    return job_id

