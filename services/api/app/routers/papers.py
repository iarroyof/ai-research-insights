# services/api/app/routers/papers.py
from __future__ import annotations

import uuid
import io
from typing import List, Optional, Dict, Any

from fastapi import APIRouter, Request, UploadFile, File, HTTPException
from pydantic import BaseModel, Field

from minio import Minio
from minio.error import S3Error

from app.config import settings
from app.summarize.conditioned import summarize_conditioned

# Optional Celery enqueue helpers (use if present)
try:
    # Expected signatures:
    #   enqueue_ingest_uris(tenant: str, uris: List[str]) -> str
    #   enqueue_ingest_links(tenant: str, links: List[dict]) -> str
    #   enqueue_ingest_pmid(tenant: str, pmids: List[str]) -> str
    from app.tasks.ingest import (
        enqueue_ingest_uris,
        enqueue_ingest_links,
        enqueue_ingest_pmid,
    )
except Exception:  # pragma: no cover
    enqueue_ingest_uris = None
    enqueue_ingest_links = None
    enqueue_ingest_pmid = None


router = APIRouter(prefix="/papers", tags=["papers"])


# ---------------------------
# Upload: PDFs/TXT (multipart)
# ---------------------------

class UploadResponse(BaseModel):
    job_id: str
    num_files: int
    objects: List[str] = Field(default_factory=list)


@router.post("/upload", response_model=UploadResponse)
async def upload_papers(req: Request, files: List[UploadFile] = File(...)):
    """
    Streams user-uploaded files to MinIO in a tenant-scoped prefix, then enqueues
    an ingestion job that:
      - virus scans (if enabled in your pipeline),
      - extracts text (PyMuPDF),
      - metadata (GROBID),
      - chunks + embeddings,
      - indexes,
      - optionally runs RE (REBEL) and writes graph.

    Returns a `job_id` (Celery task-group ID or UUID) and a list of stored URIs.
    """
    tenant = req.state.tenant_id
    if not files:
        raise HTTPException(status_code=400, detail="No files received")

    bucket = settings.minio_bucket
    endpoint = settings.minio_endpoint
    # endpoint may include scheme; Minio client handles both http/https forms:
    secure = endpoint.startswith("https://")
    endpoint_host = endpoint.replace("https://", "").replace("http://", "")

    client = Minio(
        endpoint=endpoint_host,
        access_key=settings.minio_access_key,
        secret_key=settings.minio_secret_key,
        secure=secure,
    )

    # Ensure bucket exists
    try:
        found = client.bucket_exists(bucket)
        if not found:
            client.make_bucket(bucket)
    except S3Error as e:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"MinIO bucket error: {e}")

    stored_uris: List[str] = []
    for f in files:
        # Basic content-type guard (server-side validation should follow in pipeline)
        if not (f.filename.lower().endswith(".pdf") or f.filename.lower().endswith(".txt")):
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {f.filename}")

        data = await f.read()
        if not data:
            raise HTTPException(status_code=400, detail=f"Empty file: {f.filename}")

        object_name = f"{tenant}/uploads/{uuid.uuid4()}_{f.filename}"
        try:
            client.put_object(
                bucket_name=bucket,
                object_name=object_name,
                data=io.BytesIO(data),
                length=len(data),
                content_type=f.content_type or "application/octet-stream",
            )
        except S3Error as e:  # pragma: no cover
            raise HTTPException(status_code=502, detail=f"Failed to store {f.filename}: {e}")

        stored_uris.append(f"s3://{bucket}/{object_name}")

    # Enqueue ingestion job (URIs)
    if enqueue_ingest_uris is None:
        # Give a useful message if the helper isn't wired yet
        raise HTTPException(
            status_code=501,
            detail=(
                "Ingestion task not wired. Implement app.tasks.ingest.enqueue_ingest_uris(tenant, uris) "
                "to process uploaded objects."
            ),
        )

    job_id = await enqueue_ingest_uris(tenant, stored_uris)
    return UploadResponse(job_id=job_id, num_files=len(files), objects=stored_uris)


# ---------------------------
# Link ingestion (arXiv/PubMed/Direct URLs)
# ---------------------------

class LinkItem(BaseModel):
    url: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class LinkIngestResponse(BaseModel):
    job_id: str
    num_links: int


@router.post("/link", response_model=LinkIngestResponse)
async def ingest_links(req: Request, links: List[LinkItem]):
    tenant = req.state.tenant_id
    if not links:
        raise HTTPException(status_code=400, detail="No links provided")

    if enqueue_ingest_links is None:
        raise HTTPException(
            status_code=501,
            detail=(
                "Link ingestion task not wired. Implement app.tasks.ingest.enqueue_ingest_links(tenant, links)."
            ),
        )

    # normalize to plain dicts for Celery
    payload = [l.model_dump() for l in links]
    job_id = await enqueue_ingest_links(tenant, payload)
    return LinkIngestResponse(job_id=job_id, num_links=len(links))


# ---------------------------
# NCBI/PMID ingestion
# ---------------------------

class NcbiRequest(BaseModel):
    pmids: List[str] = Field(min_length=1)


class NcbiIngestResponse(BaseModel):
    job_id: str
    num_pmids: int


@router.post("/ncbi", response_model=NcbiIngestResponse)
async def ingest_ncbi(req: Request, body: NcbiRequest):
    tenant = req.state.tenant_id
    if enqueue_ingest_pmid is None:
        raise HTTPException(
            status_code=501,
            detail=(
                "PMID ingestion task not wired. Implement app.tasks.ingest.enqueue_ingest_pmid(tenant, pmids)."
            ),
        )
    job_id = await enqueue_ingest_pmid(tenant, body.pmids)
    return NcbiIngestResponse(job_id=job_id, num_pmids=len(body.pmids))


# ---------------------------
# Conditioned paper summarization (pinned context + user question)
# ---------------------------

class ConditionedSummaryItem(BaseModel):
    paper_id: str
    sent_id: Optional[int] = None  # when provided, use exact sentence spans; otherwise, whole paper


class ConditionedSummaryRequest(BaseModel):
    message: str = Field(min_length=1, max_length=8000)
    items: List[ConditionedSummaryItem] = Field(default_factory=list)
    options: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Optional controls, e.g. {max_paragraphs:int, allow_extra_retrieval:bool, "
            "token_budget:int, confidence_min:float}"
        ),
    )


class SupportSentence(BaseModel):
    paper_id: str
    sent_id: Optional[int] = None
    text: str
    subject: Optional[str] = None
    predicate: Optional[str] = None
    object: Optional[str] = None
    pmid: Optional[str] = None
    pmcid: Optional[str] = None
    page: Optional[int] = None


class ConditionedSummaryPara(BaseModel):
    text: str
    support: List[SupportSentence]  # sentences with SVO + metadata


class ConditionedSummaryResponse(BaseModel):
    paragraphs: List[ConditionedSummaryPara]


@router.post("/summarize_conditioned", response_model=ConditionedSummaryResponse)
async def summarize_conditioned_papers(req: Request, body: ConditionedSummaryRequest):
    """
    Summarize selected papers conditioned on the user’s question and pinned context.
    Returns paragraphs + the **most relevant source sentences** for each paragraph,
    including full SVO (subject/predicate/object) and provenance metadata (pmid/pmcid/page).
    """
    tenant = req.state.tenant_id
    # Delegate to your domain implementation (GPU-aware, RAG + RE augment)
    paragraphs = await summarize_conditioned(
        tenant=tenant,
        message=body.message,
        items=[it.model_dump() for it in body.items],
        options=body.options,
    )

    # Expect summarize_conditioned to return a list of dicts with:
    #   { "text": "...", "support": [ {paper_id, sent_id, text, subject, predicate, object, pmid, pmcid, page}, ... ] }
    # Map to the response model (validates structure strictly)
    normalized: List[ConditionedSummaryPara] = []
    for p in paragraphs:
        support_items = []
        for s in p.get("support", []) or []:
            support_items.append(
                SupportSentence(
                    paper_id=s.get("paper_id"),
                    sent_id=s.get("sent_id"),
                    text=s.get("text", ""),
                    subject=s.get("subject"),
                    predicate=s.get("predicate"),
                    object=s.get("object"),
                    pmid=s.get("pmid"),
                    pmcid=s.get("pmcid"),
                    page=s.get("page"),
                )
            )
        normalized.append(ConditionedSummaryPara(text=p.get("text", ""), support=support_items))

    return ConditionedSummaryResponse(paragraphs=normalized)

