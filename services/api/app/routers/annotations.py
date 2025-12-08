# services/api/app/routers/annotations.py
from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Request, HTTPException, Path, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator

import psycopg
from psycopg.rows import dict_row

from app.config import settings

router = APIRouter(prefix="/annotations", tags=["annotations"])


# ---------- Models ----------

class BBox(BaseModel):
    # PDF.js-style rectangle or our normalized box
    # x, y in page coordinate space; width/height optional (can be 0 for notes/likes)
    x: float = 0
    y: float = 0
    width: float = 0
    height: float = 0


class AnnotationCreate(BaseModel):
    paper_id: str = Field(min_length=1, max_length=128)
    sent_id: Optional[str] = Field(default=None, ge=0)
    kind: str = Field(pattern=r"^(highlight|note|like|unlike)$")
    payload: Dict[str, Any] = Field(default_factory=dict)
    page: Optional[int] = Field(default=None, ge=1)
    bbox: Optional[BBox] = None
    idempotency_key: Optional[str] = Field(
        default=None, description="Client-provided key for de-duplication."
    )

    @field_validator("payload")
    @classmethod
    def _payload_size_guard(cls, v: Dict[str, Any]):
        # keep payloads small (notes text, small metadata)
        blob = json.dumps(v, ensure_ascii=False)
        if len(blob) > 8_000:
            raise ValueError("payload too large (>8KB)")
        return v


class AnnotationOut(BaseModel):
    id: str
    paper_id: str
    sent_id: Optional[str]
    kind: str
    payload: Dict[str, Any]
    page: Optional[int]
    bbox: Optional[BBox]
    user_id: Optional[str] = None
    created_at: datetime


class BatchCreateRequest(BaseModel):
    items: List[AnnotationCreate] = Field(min_items=1, max_items=500)


class ListResponse(BaseModel):
    items: List[AnnotationOut]
    next_cursor: Optional[str] = None


class UpdateRequest(BaseModel):
    kind: Optional[str] = Field(default=None, pattern=r"^(highlight|note|like|unlike)$")
    payload: Optional[Dict[str, Any]] = None
    page: Optional[int] = Field(default=None, ge=1)
    bbox: Optional[BBox] = None


# ---------- PG helpers ----------

def _pg_dsn() -> str:
    """
    Accept either the old config (settings.pg_dsn) or the new (settings.db.dsn),
    and normalize SQLAlchemy-style DSNs (postgresql+psycopg://...) to psycopg form.
    """
    dsn = None
    if hasattr(settings, "pg_dsn"):
        dsn = settings.pg_dsn
    elif hasattr(settings, "db") and isinstance(settings.db, dict) and settings.db.get("dsn"):
        dsn = str(settings.postgres.dsn).replace("postgresql+psycopg://", "postgresql://")
    elif hasattr(settings, "db") and getattr(settings.db, "dsn", None):
        dsn = settings.db.dsn
    if not dsn:
        raise RuntimeError("Postgres DSN not configured")

    return dsn.replace("postgresql+psycopg://", "postgresql://")


async def _with_conn(tenant: str):
    """
    Async context manager yielding a psycopg AsyncConnection with
    the tenant GUC set for Postgres RLS policies.
    """
    conn = await psycopg.AsyncConnection.connect(_pg_dsn(), row_factory=dict_row)
    try:
        # Ensure the app.tenant_id GUC is set; align with your RLS policies
        await conn.execute("select set_config('app.tenant_id', %s, true)", (tenant,))
        yield conn
    finally:
        await conn.close()


# ---------- Routes ----------

@router.post("", response_model=AnnotationOut)
async def create_annotation(req: Request, body: AnnotationCreate):
    """
    Create a single annotation. Supports idempotency via `idempotency_key` if you
    add a unique index on (tenant_id, idempotency_key).
    """
    tenant = req.state.tenant_id
    user_id = req.headers.get("x-user-id")  # optional; store if provided

    ann_id = str(uuid.uuid4())
    q = """
    insert into annotations
      (id, tenant_id, user_id, paper_id, sent_id, kind, payload_jsonb, page, bbox, idempotency_key, created_at)
    values
      (%s, %s::uuid, %s, %s, %s, %s, %s::jsonb, %s, %s::jsonb, %s, now())
    returning id, paper_id, sent_id, kind, payload_jsonb as payload, page, bbox, user_id, created_at;
    """
    bbox_json = json.dumps(body.bbox.dict()) if body.bbox else None
    payload_json = json.dumps(body.payload, ensure_ascii=False)
    idem = body.idempotency_key

    async for conn in _with_conn(tenant):
        try:
            async with conn.cursor() as cur:
                await cur.execute(
                    q,
                    (
                        ann_id, tenant, user_id, body.paper_id, body.sent_id, body.kind,
                        payload_json, body.page, bbox_json, idem,
                    ),
                )
                rec = await cur.fetchone()
            await conn.commit()
        except psycopg.errors.UniqueViolation:
            # If you defined a unique index on (tenant_id, idempotency_key),
            # return the existing row for the same idempotency key:
            if not idem:
                raise
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    select id, paper_id, sent_id, kind, payload_jsonb as payload, page, bbox, user_id, created_at
                    from annotations
                    where tenant_id = %s::uuid and idempotency_key = %s
                    """,
                    (tenant, idem),
                )
                rec = await cur.fetchone()
            await conn.commit()
            if not rec:
                raise
        except Exception as e:  # pragma: no cover
            await conn.rollback()
            raise HTTPException(status_code=500, detail=f"DB error: {e}")

    # Normalize bbox if present
    if rec and rec.get("bbox"):
        try:
            rec["bbox"] = BBox.model_validate(rec["bbox"])
        except Exception:
            rec["bbox"] = None
    return AnnotationOut(**rec)


@router.post("/batch", response_model=ListResponse)
async def create_annotations_batch(req: Request, body: BatchCreateRequest):
    """
    Batch create (max 500). Each item may include an idempotency_key.
    """
    tenant = req.state.tenant_id
    user_id = req.headers.get("x-user-id")

    q = """
    insert into annotations
      (id, tenant_id, user_id, paper_id, sent_id, kind, payload_jsonb, page, bbox, idempotency_key, created_at)
    values
      (%s, %s::uuid, %s, %s, %s, %s, %s::jsonb, %s, %s::jsonb, %s, now())
    on conflict (tenant_id, idempotency_key) do update
      set user_id = excluded.user_id  -- no-op or update as you prefer
    returning id, paper_id, sent_id, kind, payload_jsonb as payload, page, bbox, user_id, created_at;
    """

    items_out: List[AnnotationOut] = []
    async for conn in _with_conn(tenant):
        try:
            async with conn.cursor() as cur:
                for item in body.items:
                    ann_id = str(uuid.uuid4())
                    payload_json = json.dumps(item.payload, ensure_ascii=False)
                    bbox_json = json.dumps(item.bbox.dict()) if item.bbox else None
                    idem = item.idempotency_key
                    await cur.execute(
                        q,
                        (
                            ann_id, tenant, user_id, item.paper_id, item.sent_id, item.kind,
                            payload_json, item.page, bbox_json, idem,
                        ),
                    )
                    rec = await cur.fetchone()
                    if rec and rec.get("bbox"):
                        try:
                            rec["bbox"] = BBox.model_validate(rec["bbox"])
                        except Exception:
                            rec["bbox"] = None
                    items_out.append(AnnotationOut(**rec))
            await conn.commit()
        except Exception as e:  # pragma: no cover
            await conn.rollback()
            raise HTTPException(status_code=500, detail=f"DB error: {e}")

    return ListResponse(items=items_out, next_cursor=None)


@router.get("", response_model=ListResponse)
async def list_annotations(
    req: Request,
    paper_id: Optional[str] = Query(default=None),
    sent_id: Optional[str] = Query(default=None, ge=0),
    kind: Optional[str] = Query(default=None, pattern=r"^(highlight|note|like|unlike)$"),
    limit: int = Query(default=100, ge=1, le=500),
    cursor: Optional[str] = Query(default=None, description="Opaque cursor for pagination"),
):
    """
    List annotations for the current tenant with optional filters.
    Pagination uses a simple (created_at, id) seek cursor.
    """
    tenant = req.state.tenant_id

    params: List[Any] = [tenant]
    where = ["tenant_id = %s::uuid"]

    if paper_id:
        where.append("paper_id = %s")
        params.append(paper_id)
    if sent_id is not None:
        where.append("sent_id = %s")
        params.append(sent_id)
    if kind:
        where.append("kind = %s")
        params.append(kind)

    # cursor format: "<iso8601>|<uuid>"
    if cursor:
        try:
            ts_str, last_id = cursor.split("|", 1)
            where.append("(created_at, id) < (%s::timestamptz, %s::uuid)")
            params.extend([ts_str, last_id])
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid cursor")

    sql = f"""
    select id, paper_id, sent_id, kind, payload_jsonb as payload, page, bbox, user_id, created_at
    from annotations
    where {' and '.join(where)}
    order by created_at desc, id desc
    limit %s
    """
    params.append(limit + 1)  # fetch one extra to decide next_cursor

    items: List[AnnotationOut] = []
    next_cursor_val: Optional[str] = None

    async for conn in _with_conn(tenant):
        try:
            async with conn.cursor() as cur:
                await cur.execute(sql, params)
                rows = await cur.fetchall()
        except Exception as e:  # pragma: no cover
            raise HTTPException(status_code=500, detail=f"DB error: {e}")

    if rows:
        if len(rows) > limit:
            last = rows[limit - 1]
            next_cursor_val = f"{last['created_at'].isoformat()}|{last['id']}"
            rows = rows[:limit]
        for r in rows:
            if r.get("bbox"):
                try:
                    r["bbox"] = BBox.model_validate(r["bbox"])
                except Exception:
                    r["bbox"] = None
            items.append(AnnotationOut(**r))

    return ListResponse(items=items, next_cursor=next_cursor_val)


@router.put("/{annotation_id}", response_model=AnnotationOut)
async def update_annotation(
    req: Request, annotation_id: str = Path(...), body: UpdateRequest = ...
):
    """
    Update kind/payload/page/bbox of an annotation (same tenant).
    """
    tenant = req.state.tenant_id

    sets: List[str] = []
    params: List[Any] = []

    if body.kind is not None:
        sets.append("kind = %s")
        params.append(body.kind)
    if body.payload is not None:
        sets.append("payload_jsonb = %s::jsonb")
        params.append(json.dumps(body.payload, ensure_ascii=False))
    if body.page is not None:
        sets.append("page = %s")
        params.append(body.page)
    if body.bbox is not None:
        sets.append("bbox = %s::jsonb")
        params.append(json.dumps(body.bbox.model_dump()))

    if not sets:
        raise HTTPException(status_code=400, detail="No fields to update")

    params.extend([tenant, annotation_id])
    sql = f"""
    update annotations
       set {', '.join(sets)}
     where tenant_id = %s::uuid and id = %s::uuid
     returning id, paper_id, sent_id, kind, payload_jsonb as payload, page, bbox, user_id, created_at
    """

    async for conn in _with_conn(tenant):
        try:
            async with conn.cursor() as cur:
                await cur.execute(sql, params)
                rec = await cur.fetchone()
                await conn.commit()
        except Exception as e:  # pragma: no cover
            await conn.rollback()
            raise HTTPException(status_code=500, detail=f"DB error: {e}")

    if not rec:
        raise HTTPException(status_code=404, detail="Annotation not found")

    if rec.get("bbox"):
        try:
            rec["bbox"] = BBox.model_validate(rec["bbox"])
        except Exception:
            rec["bbox"] = None
    return AnnotationOut(**rec)


@router.delete("/{annotation_id}")
async def delete_annotation(req: Request, annotation_id: str = Path(...)):
    """
    Delete an annotation (same tenant).
    """
    tenant = req.state.tenant_id
    async for conn in _with_conn(tenant):
        try:
            async with conn.cursor() as cur:
                await cur.execute(
                    "delete from annotations where tenant_id=%s::uuid and id=%s::uuid",
                    (tenant, annotation_id),
                )
                deleted = cur.rowcount
            await conn.commit()
        except Exception as e:  # pragma: no cover
            await conn.rollback()
            raise HTTPException(status_code=500, detail=f"DB error: {e}")

    if not deleted:
        raise HTTPException(status_code=404, detail="Annotation not found")
    return {"status": "ok", "deleted": 1}


@router.get("/export")
async def export_annotations_ndjson(
    req: Request,
    paper_id: Optional[str] = Query(default=None),
    kind: Optional[str] = Query(default=None, pattern=r"^(highlight|note|like|unlike)$"),
):
    """
    Export annotations as NDJSON (for training / RLHF pipelines).
    """
    tenant = req.state.tenant_id

    params: List[Any] = [tenant]
    where = ["tenant_id = %s::uuid"]
    if paper_id:
        where.append("paper_id = %s")
        params.append(paper_id)
    if kind:
        where.append("kind = %s")
        params.append(kind)

    sql = f"""
    select id, paper_id, sent_id, kind, payload_jsonb as payload, page, bbox, user_id, created_at
      from annotations
     where {' and '.join(where)}
     order by created_at asc, id asc
    """

    async def gen():
        async for conn in _with_conn(tenant):
            try:
                async with conn.cursor() as cur:
                    await cur.execute(sql, params)
                    async for row in cur:
                        # row is dict thanks to dict_row
                        if row.get("bbox") and not isinstance(row["bbox"], (dict, list)):
                            # ensure JSON-serializable
                            try:
                                row["bbox"] = json.loads(row["bbox"])
                            except Exception:
                                row["bbox"] = None
                        yield (json.dumps(row, default=str) + "\n")
            except Exception as e:  # pragma: no cover
                # surface as a final line with error; client can decide
                yield json.dumps({"error": str(e)}) + "\n"

    return StreamingResponse(gen(), media_type="application/x-ndjson")

