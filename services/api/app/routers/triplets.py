# services/api/app/routers/triplets.py
from __future__ import annotations

import csv
import io
import json
import uuid
from typing import List, Dict, Any, Optional

from fastapi import APIRouter, UploadFile, File, Request, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse

from app.config import settings

# Primary RE (REBEL) adapter
from app.integrations.extraction_client import extract_triples

# Triplets utilities in your repo
from app.triplets.util import (
    triples_for_sentences,   # async (tenant, sentences, confidence_min) -> existing triples
    upsert_triples_batch,    # async (tenant, List[dict]) -> writes Neo4j + OpenSearch
)

# Graph helpers (JSON for D3/clients)
from app.triplets.graph import subgraph_by_ids  # async (tenant, ids, confidence_min) -> {"nodes":[],"edges":[]}

# Sentence store to resolve (paper_id, sent_id) -> text + metadata
from app.search.store import get_sentences_by_ids  # async (tenant, items) -> [{paper_id, sent_id, text, pmid, pmcid, ...}]

# Optional: OpenIE client (kept for your /suggest endpoint parity)
try:
    from app.clients.openie import OpenIEClient
except Exception:  # pragma: no cover
    OpenIEClient = None

# Optional: triplet search (OpenSearch-backed); if not present, we’ll raise politely
try:
    from app.triplets.search import search_triplets  # async (tenant, q, confidence_min, entity_type, pmid) -> List[dict]
except Exception:  # pragma: no cover
    search_triplets = None


router = APIRouter(prefix="/triplets", tags=["triplets"])


# ---------- Helpers ----------

def _entity_type_from_probs(ebio: float, ngen: float, other: float, ebio_min: float) -> str:
    """
    Decide entity type using provided probabilities.
    - If EBio >= ebio_min → "biomedical"
    - Else if NGen is the max → "generic"
    - Else → "other"
    """
    if ebio >= ebio_min:
        return "biomedical"
    # pick the largest of the remaining as heuristic
    if ngen >= ebio and ngen >= other:
        return "generic"
    return "other"


def _parse_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


# ---------- CSV ingest (with thresholds & batching) ----------

@router.post("/csv")
async def upload_csv(req: Request, file: UploadFile = File(...)):
    """
    Ingest CSV of triplets. Expected columns (as per your sample):

    article_id, sentence_text, subject, relation, object,
    subject_probably_EBio, subject_probably_NGen, subject_probably_otro,
    object_probably_EBio,  object_probably_NGen,  object_probably_otro

    Behavior:
      - validates minimal fields
      - computes subject_entity_type & object_entity_type
      - computes confidence (min of max(subject_probs), max(object_probs))
      - applies thresholds: subject_ebio_min, object_ebio_min, confidence_min
      - bulk upserts to Neo4j + OpenSearch
    """
    tenant = req.state.tenant_id

    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Invalid file type (expected .csv)")

    raw = await file.read()
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8")))

    subj_thr = float(getattr(settings, "subject_ebio_min", 0.70))
    obj_thr = float(getattr(settings, "object_ebio_min", 0.70))
    conf_min = float(getattr(settings, "confidence_min", 0.60))

    batch: List[Dict[str, Any]] = []
    accepted = 0
    rejected = 0
    BATCH_SIZE = 1000

    for r in reader:
        sent_text = (r.get("sentence_text") or "").strip()
        subj = (r.get("subject") or "").strip()
        rel = (r.get("relation") or "").strip()
        obj = (r.get("object") or "").strip()
        if not sent_text or not subj or not obj:
            rejected += 1
            continue

        # parse superset of fields we know about
        sp_ebio = _parse_float(r.get("subject_probably_EBio"))
        sp_ngen = _parse_float(r.get("subject_probably_NGen"))
        sp_other = _parse_float(r.get("subject_probably_otro"))
        op_ebio = _parse_float(r.get("object_probably_EBio"))
        op_ngen = _parse_float(r.get("object_probably_NGen"))
        op_other = _parse_float(r.get("object_probably_otro"))

        # entity typing
        subject_entity_type = _entity_type_from_probs(sp_ebio, sp_ngen, sp_other, subj_thr)
        object_entity_type = _entity_type_from_probs(op_ebio, op_ngen, op_other, obj_thr)

        # confidence = min of max probs for subject/object (heuristic)
        s_max = max(sp_ebio, sp_ngen, sp_other)
        o_max = max(op_ebio, op_ngen, op_other)
        confidence = min(s_max, o_max)

        # thresholds
        if (sp_ebio < subj_thr and op_ebio < obj_thr) or (confidence < conf_min):
            rejected += 1
            continue

        row_doc = {
            "sentence_text": sent_text,
            "subject": subj,
            "predicate": rel,
            "object": obj,
            "subject_probs": {
                "EBio": sp_ebio,
                "NGen": sp_ngen,
                "other": sp_other,
            },
            "object_probs": {
                "EBio": op_ebio,
                "NGen": op_ngen,
                "other": op_other,
            },
            "subject_entity_type": subject_entity_type,
            "object_entity_type": object_entity_type,
            "confidence": confidence,
            "method": "csv",
            # Optional provenance
            "paper_id": r.get("article_id") or None,
            "pmid": r.get("pmid") or None,
            "pmcid": r.get("pmcid") or None,
        }

        batch.append(row_doc)
        accepted += 1

        if len(batch) >= BATCH_SIZE:
            await upsert_triples_batch(tenant, batch)
            batch.clear()

    if batch:
        await upsert_triples_batch(tenant, batch)

    job_id = f"csv:{uuid.uuid4()}"
    return {"accepted": accepted, "filtered": rejected, "job_id": job_id}


# ---------- Suggest (OpenIE/RE) ----------

@router.post("/suggest")
async def suggest_triplets(sentences: List[str]):
    """
    Run an extractor on provided sentences. Keeps your existing OpenIE endpoint but
    prefers REBEL if available.
    """
    # Prefer REBEL primary (already used elsewhere)
    try:
        result = await extract_triples(sentences)
        return result["triples"]

    except Exception:
        # Optional fallback to OpenIE client if wired
        if OpenIEClient is None:
            raise
        client = OpenIEClient()
        res = await client.extract(sentences)
        return res


# ---------- Build graph from selection (extract on demand if missing) ----------

@router.post("/graph/build")
async def build_graph_from_selection(
    req: Request,
    items: List[Dict[str, Any]],
    confidence_min: float = Query(default=None, description="Min confidence for triples"),
):
    """
    items: [{paper_id: str, sent_id: int}]
    Steps:
      1) Fetch sentences by ids
      2) If no triples exist, run RE on demand and persist (Neo4j + OpenSearch)
      3) Return triple_ids to visualize
    """
    tenant = req.state.tenant_id
    if confidence_min is None:
        confidence_min = float(getattr(settings, "confidence_min", 0.60))

    # 1) Resolve sentences
    sents = await get_sentences_by_ids(tenant, items)  # -> [{paper_id, sent_id, text, pmid, pmcid, ...}]

    # 2) Check existing triples
    existing = await triples_for_sentences(tenant, sents, confidence_min)
    have_by_sent = {e["sent_id"] for e in existing}

    missing = [s for s in sents if s.get("sent_id") not in have_by_sent]

    # Extract for missing sentences via REBEL
    new_triples: List[Dict[str, Any]] = []
    if missing:
        batch_texts = [m["text"] for m in missing]
        result = await extract_triples(batch_texts)
        triples = result["triples"]
        for m, t in zip(missing, triples):
            new_triples.append(
                    {
                        "sentence_text": m["text"],
                        "subject": t["subject"],
                        "predicate": t.get("relation") or t.get("predicate"),
                        "object": t["object"],
                        "confidence": float(t.get("score", 0.5)),
                        "method": "rebel",
                        "paper_id": m["paper_id"],
                        "sent_id": m["sent_id"],
                        "pmid": m.get("pmid"),
                        "pmcid": m.get("pmcid"),
                    }
                )
        if new_triples:
            await upsert_triples_batch(tenant, new_triples)

    # 3) Gather all triple_ids now
    existing2 = await triples_for_sentences(tenant, sents, confidence_min)
    triple_ids = [e["triple_id"] for e in existing2]
    return {"triple_ids": triple_ids, "count": len(triple_ids)}


# ---------- JSON graph API ----------

@router.get("/graph")
async def triplets_graph(
    req: Request,
    triple_ids: str = Query(..., description="Comma-separated list of triple IDs"),
    confidence_min: float = Query(default=None),
):
    tenant = req.state.tenant_id
    if confidence_min is None:
        confidence_min = float(getattr(settings, "confidence_min", 0.60))
    ids = [t for t in (triple_ids or "").split(",") if t.strip()]
    graph = await subgraph_by_ids(tenant, ids, confidence_min)
    return JSONResponse(graph)


# ---------- Minimal HTML viewer (for quick D3 previews / debug) ----------

@router.get("/graph/view", response_class=HTMLResponse)
async def graph_view(
    req: Request,
    triple_ids: str = Query(..., description="Comma-separated triple IDs"),
    confidence_min: float = Query(default=None),
):
    tenant = req.state.tenant_id
    if confidence_min is None:
        confidence_min = float(getattr(settings, "confidence_min", 0.60))
    ids = [t for t in (triple_ids or "").split(",") if t.strip()]
    graph = await subgraph_by_ids(tenant, ids, confidence_min)

    # Render a minimal static page; in production serve a real JS bundle.
    graph_json = json.dumps(graph)
    node_count = len(graph.get("nodes", []))
    edge_count = len(graph.get("edges", []))

    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Triplets Graph</title>
  <style>body {{ margin:0; font-family: ui-sans-serif, system-ui, sans-serif; }}</style>
</head>
<body>
  <div style="padding:12px">
    <h3>Triplets Graph</h3>
    <p><strong>Nodes:</strong> {node_count} &nbsp; <strong>Edges:</strong> {edge_count}</p>
    <details>
      <summary>Raw JSON</summary>
      <pre id="data" style="white-space: pre-wrap">{graph_json}</pre>
    </details>
  </div>
</body>
</html>
"""


# ---------- Triplet search (if wired) ----------

@router.get("/search")
async def triplets_search(
    req: Request,
    q: str = Query("", min_length=0, max_length=2000),
    confidence_min: float = Query(default=None),
    entity_type: Optional[str] = Query(default=None, regex="^(biomedical|generic|other)$"),
    pmid: Optional[str] = None,
):
    """
    Search triplets by text/fields. Requires app.triplets.search.search_triplets helper.
    """
    if search_triplets is None:
        raise HTTPException(status_code=501, detail="Triplet search is not enabled in this build.")
    tenant = req.state.tenant_id
    if confidence_min is None:
        confidence_min = float(getattr(settings, "confidence_min", 0.60))
    res = await search_triplets(tenant, q, confidence_min, entity_type, pmid)
    return {"results": res}

