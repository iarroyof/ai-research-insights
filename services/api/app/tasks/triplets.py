# services/api/app/tasks/triplets.py
from __future__ import annotations

import csv
import io
import uuid

from celery import shared_task
from opensearchpy.helpers import bulk

from app.config import settings
from app.graph.neo_client import neo_session
from app.search.os_client import os_client, _index_prefix


def _csv_thresholds():
    """
    Return subject/object thresholds and confidence from new or old settings.
    """
    try:
        subj_thr = settings.csv.subject_thresholds
        obj_thr = settings.csv.object_thresholds
        conf_min = float(settings.csv.confidence_min)
        return subj_thr, obj_thr, conf_min
    except Exception:
        # old layout
        subj_min = float(settings.subject_ebio_min)
        obj_min = float(settings.object_ebio_min)
        conf_min = float(settings.confidence_min)
        # map to dicts for compatibility (EBio/NGen/Other)
        subj_thr = {"EBio": subj_min, "NGen": 0.7, "Other": 0.0}
        obj_thr = {"EBio": obj_min, "NGen": 0.7, "Other": 0.0}
        return subj_thr, obj_thr, conf_min


@shared_task(name="triplets.csv.ingest")
def triplets_csv_ingest(tenant: str, job_id: str, content: bytes) -> dict:
    """
    Ingest CSV rows into Neo4j + OpenSearch with probability-typed entities.
    """
    subj_thr, obj_thr, conf_min = _csv_thresholds()

    reader = csv.DictReader(io.StringIO(content.decode("utf-8")))
    ingested = rejected = 0

    cypher_batches: list[tuple[str, dict]] = []
    os_docs: list[dict] = []

    def pick_type(prob: dict, thr: dict) -> str:
        if float(prob.get("EBio", 0.0)) >= float(thr.get("EBio", 0.7)):
            return "biomedical"
        if float(prob.get("NGen", 0.0)) >= float(thr.get("NGen", 0.7)):
            return "generic"
        return "other"

    for row in reader:
        try:
            sent = (row.get("sentence_text") or "").strip()
            sub = (row.get("subject") or "").strip()
            pred = (row.get("relation") or "").strip()
            obj = (row.get("object") or "").strip()

            if not sent or not sub or not obj:
                rejected += 1
                continue

            sp = {
                "EBio": float(row.get("subject_probably_EBio", 0.0) or 0.0),
                "NGen": float(row.get("subject_probably_NGen", 0.0) or 0.0),
                "Other": float(row.get("subject_probably_otro", 0.0) or 0.0),
            }
            op = {
                "EBio": float(row.get("object_probably_EBio", 0.0) or 0.0),
                "NGen": float(row.get("object_probably_NGen", 0.0) or 0.0),
                "Other": float(row.get("object_probably_otro", 0.0) or 0.0),
            }

            conf = min(max(sp.values()), max(op.values()))
            if conf < conf_min:
                rejected += 1
                continue

            subject_type = pick_type(sp, subj_thr)
            object_type = pick_type(op, obj_thr)
            triple_id = str(uuid.uuid4())

            # Neo4j upserts (batched)
            cypher = """
            MERGE (s:Entity {tenant_id:$tenant, canonical:$s})
              ON CREATE SET s.types=[$stype], s.probs=$sp
            MERGE (o:Entity {tenant_id:$tenant, canonical:$o})
              ON CREATE SET o.types=[$otype], o.probs=$op
            MERGE (t:Triple {tenant_id:$tenant, id:$tid})
              ON CREATE SET t.subject=$s, t.predicate=$p, t.object=$o,
                            t.confidence=$conf, t.method="csv"
            MERGE (s)-[:AS_SUBJECT_OF]->(t)<-[:AS_OBJECT_OF]-(o)
            """
            params = {
                "tenant": tenant,
                "s": sub,
                "o": obj,
                "p": pred,
                "tid": triple_id,
                "conf": float(conf),
                "sp": sp,
                "op": op,
                "stype": subject_type,
                "otype": object_type,
            }
            cypher_batches.append((cypher, params))

            # OpenSearch doc
            os_docs.append(
                {
                    "_index": f"{_index_prefix()}{tenant}_triplets",
                    "_op_type": "index",
                    "_id": triple_id,
                    "triple_id": triple_id,
                    "sentence_text": sent,
                    "subject": sub,
                    "predicate": pred,
                    "object": obj,
                    "subject_probs": sp,
                    "object_probs": op,
                    "subject_entity_type": subject_type,
                    "object_entity_type": object_type,
                    "confidence": float(conf),
                }
            )
            ingested += 1
        except Exception:
            rejected += 1

    # Execute Neo4j batch
    if cypher_batches:
        with neo_session() as sess:
            tx = sess.begin_transaction()
            try:
                for cy, pa in cypher_batches:
                    tx.run(cy, **pa)
                tx.commit()
            except Exception:
                tx.rollback()
                raise

    # Execute OpenSearch bulk
    if os_docs:
        bulk(os_client(), os_docs, raise_on_error=True, refresh=True)

    return {"job_id": job_id, "ingested": ingested, "rejected": rejected}

