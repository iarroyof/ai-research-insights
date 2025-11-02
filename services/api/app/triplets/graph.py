# services/api/app/triplets/graph.py
from __future__ import annotations

from typing import List, Dict, Any
from app.graph.neo_client import neo_session


async def subgraph_by_ids(tenant: str, triple_ids: List[str], confidence_min: float) -> Dict[str, Any]:
    """
    Returns a small subgraph JSON for visualization (D3-friendly).
    """
    cypher = """
    MATCH (t:Triple {tenant_id: $tenant})
    WHERE t.id IN $ids AND t.confidence >= $cmin
    OPTIONAL MATCH (s:Entity)-[:AS_SUBJECT_OF]->(t)<-[:AS_OBJECT_OF]-(o:Entity)
    OPTIONAL MATCH (t)-[:APPEARS_IN]->(sent:Sentence)-[:PART_OF]->(p:Paper)
    RETURN t, s, o, sent, p
    """
    nodes: dict[str, dict] = {}
    edges: list[dict] = []

    with neo_session() as sess:
        recs = sess.run(cypher, tenant=tenant, ids=triple_ids, cmin=confidence_min)
        for r in recs:
            t = r.get("t")
            if t:
                tid = t["id"]
                nodes[tid] = {"id": tid, "type": "triple", **dict(t)}

            s = r.get("s")
            if s:
                sid = s["id"]
                nodes[sid] = {"id": sid, "type": "entity", **dict(s)}
                edges.append({"source": sid, "target": tid, "type": "AS_SUBJECT_OF"})

            o = r.get("o")
            if o:
                oid = o["id"]
                nodes[oid] = {"id": oid, "type": "entity", **dict(o)}
                edges.append({"source": oid, "target": tid, "type": "AS_OBJECT_OF"})

            sent = r.get("sent")
            if sent:
                seid = sent["id"]
                nodes[seid] = {"id": seid, "type": "sentence", **dict(sent)}
                edges.append({"source": tid, "target": seid, "type": "APPEARS_IN"})

            p = r.get("p")
            if p:
                pid = p["id"]
                nodes[pid] = {"id": pid, "type": "paper", **dict(p)}
                if sent:
                    edges.append({"source": seid, "target": pid, "type": "PART_OF"})

    return {"nodes": list(nodes.values()), "edges": edges}

