# services/api/app/routers/triplets.py
from __future__ import annotations

import csv
import io
import json
import uuid
from typing import List, Dict, Any, Optional

from fastapi import APIRouter, UploadFile, File, Request, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse

from app.ner import classify_entities, get_classifier
from app.neo4j_sync import sync_batch_to_neo4j, ensure_neo4j_constraints
from app.config import settings

# Primary RE (REBEL) adapter
from app.integrations.extraction_client import extract_triples

# Triplets utilities in your repo
from app.triplets.util import (
    triples_for_sentences,
    triples_for_papers,
    upsert_triples_batch,
)

# Graph helpers (JSON for D3/clients)
from app.triplets.graph import subgraph_by_ids

# Sentence store to resolve (paper_id, sent_id) -> text + metadata
from app.search.store import get_sentences_by_ids

# Optional: triplet search (OpenSearch-backed)
try:
    from app.triplets.search import search_triplets
except Exception:
    search_triplets = None


router = APIRouter(prefix="/triplets", tags=["triplets"])


# ---------- Helpers ----------

def _entity_type_from_probs(ebio: float, ngen: float, other: float, ebio_min: float) -> str:
    """
    Decide entity type using provided probabilities.
    """
    if ebio >= ebio_min:
        return "biomedical"
    if ngen >= ebio and ngen >= other:
        return "generic"
    return "other"


def _parse_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


# ---------- CSV ingest ----------

@router.post("/csv")
async def upload_csv(req: Request, file: UploadFile = File(...)):
    """
    Ingest CSV of triplets with validation and batching.
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

        sp_ebio = _parse_float(r.get("subject_probably_EBio"))
        sp_ngen = _parse_float(r.get("subject_probably_NGen"))
        sp_other = _parse_float(r.get("subject_probably_otro"))
        op_ebio = _parse_float(r.get("object_probably_EBio"))
        op_ngen = _parse_float(r.get("object_probably_NGen"))
        op_other = _parse_float(r.get("object_probably_otro"))

        subject_entity_type = _entity_type_from_probs(sp_ebio, sp_ngen, sp_other, subj_thr)
        object_entity_type = _entity_type_from_probs(op_ebio, op_ngen, op_other, obj_thr)

        s_max = max(sp_ebio, sp_ngen, sp_other)
        o_max = max(op_ebio, op_ngen, op_other)
        confidence = min(s_max, o_max)

        if (sp_ebio < subj_thr and op_ebio < obj_thr) or (confidence < conf_min):
            rejected += 1
            continue

        row_doc = {
            "sentence_text": sent_text,
            "subject": subj,
            "predicate": rel,
            "object": obj,
            "subject_probs": {"EBio": sp_ebio, "NGen": sp_ngen, "other": sp_other},
            "object_probs": {"EBio": op_ebio, "NGen": op_ngen, "other": op_other},
            "subject_entity_type": subject_entity_type,
            "object_entity_type": object_entity_type,
            "confidence": confidence,
            "method": "csv",
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


# ---------- Suggest (RE) ----------

@router.post("/suggest")
async def suggest_triplets(sentences: List[str]):
    """
    Run the REBEL extractor on provided sentences.
    """
    result = await extract_triples(sentences)
    return result["triples"]


# ---------- Build graph from selection ----------

@router.post("/graph/build")
async def build_graph_from_selection(
    req: Request,
    items: List[Dict[str, Any]],
    confidence_min: float = Query(default=None, description="Min confidence for triples"),
    ebio_min: float = Query(default=0.3, description="Min EBio probability for biomedical relevance"),
    top_k: int = Query(default=50, description="Max number of triplets to return"),
):
    """
    Build graph from selected items using semantic search and quality filtering.

    This endpoint:
    1. Extracts key terms from selected sentences
    2. Performs semantic search for relevant triplets
    3. Applies quality filters (stopwords, deduplication, biomedical priority)
    4. Returns top-k most relevant triplets
    """
    from app.triplets.filters import search_triplets_semantic, extract_key_terms

    tenant = req.state.tenant_id
    if confidence_min is None:
        confidence_min = float(getattr(settings.csv.thresholds, "confidence_min", 0.60))

    # Resolve sentences if needed
    if items and isinstance(items[0], dict) and items[0].get("text"):
        sents = items
    else:
        sents = await get_sentences_by_ids(tenant, items)

    # Derive paper_ids from sentences
    paper_ids = sorted({
        s.get("paper_id") or (
            f"{s['pmcid']}.txt"
            if s.get("pmcid") and not str(s.get("paper_id", "")).endswith(".txt")
            else None
        )
        for s in sents
        if s.get("paper_id") or s.get("pmcid")
    })
    paper_ids = [pid for pid in paper_ids if pid]

    # Extract key terms from selected sentences for semantic search
    query = extract_key_terms(sents)

    # Semantic search with quality filtering
    filtered_triplets = await search_triplets_semantic(
        tenant=tenant,
        query=query,
        paper_ids=paper_ids,
        confidence_min=confidence_min,
        ebio_min=ebio_min,
        top_k=top_k,
    )
    
    triple_ids = [t.get("_id") for t in filtered_triplets if t.get("_id")]

    return {
        "triple_ids": triple_ids,
        "count": len(triple_ids),
        "debug": {
            "mode": "semantic_search_filtered",
            "items_in": len(items),
            "paper_ids": paper_ids,
            "query_terms": query[:200] + "..." if len(query) > 200 else query,
            "triples_found": len(triple_ids),
            "confidence_min": confidence_min,
            "ebio_min": ebio_min,
            "top_k": top_k,
        },
    }

# ---------- JSON graph API ----------

@router.get("/graph")
async def triplets_graph(
    req: Request,
    triple_ids: str = Query(..., description="Comma-separated list of triple IDs"),
    confidence_min: float = Query(default=None),
):
    """
    Get graph data as JSON (for programmatic access).
    """
    tenant = req.state.tenant_id
    if confidence_min is None:
        confidence_min = float(getattr(settings, "confidence_min", 0.60))
    
    ids = [t.strip() for t in (triple_ids or "").split(",") if t.strip()]
    
    if not ids:
        raise HTTPException(status_code=400, detail="No triple IDs provided")
    
    graph = await subgraph_by_ids(tenant, ids, confidence_min)
    return JSONResponse(graph)


# ---------- HTML viewer (browser-friendly) ----------

@router.get("/graph/view", tags=["graph"])
async def graph_view(
    req: Request,
    triple_ids: str = Query(..., description="Comma-separated triple IDs"),
    confidence_min: float = Query(default=None),
    tenant: str = Query(default=None, description="Tenant ID (fallback if header not present)"),
):
    """
    Interactive vis.js graph viewer with entity type filtering.
    """
    tenant_id = getattr(req.state, "tenant_id", None) or tenant or "default"
    
    if confidence_min is None:
        confidence_min = float(getattr(settings, "confidence_min", 0.60))

    ids = [t.strip() for t in (triple_ids or "").split(",") if t.strip()]

    if not ids:
        return HTMLResponse(content="<h3>Error: No triple IDs provided</h3>", status_code=400)

    try:
        graph = await subgraph_by_ids(tenant_id, ids, confidence_min)
        graph_json = json.dumps(graph)
        
        return HTMLResponse(content=f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Sabia Interactive Knowledge Graph</title>
    <script src="https://unpkg.com/vis-network@9.1.6/dist/vis-network.min.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #f5f5f5; }}
        
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px 30px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .header h1 {{ font-size: 24px; margin-bottom: 8px; }}
        .header .stats {{ opacity: 0.9; font-size: 14px; }}
        
        .controls {{
            background: white;
            padding: 20px 30px;
            border-bottom: 1px solid #e0e0e0;
            display: flex;
            gap: 20px;
            flex-wrap: wrap;
            align-items: center;
        }}
        .control-group {{ display: flex; flex-direction: column; gap: 5px; }}
        .control-group label {{ font-size: 12px; color: #666; font-weight: 500; }}
        .control-group select,
        .control-group input {{ padding: 8px 12px; border: 1px solid #ddd; border-radius: 6px; font-size: 14px; }}
        .control-group select {{ min-width: 200px; }}
        .control-group input[type="range"] {{ width: 200px; }}
        .slider-value {{ display: inline-block; min-width: 45px; font-weight: 600; color: #667eea; }}
        
        button {{
            padding: 8px 16px;
            background: #667eea;
            color: white;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-size: 14px;
            font-weight: 500;
            transition: background 0.2s;
        }}
        button:hover {{ background: #5568d3; }}
        
        #network {{
            width: 100%;
            height: calc(100vh - 200px);
            background: white;
            border-top: 3px solid #667eea;
        }}
        
        .legend {{
            position: fixed;
            bottom: 20px;
            right: 20px;
            background: white;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            max-width: 250px;
            max-height: 400px;
            overflow-y: auto;
        }}
        .legend h4 {{ margin-bottom: 10px; font-size: 14px; color: #333; }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 8px;
            margin: 6px 0;
            font-size: 12px;
        }}
        .legend-color {{
            width: 16px;
            height: 16px;
            border-radius: 3px;
            border: 1px solid #ddd;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🔬 Sabia Biomedical Knowledge Graph</h1>
        <div class="stats" id="stats">Loading...</div>
    </div>
    
    <div class="controls">
        <div class="control-group">
            <label>Filter by Entity Type</label>
            <select id="typeFilter">
                <option value="">All Types</option>
            </select>
        </div>
        
        <div class="control-group">
            <label>Confidence Threshold: <span class="slider-value" id="confValue">{confidence_min:.0%}</span></label>
            <input type="range" id="confSlider" min="0" max="100" value="{int(confidence_min * 100)}" step="5">
        </div>
        
        <div class="control-group">
            <label>Search Entities</label>
            <input type="text" id="search" placeholder="Type to search...">
        </div>
        
        <button onclick="resetFilters()">Reset Filters</button>
    </div>
    
    <div id="network"></div>
    
    <div class="legend">
        <h4>Entity Types</h4>
        <div id="legendContent"></div>
    </div>

    <script>
        const rawData = {graph_json};
        
        const entityColors = {{
            'gene_or_gene_product': '#FF6B6B',
            'protein': '#FF6B6B',
            'simple_chemical': '#4ECDC4',
            'chemical': '#4ECDC4',
            'disease': '#FFE66D',
            'cancer': '#FFE66D',
            'cell': '#95E1D3',
            'tissue': '#95E1D3',
            'organ': '#95E1D3',
            'organism': '#A8E6CF',
            'organization': '#FFA07A',
            'person': '#DDA15E',
            'geopolitical_entity': '#DDA15E',
            'location': '#DDA15E',
            'entity': '#B8B8D1'
        }};
        
        const entityLabels = {{
            'gene_or_gene_product': 'Gene/Protein',
            'simple_chemical': 'Chemical',
            'disease': 'Disease',
            'cancer': 'Cancer',
            'cell': 'Cell/Tissue',
            'organism': 'Organism',
            'organization': 'Organization',
            'person': 'Person/Location',
            'entity': 'Other'
        }};
        
        // Prepare nodes
        const nodes = rawData.nodes.map(n => ({{
            id: n.id,
            label: n.label.length > 40 ? n.label.substring(0, 37) + '...' : n.label,
            title: `<b>${{n.label}}</b><br>Type: ${{n.type}}`,
            color: entityColors[n.type] || '#B8B8D1',
            type: n.type,
            fullLabel: n.label,
            font: {{ size: 14 }},
            shape: 'dot',
            size: 20
        }}));
        
        // Prepare edges
        const edges = rawData.edges.map(e => ({{
            id: e.id,
            from: e.source,
            to: e.target,
            label: e.label,
            title: `Relation: ${{e.label}}<br>Confidence: ${{(e.confidence * 100).toFixed(1)}}%`,
            confidence: e.confidence,
            arrows: 'to',
            color: {{ color: '#999', highlight: '#667eea' }},
            width: 2,
            smooth: {{ type: 'curvedCW', roundness: 0.2 }}
        }}));
        
        // Initialize network
        const container = document.getElementById('network');
        const data = {{ nodes: new vis.DataSet(nodes), edges: new vis.DataSet(edges) }};
        const options = {{
            physics: {{
                enabled: true,
                barnesHut: {{
                    gravitationalConstant: -3000,
                    centralGravity: 0.3,
                    springLength: 150,
                    springConstant: 0.04
                }},
                stabilization: {{ iterations: 200 }}
            }},
            interaction: {{
                hover: true,
                tooltipDelay: 100,
                hideEdgesOnDrag: true,
                hideEdgesOnZoom: true
            }},
            nodes: {{
                borderWidth: 2,
                borderWidthSelected: 4
            }},
            edges: {{
                font: {{ size: 11, align: 'middle' }}
            }}
        }};
        
        const network = new vis.Network(container, data, options);
        
        // Update stats
        document.getElementById('stats').textContent = 
            `${{nodes.length}} entities • ${{edges.length}} relationships • Confidence ≥ {confidence_min:.0%}`;
        
        // Populate type filter
        const types = [...new Set(nodes.map(n => n.type))].sort();
        const typeFilter = document.getElementById('typeFilter');
        types.forEach(type => {{
            const option = document.createElement('option');
            option.value = type;
            option.textContent = entityLabels[type] || type;
            typeFilter.appendChild(option);
        }});
        
        // Populate legend
        const legendContent = document.getElementById('legendContent');
        Object.entries(entityLabels).forEach(([key, label]) => {{
            if (types.includes(key)) {{
                const item = document.createElement('div');
                item.className = 'legend-item';
                item.innerHTML = `
                    <div class="legend-color" style="background: ${{entityColors[key]}}"></div>
                    <span>${{label}}</span>
                `;
                legendContent.appendChild(item);
            }}
        }});
        
        // Filter functions
        function applyFilters() {{
            const selectedType = typeFilter.value;
            const searchText = document.getElementById('search').value.toLowerCase();
            const confThreshold = parseInt(document.getElementById('confSlider').value) / 100;
            
            nodes.forEach(node => {{
                const typeMatch = !selectedType || node.type === selectedType;
                const searchMatch = !searchText || node.fullLabel.toLowerCase().includes(searchText);
                const visible = typeMatch && searchMatch;
                data.nodes.update({{ id: node.id, hidden: !visible }});
            }});
            
            edges.forEach(edge => {{
                const confMatch = edge.confidence >= confThreshold;
                const sourceVisible = !data.nodes.get(edge.from).hidden;
                const targetVisible = !data.nodes.get(edge.to).hidden;
                data.edges.update({{ id: edge.id, hidden: !confMatch || !sourceVisible || !targetVisible }});
            }});
        }}
        
        function resetFilters() {{
            typeFilter.value = '';
            document.getElementById('search').value = '';
            document.getElementById('confSlider').value = {int(confidence_min * 100)};
            document.getElementById('confValue').textContent = '{confidence_min:.0%}';
            applyFilters();
        }}
        
        // Event listeners
        typeFilter.addEventListener('change', applyFilters);
        document.getElementById('search').addEventListener('input', applyFilters);
        document.getElementById('confSlider').addEventListener('input', e => {{
            const val = parseInt(e.target.value) / 100;
            document.getElementById('confValue').textContent = (val * 100).toFixed(0) + '%';
            applyFilters();
        }});
        
        // Click to highlight neighbors
        network.on('click', params => {{
            if (params.nodes.length > 0) {{
                const nodeId = params.nodes[0];
                const connectedNodes = network.getConnectedNodes(nodeId);
                const connectedEdges = network.getConnectedEdges(nodeId);
                
                nodes.forEach(n => {{
                    const opacity = n.id === nodeId || connectedNodes.includes(n.id) ? 1 : 0.2;
                    data.nodes.update({{ id: n.id, opacity }});
                }});
            }} else {{
                nodes.forEach(n => data.nodes.update({{ id: n.id, opacity: 1 }}));
            }}
        }});
    </script>
</body>
</html>""")
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return HTMLResponse(content=f"<h3>Error: {{str(e)}}</h3>", status_code=500)

# ---------- Triplet search ----------

@router.get("/search")
async def triplets_search(
    req: Request,
    q: str = Query("", min_length=0, max_length=2000),
    confidence_min: float = Query(default=None),
    entity_type: Optional[str] = Query(default=None, regex="^(biomedical|generic|other)$"),
    pmid: Optional[str] = None,
):
    """
    Search triplets by text/fields.
    """
    if search_triplets is None:
        raise HTTPException(status_code=501, detail="Triplet search is not enabled in this build.")
    
    tenant = req.state.tenant_id
    if confidence_min is None:
        confidence_min = float(getattr(settings, "confidence_min", 0.60))
    
    res = await search_triplets(tenant, q, confidence_min, entity_type, pmid)
    return {"results": res}

@router.post("/classify", tags=["classification"])
async def classify_entities_endpoint(
    req: Request,
    body: dict,
) -> dict:
    """
    Classify entities using dual NER pipeline.
    """
    entities = body.get("entities", [])
    if not entities:
        raise HTTPException(status_code=400, detail="No entities provided")
    
    classifier = get_classifier()
    
    if not classifier.is_ready:
        raise HTTPException(
            status_code=503,
            detail="NER models not loaded. Please install models first."
        )
    
    results = classify_entities(entities)
    
    return {
        "results": [r.to_dict() for r in results],
        "models_available": classifier.available_models,
    }


@router.post("/sync", tags=["sync"])
async def sync_to_neo4j(
    req: Request,
    tenant: str = Query(default=None),
    max_triplets: Optional[int] = Query(default=None),
) -> dict:
    """
    Manually trigger Neo4j sync for existing triplets.
    """
    tenant_id = getattr(getattr(req, "state", None), "tenant_id", None) or tenant or "default"
    
    # Determine index name
    index = f"triplets_{tenant_id}"
    
    try:
        # Ensure Neo4j constraints exist
        await ensure_neo4j_constraints()
        
        # Run sync
        stats = await sync_batch_to_neo4j(
            tenant=tenant_id,
            index=index,
            batch_size=1000,
            max_triplets=max_triplets,
        )
        
        return {
            "status": "success",
            "tenant": tenant_id,
            "index": index,
            "stats": stats,
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Sync failed: {str(e)}")
