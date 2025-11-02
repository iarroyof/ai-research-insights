# services/api/app/summarize/conditioned.py

from __future__ import annotations
from typing import List, Dict, Any
from app.rag.context import fetch_pinned_snippets
from app.triplets.util import triples_for_sentences

async def summarize_conditioned(tenant: str, message: str, items: List[dict], options: Dict[str, Any]):
    """
    1) Resolve pinned snippets
    2) Build a 'summarize conditioned' prompt (user msg + pinned)
    3) Call LLM (non-stream is fine here; or collect stream)
    4) For each paragraph, re-rank and attach top-k support sentences + SVO (from triplets)
    """
    # Lazy import to avoid circular dependency during module initialization
    from app.clients.llm import stream_completion
    
    pinned = await fetch_pinned_snippets(tenant, items)
    # (Very) simple prompt for now; you may want a stricter template
    context_block = "\n".join([f"- {s['text']}" for s in pinned])
    prompt = f"""You are summarizing research findings conditioned on the pinned context.

Context:
{context_block}

User question:
{message}

Write a concise, well-structured summary in 2-4 paragraphs. Reference the most relevant source sentences for each paragraph (we will attach them after)."""

    # Here we could use a non-stream call; reuse stream and collect:
    text = ""
    async for chunk in stream_completion(prompt):
        text += chunk

    # Very naive paragraph split
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]

    # Attach support (top-k sentences; use triplets_for_sentences for SVO)
    support_base = await triples_for_sentences(tenant, pinned, confidence_min=0.1)
    # Build fast maps (TODO: real retrieval/ranking per paragraph)
    para_objs = []
    for p in paras:
        # naive: attach first N supporting sentences with their SVOs
        support = []
        for s in pinned[:5]:
            svos = [t for t in support_base if t.get("sent_id") == s.get("sent_id")]
            support.append({
                "sentence": s["text"],
                "paper_id": s["paper_id"], "title": s.get("title"),
                "pmid": s.get("pmid"), "pmcid": s.get("pmcid"),
                "page": s.get("page"), "sent_id": s.get("sent_id"),
                "svos": [{"subject": t.get("subject"), "predicate": t.get("predicate"), "object": t.get("object"),
                          "confidence": t.get("confidence")} for t in svos]
            })
        para_objs.append({"text": p, "support": support})
    return para_objs
