from typing import List, Dict

def build_prompt(message: str, snippets: List[Dict]):
    ctx = []
    for s in snippets:
        pmid = s.get("pmid","")
        pmcid = s.get("pmcid","")
        page = s.get("page","")
        sent = s.get("sent_id","")
        ctx.append(f"[paper:{s['paper_id']} pmid:{pmid} pmcid:{pmcid} page:{page} sent:{sent}] {s['text']}")
    context_block = "\n".join(ctx)
    return (
        "You are a scientific assistant. Answer using ONLY the provided context. "
        "Cite sources with (paper_id:sent_id). If uncertain, say so. "
        "Do not treat missing evidence as evidence that a relation has no plausible connection.\n\n"
        f"CONTEXT:\n{context_block}\n\nQUESTION: {message}\nANSWER:"
    )
