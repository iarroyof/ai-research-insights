from typing import List, Dict

def rrf_fuse(bm25: List[Dict], vec: List[Dict], k=50, c=60):
    scores = {}
    for i, d in enumerate(bm25[:k]):
        scores.setdefault(d["_id"], 0.0)
        scores[d["_id"]] += 1.0 / (c + i + 1)
    for i, d in enumerate(vec[:k]):
        scores.setdefault(d["_id"], 0.0)
        scores[d["_id"]] += 1.0 / (c + i + 1)
    order = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    seen = set()
    merged = []
    for doc_id, _ in order:
        if doc_id in seen:
            continue
        # pick from whichever list first
        for candidate in (bm25 + vec):
            if candidate["_id"] == doc_id:
                merged.append(candidate)
                seen.add(doc_id)
                break
    return merged
