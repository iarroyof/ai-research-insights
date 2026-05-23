from __future__ import annotations

from evals.lung_factuality_lab.src.schemas import ExtractedClaim, GoldClaim, MechanismGraph


def find_best_gold_claim(claim: ExtractedClaim, gold_claims: dict[str, GoldClaim]) -> tuple[GoldClaim | None, float]:
    best: GoldClaim | None = None
    best_score = 0.0
    claim_entities = {e.lower() for e in claim.entities}
    claim_text = claim.text.lower()
    for gold in gold_claims.values():
        entity_hits = sum(
            1
            for entity in gold.entities
            if entity.lower() in claim_entities or _node_present(entity, claim_text)
        )
        entity_score = entity_hits / max(1, len(gold.entities))
        text_score = sum(1 for term in _terms(gold.claim) if term in claim_text) / max(1, len(_terms(gold.claim)))
        score = max(entity_score, text_score)
        if score > best_score:
            best = gold
            best_score = score
    return best, round(best_score, 4)


def find_best_mechanism_graph(claims: list[ExtractedClaim], graphs: dict[str, MechanismGraph]) -> tuple[MechanismGraph | None, list[str], float]:
    text = " ".join(claim.text for claim in claims).lower()
    best: MechanismGraph | None = None
    best_missing: list[str] = []
    best_score = -1.0
    for graph in graphs.values():
        missing = [node for node in graph.required_nodes if not _node_present(node, text)]
        score = 1.0 - len(missing) / max(1, len(graph.required_nodes))
        if score > best_score:
            best = graph
            best_missing = missing
            best_score = score
    return best, best_missing, round(max(0.0, best_score), 4)


NODE_ALIASES = {
    "CAF/stromal fibroblast": ["caf", "stromal fibroblast", "fibroblast"],
    "MET/c-MET signaling": ["met", "c-met", "met/c-met", "met/c-met signaling"],
    "MET/c-MET": ["met", "c-met", "met/c-met"],
    "downstream signaling": ["downstream", "signaling", "pathway", "pathways"],
    "M2-like TAMs": ["m2-like tam", "m2 tam", "tam", "macrophage"],
    "M2-like polarization": ["m2-like", "m2", "polarization", "polarized"],
    "CD8 T cells": ["cd8", "cd8+", "t-cell", "t cell", "effector t"],
    "effector T cells/CD8+ T cells": ["cd8", "cd8+", "t-cell", "t cell", "effector t"],
    "effector T cells": ["effector t", "t-cell", "t cell", "cd8"],
    "immunosuppression": ["immunosuppression", "immune suppression", "suppress", "suppresses", "suppressed"],
    "immune escape": ["immune escape", "immune suppression", "immunosuppression", "impaired cytotoxic"],
    "HIF": ["hif", "hif-linked"],
    "HIF-1α/HIF signaling": ["hif", "hif-1", "hif-1a", "hif-1α", "hif signaling"],
    "PD-L1 or T-cell dysfunction": ["pd-l1", "pdl1", "t-cell dysfunction", "t cell dysfunction", "cytotoxic"],
    "MDSCs": ["mdsc", "mdscs", "myeloid-derived suppressor"],
    "Tregs": ["treg", "tregs", "regulatory t"],
    "ECM remodeling/stiffness": ["ecm", "matrix", "stiffness", "remodeling", "collagen"],
    "invasion or drug delivery or immune infiltration": ["invasion", "drug delivery", "immune infiltration", "motility"],
    "CAF heterogeneity": ["caf heterogeneity", "heterogeneity", "caf"],
    "context dependence": ["context", "context-dependent", "caveat", "not all", "depending"],
}


def _node_present(node: str, text: str) -> bool:
    aliases = NODE_ALIASES.get(node, [node])
    for alias in aliases:
        if alias.lower() in text:
            return True
    return False


def _terms(text: str) -> list[str]:
    import re

    stop = {"the", "and", "can", "with", "that", "from", "into", "contexts"}
    return [t for t in re.findall(r"[a-zA-Z][a-zA-Z0-9/-]{2,}", text.lower()) if t not in stop]
