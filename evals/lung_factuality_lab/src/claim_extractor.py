from __future__ import annotations

import hashlib
import re

from evals.lung_factuality_lab.src.schemas import ExtractedClaim, Relation


ENTITY_ALIASES = {
    "CAF": ["caf", "cancer-associated fibroblast", "fibroblast"],
    "HGF": ["hgf", "hepatocyte growth factor"],
    "MET/c-MET": ["met", "c-met", "met/c-met"],
    "EMT": ["emt", "epithelial-mesenchymal"],
    "TAM": ["tam", "macrophage"],
    "CD8 T cells": ["cd8", "t cell", "t-cell"],
    "PD-L1": ["pd-l1", "pdl1"],
    "hypoxia": ["hypoxia", "hif"],
    "HIF": ["hif", "hif-1", "hif-1α"],
    "MDSC": ["mdsc", "myeloid-derived suppressor"],
    "Treg": ["treg", "regulatory t"],
    "ECM": ["ecm", "matrix", "stiffness", "collagen"],
    "immune escape": ["immune escape", "immune evasion"],
    "angiogenesis": ["angiogenesis", "angiogenic"],
    "lung cancer": ["lung cancer", "nsclc", "lung carcinoma"],
    "breast cancer": ["breast cancer"],
    "pancreatic cancer": ["pancreatic cancer"],
}

PREDICATES = {
    "activates": ["activates", "activate", "increases", "promotes", "stimulates"],
    "decreases": ["decrease", "decreases", "reduce", "reduces", "block", "blocks", "suppress", "suppresses", "inhibit", "inhibits"],
    "contributes_to": ["contributes", "drives", "supports", "promotes", "causes"],
    "irrelevant": ["irrelevant", "not needed", "unnecessary"],
}

NEGATORS = {"not", "never", "without", "doesn't", "does", "no"}


def extract_claims(answer: str, *, turn: int) -> list[ExtractedClaim]:
    sentences = _split_sentences(answer)
    out: list[ExtractedClaim] = []
    for idx, sentence in enumerate(sentences):
        if sentence.lower().startswith(("i would not agree", "i do not agree", "i would not")):
            continue
        lowered = sentence.lower()
        if sentence.endswith("?") and (len(sentence.split()) <= 16 or "treated as authoritative" in lowered or "guide the benchmark" in lowered):
            continue
        if lowered.startswith(("please clarify", "would you like")):
            continue
        entities = _entities(sentence)
        if not entities and len(sentence.split()) < 6:
            continue
        relation = _relation(sentence, entities)
        polarity = "negated" if set(_tokens(sentence)) & NEGATORS else "affirmed"
        claim_id = "claim_" + hashlib.sha256(f"{turn}\n{idx}\n{sentence}".encode("utf-8")).hexdigest()[:16]
        out.append(
            ExtractedClaim(
                claim_id=claim_id,
                turn=turn,
                text=sentence,
                entities=entities,
                relation=relation,
                polarity=polarity,
                confidence=0.7 if relation.predicate else 0.45,
            )
        )
    return out


def _split_sentences(text: str) -> list[str]:
    cleaned = " ".join((text or "").replace("\n", " ").split())
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", cleaned)
    return [part.strip(" -") for part in parts if part.strip()]


def _tokens(text: str) -> list[str]:
    return [tok.lower() for tok in re.findall(r"[A-Za-z][A-Za-z0-9/-]{1,}", text or "")]


def _entities(text: str) -> list[str]:
    lower = text.lower()
    out: list[str] = []
    for canonical, aliases in ENTITY_ALIASES.items():
        if any(_alias_in_text(lower, alias) for alias in aliases):
            out.append(canonical)
    return out


def _alias_in_text(lower_text: str, alias: str) -> bool:
    escaped = re.escape(alias.lower())
    return bool(re.search(rf"(?<![a-z0-9]){escaped}(?![a-z0-9])", lower_text))


def _relation(text: str, entities: list[str]) -> Relation:
    lower = text.lower()
    predicate = ""
    for canonical, aliases in PREDICATES.items():
        if any(alias in lower for alias in aliases):
            predicate = canonical
            break
    subject = entities[0] if entities else ""
    obj = entities[1] if len(entities) > 1 else ""
    if "HGF" in entities and "MET/c-MET" in entities:
        subject = "HGF"
        obj = "MET/c-MET signaling"
    if "caf" in lower and "hgf" in lower and not predicate:
        subject = "CAF"
        predicate = "secretes_or_produces"
        obj = "HGF"
    return Relation(subject=subject, predicate=predicate, object=obj)
