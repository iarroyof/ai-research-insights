from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any

from app.memory.rewards import lexical_overlap


BLOCKING_NONE = "none"
NEGATION_RE = re.compile(r"\b(no|not|never|without|absent|negative|fails?|cannot|can't|does not|do not|did not)\b", re.I)
TEMPORAL_RE = re.compile(r"\b(acute|chronic|early|late|baseline|after|before|post[- ]?treatment|pre[- ]?treatment|\d+\s*(day|week|month|year)s?)\b", re.I)
SPECIES_RE = re.compile(r"\b(mouse|mice|murine|rat|rats|human|patients?|cell line|in vitro|in vivo|xenograft|organoid)\b", re.I)
CELL_LINE_RE = re.compile(r"\b[A-Z]{1,5}[- ]?\d{1,4}[A-Z]?\b")
SECTION_WEAK = {"introduction", "background", "discussion"}

STOP = {
    "the",
    "and",
    "for",
    "that",
    "this",
    "with",
    "from",
    "into",
    "what",
    "when",
    "where",
    "which",
    "would",
    "could",
    "should",
    "about",
    "there",
    "their",
    "study",
    "studies",
    "reports",
    "reported",
    "shows",
    "showed",
    "used",
    "using",
    "risk",
    "response",
    "disease",
}

DISEASE_TERMS = {
    "cancer",
    "carcinoma",
    "tumor",
    "tumour",
    "cardiovascular",
    "diabetes",
    "insulin",
    "thrombosis",
    "thrombotic",
    "inflammation",
    "infection",
}

RELATION_TERMS = {
    "inhibit",
    "inhibits",
    "inhibited",
    "activate",
    "activates",
    "reduce",
    "reduces",
    "increase",
    "increases",
    "decrease",
    "decreases",
    "predict",
    "predicts",
    "improve",
    "improves",
    "cause",
    "causes",
    "associate",
    "associated",
    "correlate",
    "correlates",
    "bind",
    "binds",
    "regulate",
    "regulates",
    "use",
    "uses",
}


@dataclass(frozen=True)
class ComparabilityResult:
    comparable: bool
    score: float
    reasons: list[str]
    blocking_mismatch: str = BLOCKING_NONE

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _tokens(text: str) -> list[str]:
    return [t.lower() for t in re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", text or "")]


def _content_terms(text: str) -> set[str]:
    return {t for t in _tokens(text) if t not in STOP}


def _terms_matching(text: str, choices: set[str]) -> set[str]:
    toks = set(_tokens(text))
    return {t for t in toks if t in choices}


def _relations(text: str) -> set[str]:
    rels = _terms_matching(text, RELATION_TERMS)
    out: set[str] = set()
    for rel in rels:
        if rel.endswith("s"):
            rel = rel[:-1]
        if rel.endswith("ed"):
            rel = rel[:-2]
        out.add(rel)
    return out


def _named_entities(text: str) -> set[str]:
    entities = {m.group(0).lower() for m in re.finditer(r"\b[A-Z][A-Za-z0-9-]*(?:\s+[A-Z][A-Za-z0-9-]*){0,3}\b", text or "")}
    entities |= {t for t in _content_terms(text) if t in {"aspirin", "metformin", "platelet", "aggregation", "biomarker", "chemotherapy"}}
    return entities


def _overlap(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / max(1, len(a | b))


def _mismatch(a: set[str], b: set[str]) -> bool:
    return bool(a and b and not (a & b))


def compare_premise_hypothesis(
    premise: str,
    hypothesis: str,
    *,
    evidence: dict[str, Any] | None = None,
    claim: dict[str, Any] | None = None,
) -> ComparabilityResult:
    reasons: list[str] = []
    premise_terms = _content_terms(premise)
    hypothesis_terms = _content_terms(hypothesis)
    term_overlap = _overlap(premise_terms, hypothesis_terms)
    lexical = lexical_overlap(premise, hypothesis)

    p_entities = _named_entities(premise)
    h_entities = set(str(e).lower() for e in (claim or {}).get("entities", []) if e) or _named_entities(hypothesis)
    entity_overlap = _overlap(p_entities, h_entities)

    p_disease = _terms_matching(premise, DISEASE_TERMS)
    h_disease = _terms_matching(hypothesis, DISEASE_TERMS)
    p_rel = _relations(premise)
    h_rel = set(str(r).lower().rstrip("s") for r in (claim or {}).get("relations", []) if r) or _relations(hypothesis)

    score = 0.20 * lexical + 0.25 * term_overlap + 0.35 * max(entity_overlap, _overlap(premise_terms, h_entities)) + 0.20 * _overlap(p_rel, h_rel)

    blocking = BLOCKING_NONE
    if _mismatch(p_disease, h_disease):
        blocking = "disease"
        reasons.append("disease_mismatch")
        score -= 0.25
    if h_entities and p_entities and not (h_entities & p_entities):
        blocking = "entity" if blocking == BLOCKING_NONE else blocking
        reasons.append("entity_mismatch")
        score -= 0.35
    if p_rel and h_rel and not (p_rel & h_rel):
        reasons.append("relation_mismatch")
        score -= 0.15

    p_species = {m.group(0).lower() for m in SPECIES_RE.finditer(premise or "")}
    h_species = {m.group(0).lower() for m in SPECIES_RE.finditer(hypothesis or "")}
    if _mismatch(p_species, h_species):
        blocking = "species" if blocking == BLOCKING_NONE else blocking
        reasons.append("species_or_population_mismatch")
        score -= 0.20

    p_cell = {m.group(0).lower().replace(" ", "-") for m in CELL_LINE_RE.finditer(premise or "")}
    h_cell = {m.group(0).lower().replace(" ", "-") for m in CELL_LINE_RE.finditer(hypothesis or "")}
    if _mismatch(p_cell, h_cell):
        blocking = "cell_line" if blocking == BLOCKING_NONE else blocking
        reasons.append("cell_line_mismatch")
        score -= 0.20

    if bool(NEGATION_RE.search(premise or "")) != bool(NEGATION_RE.search(hypothesis or "")):
        reasons.append("negation_mismatch")
        score += 0.05

    p_time = {m.group(0).lower() for m in TEMPORAL_RE.finditer(premise or "")}
    h_time = {m.group(0).lower() for m in TEMPORAL_RE.finditer(hypothesis or "")}
    if _mismatch(p_time, h_time):
        blocking = "temporality" if blocking == BLOCKING_NONE else blocking
        reasons.append("temporality_mismatch")
        score -= 0.15

    section = str((evidence or {}).get("section") or "").lower()
    if section in SECTION_WEAK:
        reasons.append("weak_section_context")
        score -= 0.05

    score = round(max(0.0, min(1.0, score)), 4)
    if score >= 0.65:
        reasons.append("high_overlap")
    elif score >= 0.35:
        reasons.append("low_but_usable_overlap")
    else:
        if blocking == BLOCKING_NONE:
            blocking = "entity" if h_entities or p_entities else "relation"
        reasons.append("insufficient_overlap")

    return ComparabilityResult(
        comparable=score >= 0.35,
        score=score,
        reasons=reasons,
        blocking_mismatch=blocking if score < 0.35 or blocking != BLOCKING_NONE else BLOCKING_NONE,
    )
