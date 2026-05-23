from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass
from typing import Any, Iterable, List


CLAIM_TYPES_REQUIRING_CITATION = {
    "biomedical_fact",
    "method",
    "recommendation",
    "uncertainty",
}

NON_FACTUAL_STARTS = (
    "thanks",
    "thank you",
    "sure",
    "here is",
    "here are",
    "i can",
    "i will",
    "let me know",
    "please",
)

BIOMEDICAL_TERMS = {
    "aggregation",
    "aspirin",
    "biomarker",
    "cancer",
    "cardiovascular",
    "cell",
    "chemotherapy",
    "clinical",
    "disease",
    "expression",
    "gene",
    "inflammation",
    "inhibits",
    "insulin",
    "metformin",
    "mutation",
    "patient",
    "platelet",
    "protein",
    "risk",
    "sensitivity",
    "study",
    "thrombotic",
    "treatment",
    "tumor",
}

RELATION_PATTERNS = (
    r"\b(inhibits?|activates?|induces?|reduces?|increases?|decreases?|predicts?|improves?|worsens?|causes?|associates?|correlates?|binds?|expresses?|regulates?)\b",
    r"\b(is|are|was|were|has|have|had|uses?|used|reports?|reported)\b",
)

NEGATION_RE = re.compile(r"\b(no|not|never|without|absent|negative|fails?|cannot|can't|does not|do not|did not)\b", re.I)
SPECULATION_RE = re.compile(r"\b(may|might|could|possibly|possible|suggests?|suggested|likely|uncertain|unclear|appears?)\b", re.I)
CITATION_RE = re.compile(r"\b(PMID|PMCID|doi)\b|\[[0-9,\s-]+\]|\([A-Z][A-Za-z-]+ et al\.,? \d{4}\)", re.I)


@dataclass(frozen=True)
class AnswerSentence:
    sentence_id: str
    text: str
    index: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AtomicClaim:
    claim_id: str
    answer_sentence_id: str
    answer_sentence: str
    claim_text: str
    claim_type: str
    entities: list[str]
    relations: list[str]
    negation: bool
    speculation: str
    requires_citation: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def stable_id(*parts: str, prefix: str = "") -> str:
    digest = hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()[:24]
    return f"{prefix}{digest}" if prefix else digest


def _normalize_ws(text: str) -> str:
    return " ".join((text or "").replace("\n", " ").split())


def split_candidate_sentences(answer: str) -> list[AnswerSentence]:
    """Split assistant text into sentence candidates without breaking common biomedical abbreviations."""
    text = _normalize_ws(answer)
    if not text:
        return []

    protected = {
        "e.g.": "e<dot>g<dot>",
        "i.e.": "i<dot>e<dot>",
        "Fig.": "Fig<dot>",
        "Dr.": "Dr<dot>",
        "vs.": "vs<dot>",
        "et al.": "et al<dot>",
    }
    for src, dst in protected.items():
        text = text.replace(src, dst)

    raw_sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9(\[])|[\n\r]+", text)
    out: list[AnswerSentence] = []
    for raw in raw_sentences:
        sentence = raw.strip()
        for src, dst in protected.items():
            sentence = sentence.replace(dst, src)
        sentence = sentence.strip(" -")
        if not sentence:
            continue
        out.append(AnswerSentence(stable_id(str(len(out)), sentence, prefix="sent_"), sentence, len(out)))
    return out


def _tokens(text: str) -> list[str]:
    return [t.lower() for t in re.findall(r"[A-Za-z][A-Za-z0-9_-]{1,}", text or "")]


def _is_non_factual(sentence: str) -> bool:
    s = sentence.strip().lower()
    if not s:
        return True
    if s.endswith("?"):
        return True
    return any(s.startswith(prefix) for prefix in NON_FACTUAL_STARTS)


def _subject_prefix(sentence: str) -> str:
    match = re.match(
        r"^(.+?)\s+\b(inhibits?|activates?|induces?|reduces?|increases?|decreases?|predicts?|improves?|worsens?|causes?|associates?|correlates?|binds?|expresses?|regulates?|is|are|was|were|has|have|had|uses?|used|reports?|reported)\b",
        sentence,
        flags=re.I,
    )
    if not match:
        return ""
    prefix = match.group(1).strip(" ,;")
    if len(prefix.split()) > 8:
        return ""
    return prefix


def _finish_sentence(text: str) -> str:
    text = text.strip(" ,;")
    return text if not text or text[-1] in ".!?" else f"{text}."


def _split_compound_claims(sentence: str) -> list[str]:
    """Conservative atomizer for same-subject conjunctions."""
    sentence = sentence.strip()
    if not sentence:
        return []
    body = sentence[:-1] if sentence[-1] in ".!?" else sentence
    subject = _subject_prefix(body)
    if not subject:
        return [_finish_sentence(body)]

    parts = re.split(r"\s*;\s*|\s+\band\b\s+", body)
    if len(parts) <= 1:
        return [_finish_sentence(body)]

    claims: list[str] = []
    first = parts[0].strip()
    if first:
        claims.append(_finish_sentence(first))
    for part in parts[1:]:
        piece = part.strip()
        if not piece:
            continue
        if re.match(r"^(is|are|was|were|has|have|had|inhibits?|activates?|induces?|reduces?|increases?|decreases?|predicts?|improves?|worsens?|causes?|associates?|correlates?|binds?|expresses?|regulates?|uses?|used|reports?|reported)\b", piece, re.I):
            piece = f"{subject} {piece}"
        claims.append(_finish_sentence(piece))
    return claims or [_finish_sentence(body)]


def _extract_entities(text: str) -> list[str]:
    entities: list[str] = []
    seen: set[str] = set()
    for phrase in re.findall(r"\b[A-Z][A-Za-z0-9-]*(?:\s+[A-Z][A-Za-z0-9-]*){0,3}\b", text or ""):
        key = phrase.lower()
        if key not in seen and key not in {"the", "this", "these"}:
            seen.add(key)
            entities.append(phrase)
    toks = _tokens(text)
    for term in toks:
        if term in BIOMEDICAL_TERMS and term not in seen:
            seen.add(term)
            entities.append(term)
    return entities


def _extract_relations(text: str) -> list[str]:
    relations: list[str] = []
    seen: set[str] = set()
    for pattern in RELATION_PATTERNS:
        for match in re.finditer(pattern, text or "", flags=re.I):
            rel = match.group(1).lower()
            if rel not in seen:
                seen.add(rel)
                relations.append(rel)
    return relations


def _speculation(text: str) -> str:
    match = SPECULATION_RE.search(text or "")
    if not match:
        return "none"
    token = match.group(1).lower()
    if token in {"likely", "appears", "appear"}:
        return "likely"
    if token in {"uncertain", "unclear"}:
        return "uncertain"
    return "possible"


def classify_claim_type(claim_text: str) -> str:
    lower = claim_text.lower()
    toks = set(_tokens(claim_text))
    if CITATION_RE.search(claim_text):
        return "citation_statement"
    if _is_non_factual(claim_text):
        return "other"
    if {"method", "methods", "cohort", "assay", "sequencing", "trial", "randomized"} & toks:
        return "method"
    if {"should", "recommend", "recommended", "consider"} & toks:
        return "recommendation"
    if SPECULATION_RE.search(claim_text):
        return "uncertainty"
    if toks & BIOMEDICAL_TERMS or re.search(r"\b[A-Z0-9-]{2,}\b", claim_text):
        return "biomedical_fact"
    if re.search(r"\b(study|paper|analysis|result|finding|reports?|reported|shows?|showed)\b", lower):
        return "biomedical_fact"
    return "other"


def requires_citation(claim_text: str, claim_type: str | None = None) -> bool:
    claim_type = claim_type or classify_claim_type(claim_text)
    return claim_type in CLAIM_TYPES_REQUIRING_CITATION


def extract_atomic_claims(answer: str) -> list[AtomicClaim]:
    claims: list[AtomicClaim] = []
    for sentence in split_candidate_sentences(answer):
        if _is_non_factual(sentence.text):
            continue
        for claim_text in _split_compound_claims(sentence.text):
            claim_text = _finish_sentence(_normalize_ws(claim_text))
            if _is_non_factual(claim_text):
                continue
            claim_type = classify_claim_type(claim_text)
            claim = AtomicClaim(
                claim_id=stable_id(sentence.sentence_id, claim_text, prefix="claim_"),
                answer_sentence_id=sentence.sentence_id,
                answer_sentence=sentence.text,
                claim_text=claim_text,
                claim_type=claim_type,
                entities=_extract_entities(claim_text),
                relations=_extract_relations(claim_text),
                negation=bool(NEGATION_RE.search(claim_text)),
                speculation=_speculation(claim_text),
                requires_citation=requires_citation(claim_text, claim_type),
            )
            claims.append(claim)
    return claims


def claims_to_dicts(claims: Iterable[AtomicClaim]) -> list[dict[str, Any]]:
    return [c.to_dict() if isinstance(c, AtomicClaim) else dict(c) for c in claims]
