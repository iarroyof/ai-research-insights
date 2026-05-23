from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, List


@dataclass(frozen=True)
class EvidenceCandidate:
    evidence_id: str
    source: str
    paper_id: str | None = None
    pmid: str | None = None
    pmcid: str | None = None
    title: str | None = None
    section: str | None = None
    sent_id: str | None = None
    sentence_text: str = ""
    window_text: str = ""
    mesh_terms: list[str] = field(default_factory=list)
    disease_terms: list[str] = field(default_factory=list)
    retrieval_score: float = 0.0
    triplet_links: list[dict[str, Any]] = field(default_factory=list)
    was_in_model_prompt: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def stable_evidence_id(*parts: str) -> str:
    return "ev_" + hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()[:24]


def _first_text(item: dict[str, Any], keys: Iterable[str]) -> str:
    for key in keys:
        value = item.get(key)
        if value is not None and str(value).strip():
            return " ".join(str(value).split())
    return ""


def _first_value(item: dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        value = item.get(key)
        if value not in (None, ""):
            return value
    return None


def _as_str(value: Any) -> str | None:
    if value in (None, ""):
        return None
    return str(value)


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def evidence_from_mapping(
    item: dict[str, Any],
    *,
    source: str,
    was_in_model_prompt: bool = False,
) -> EvidenceCandidate | None:
    sentence_text = _first_text(
        item,
        (
            "sentence_text",
            "origin_sentence",
            "source_sentence",
            "text",
            "snippet",
            "content",
        ),
    )
    if not sentence_text:
        return None

    paper_id = _as_str(_first_value(item, ("paper_id", "article_id", "paper", "id")))
    sent_id = _as_str(_first_value(item, ("sent_id", "sentence_id", "chunk_id")))
    pmid = _as_str(_first_value(item, ("pmid", "PMID")))
    pmcid = _as_str(_first_value(item, ("pmcid", "PMCID")))
    section = _as_str(_first_value(item, ("section", "section_title", "heading")))
    title = _as_str(_first_value(item, ("title", "paper_title", "article_title")))
    window_text = _first_text(item, ("window_text", "source_window", "context", "text", "sentence_text"))
    score = _first_value(item, ("retrieval_score", "_score", "score", "confidence"))

    triplet_links: list[dict[str, Any]] = []
    if source == "triplet" or any(k in item for k in ("subject", "relation", "predicate", "object")):
        triplet_links.append(
            {
                "subject": item.get("subject"),
                "relation": item.get("relation") or item.get("predicate"),
                "object": item.get("object"),
                "confidence": item.get("confidence"),
            }
        )
    for link in _as_list(item.get("triplet_links")):
        if isinstance(link, dict):
            triplet_links.append(link)

    evidence_id = stable_evidence_id(
        source,
        paper_id or "",
        pmid or "",
        pmcid or "",
        sent_id or "",
        sentence_text[:240],
    )
    return EvidenceCandidate(
        evidence_id=evidence_id,
        source=source,
        paper_id=paper_id,
        pmid=pmid,
        pmcid=pmcid,
        title=title,
        section=section,
        sent_id=sent_id,
        sentence_text=sentence_text,
        window_text=window_text or sentence_text,
        mesh_terms=[str(v) for v in _as_list(item.get("mesh_terms"))],
        disease_terms=[str(v) for v in _as_list(item.get("disease_terms"))],
        retrieval_score=float(score or 0.0),
        triplet_links=triplet_links,
        was_in_model_prompt=was_in_model_prompt,
    )


def _append_unique(out: list[EvidenceCandidate], seen: set[str], candidate: EvidenceCandidate | None) -> None:
    if not candidate or candidate.evidence_id in seen:
        return
    seen.add(candidate.evidence_id)
    out.append(candidate)


def gather_evidence_candidates(
    *,
    prompt_context: Iterable[dict[str, Any]] | None = None,
    pinned_snippets: Iterable[dict[str, Any]] | None = None,
    source_sentences: Iterable[dict[str, Any]] | None = None,
    triplet_results: Iterable[dict[str, Any]] | None = None,
) -> list[EvidenceCandidate]:
    """Normalize prompt, pinned, source, and triplet-linked sentences into evidence candidates."""
    out: list[EvidenceCandidate] = []
    seen: set[str] = set()

    for item in prompt_context or []:
        source = str(item.get("source") or "prompt_context")
        _append_unique(out, seen, evidence_from_mapping(item, source=source, was_in_model_prompt=True))

    for item in pinned_snippets or []:
        _append_unique(out, seen, evidence_from_mapping(item, source="pinned", was_in_model_prompt=True))

    for item in source_sentences or []:
        source = str(item.get("source") or "memory")
        _append_unique(out, seen, evidence_from_mapping(item, source=source, was_in_model_prompt=False))

    for item in triplet_results or []:
        _append_unique(out, seen, evidence_from_mapping(item, source="triplet", was_in_model_prompt=bool(item.get("was_in_model_prompt"))))

    return out


def evidence_to_dicts(candidates: Iterable[EvidenceCandidate]) -> list[dict[str, Any]]:
    return [c.to_dict() if isinstance(c, EvidenceCandidate) else dict(c) for c in candidates]
