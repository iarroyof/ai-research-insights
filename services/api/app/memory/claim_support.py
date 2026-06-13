from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Iterable

from app.memory.claims import AtomicClaim
from app.memory.comparability import compare_premise_hypothesis
from app.memory.evidence import EvidenceCandidate


NliFunc = Callable[[str, str], Awaitable[dict[str, Any]]]
NliBatchFunc = Callable[[list[tuple[str, str]]], Awaitable[list[dict[str, Any]]]]


@dataclass(frozen=True)
class EvidenceSupport:
    evidence_id: str
    status: str
    nli_label: str | None
    nli_scores: dict[str, float]
    comparability: dict[str, Any]
    evidence: dict[str, Any]
    nli_meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ClaimSupport:
    claim_id: str
    claim: str
    answer_sentence: str
    requires_citation: bool
    status: str
    best_entailment: float = 0.0
    max_contradiction: float = 0.0
    best_evidence_id: str | None = None
    candidate_count: int = 0
    prompt_supported: bool = False
    needs_user_confirmation: bool = False
    evidence: list[EvidenceSupport] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["evidence"] = [item.to_dict() if isinstance(item, EvidenceSupport) else item for item in self.evidence]
        return data


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _stable_id(*parts: str) -> str:
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()[:32]


def _as_dict(value: AtomicClaim | EvidenceCandidate | dict[str, Any]) -> dict[str, Any]:
    if isinstance(value, AtomicClaim) or isinstance(value, EvidenceCandidate):
        return value.to_dict()
    return dict(value)


def _scores(nli: dict[str, Any]) -> dict[str, float]:
    return {
        "entailment": float(nli.get("entailment", 0.0) or 0.0),
        "contradiction": float(nli.get("contradiction", 0.0) or 0.0),
        "neutral": float(nli.get("neutral", 0.0) or 0.0),
    }


def _thresholds() -> tuple[float, float]:
    try:
        from app.config import settings

        return (
            float(getattr(settings.memory, "nli_min_entailment", 0.55)),
            float(getattr(settings.memory, "nli_contradiction_threshold", 0.45)),
        )
    except Exception:
        return 0.55, 0.45


def _status_from_nli(scores: dict[str, float]) -> str:
    min_entailment, contradiction_threshold = _thresholds()
    if scores["contradiction"] >= contradiction_threshold:
        return "contradicted"
    if scores["entailment"] >= min_entailment and scores["entailment"] >= scores["neutral"]:
        return "entailed"
    return "unsupported"


async def assess_claim_support(
    claims: Iterable[AtomicClaim | dict[str, Any]],
    evidence_candidates: Iterable[EvidenceCandidate | dict[str, Any]],
    *,
    nli_func: NliFunc | None = None,
    nli_batch_func: NliBatchFunc | None = None,
    max_nli_pairs_per_claim: int = 8,
) -> list[ClaimSupport]:
    if nli_func is None:
        from app.memory.nli import classify_nli, classify_nli_batch

        nli_func = classify_nli
        if nli_batch_func is None:
            nli_batch_func = classify_nli_batch
    evidence_dicts = [_as_dict(e) for e in evidence_candidates]
    out: list[ClaimSupport] = []

    for claim_obj in claims:
        claim = _as_dict(claim_obj)
        claim_text = str(claim.get("claim_text") or claim.get("claim") or "").strip()
        if not claim_text:
            continue
        requires_cite = bool(claim.get("requires_citation", True))
        if not requires_cite:
            out.append(
                ClaimSupport(
                    claim_id=str(claim.get("claim_id") or _stable_id(claim_text)),
                    claim=claim_text,
                    answer_sentence=str(claim.get("answer_sentence") or claim_text),
                    requires_citation=False,
                    status="not_checked",
                    candidate_count=len(evidence_dicts),
                )
            )
            continue

        support_items: list[EvidenceSupport] = []
        pending_nli: list[tuple[dict[str, Any], str, dict[str, Any]]] = []
        checked = 0
        for evidence in evidence_dicts:
            premise = str(evidence.get("sentence_text") or evidence.get("text") or "").strip()
            if not premise:
                continue
            comparability = compare_premise_hypothesis(premise, claim_text, evidence=evidence, claim=claim).to_dict()
            if not comparability["comparable"]:
                support_items.append(
                    EvidenceSupport(
                        evidence_id=str(evidence.get("evidence_id") or _stable_id(premise)),
                        status="not_comparable",
                        nli_label=None,
                        nli_scores={"entailment": 0.0, "contradiction": 0.0, "neutral": 0.0},
                        comparability=comparability,
                        evidence=evidence,
                    )
                )
                continue
            if checked >= max_nli_pairs_per_claim:
                support_items.append(
                    EvidenceSupport(
                        evidence_id=str(evidence.get("evidence_id") or _stable_id(premise)),
                        status="not_checked",
                        nli_label=None,
                        nli_scores={"entailment": 0.0, "contradiction": 0.0, "neutral": 0.0},
                        comparability=comparability,
                        evidence=evidence,
                    )
                )
                continue
            checked += 1
            pending_nli.append((evidence, premise, comparability))

        if pending_nli:
            if nli_batch_func is not None:
                nli_results = await nli_batch_func([(premise, claim_text) for _, premise, _ in pending_nli])
            else:
                nli_results = [await nli_func(premise, claim_text) for _, premise, _ in pending_nli]
        else:
            nli_results = []

        for (evidence, premise, comparability), nli in zip(pending_nli, nli_results):
            scores = _scores(nli)
            label = str(nli.get("label") or max(scores, key=lambda k: scores[k]))
            nli_meta = {
                key: value
                for key, value in nli.items()
                if key not in {"label", "entailment", "contradiction", "neutral"}
            }
            support_items.append(
                EvidenceSupport(
                    evidence_id=str(evidence.get("evidence_id") or _stable_id(premise)),
                    status=_status_from_nli(scores),
                    nli_label=label,
                    nli_scores=scores,
                    nli_meta=nli_meta,
                    comparability=comparability,
                    evidence=evidence,
                )
            )

        entailed = [item for item in support_items if item.status == "entailed"]
        contradicted = [item for item in support_items if item.status == "contradicted"]
        checked_items = [item for item in support_items if item.status in {"entailed", "contradicted", "unsupported"}]

        best_entailment = max((item.nli_scores.get("entailment", 0.0) for item in support_items), default=0.0)
        max_contradiction = max((item.nli_scores.get("contradiction", 0.0) for item in support_items), default=0.0)
        if contradicted:
            status = "contradicted"
            best_evidence_id = max(contradicted, key=lambda item: item.nli_scores.get("contradiction", 0.0)).evidence_id
        elif entailed:
            status = "entailed"
            best_evidence_id = max(entailed, key=lambda item: item.nli_scores.get("entailment", 0.0)).evidence_id
        elif checked_items:
            status = "unsupported"
            best_evidence_id = None
        elif support_items:
            status = "unsupported"
            best_evidence_id = None
        else:
            status = "unsupported"
            best_evidence_id = None

        prompt_supported = any(
            item.status == "entailed" and bool(item.evidence.get("was_in_model_prompt"))
            for item in support_items
        )
        out.append(
            ClaimSupport(
                claim_id=str(claim.get("claim_id") or _stable_id(claim_text)),
                claim=claim_text,
                answer_sentence=str(claim.get("answer_sentence") or claim_text),
                requires_citation=True,
                status=status,
                best_entailment=round(best_entailment, 4),
                max_contradiction=round(max_contradiction, 4),
                best_evidence_id=best_evidence_id,
                candidate_count=len(evidence_dicts),
                prompt_supported=prompt_supported,
                needs_user_confirmation=status == "contradicted",
                evidence=support_items,
            )
        )
    return out


def build_evidence_table(
    *,
    answer_id: str,
    session_id: str,
    turn_index: int,
    claim_support: Iterable[ClaimSupport | dict[str, Any]],
    tenant: str | None = None,
) -> dict[str, Any]:
    claims = [item.to_dict() if isinstance(item, ClaimSupport) else dict(item) for item in claim_support]
    return {
        "doc_type": "evidence_table",
        "evidence_table_id": _stable_id(session_id, str(turn_index), answer_id),
        "tenant": tenant,
        "session_id": session_id,
        "turn_index": turn_index,
        "answer_id": answer_id,
        "claims": claims,
        "claim_count": len(claims),
        "status_counts": {status: sum(1 for c in claims if c.get("status") == status) for status in ("entailed", "contradicted", "unsupported", "not_comparable", "not_checked")},
        "created_at": _now(),
    }


def evidence_table_debug_payload(table: dict[str, Any]) -> dict[str, Any]:
    return {
        "evidence_table_id": table.get("evidence_table_id"),
        "answer_id": table.get("answer_id"),
        "claim_count": table.get("claim_count", 0),
        "status_counts": table.get("status_counts", {}),
        "claims": [
            {
                "claim_id": claim.get("claim_id"),
                "claim": claim.get("claim"),
                "status": claim.get("status"),
                "best_evidence_id": claim.get("best_evidence_id"),
                "best_entailment": claim.get("best_entailment", 0.0),
                "max_contradiction": claim.get("max_contradiction", 0.0),
                "needs_user_confirmation": claim.get("needs_user_confirmation", False),
                "nli": [
                    {
                        "evidence_id": item.get("evidence_id"),
                        "status": item.get("status"),
                        "label": item.get("nli_label"),
                        "scores": item.get("nli_scores", {}),
                        "metadata": item.get("nli_meta", {}),
                        "provider": (item.get("nli_meta") or {}).get("provider"),
                        "model": (item.get("nli_meta") or {}).get("model"),
                        "panel_success_count": (item.get("nli_meta") or {}).get("panel_success_count"),
                        "panel_size": (item.get("nli_meta") or {}).get("panel_size"),
                        "panel_agreement": (item.get("nli_meta") or {}).get("panel_agreement"),
                        "panel": (item.get("nli_meta") or {}).get("panel"),
                        "premise_sentence": (item.get("evidence") or {}).get("sentence_text"),
                        "hypothesis_claim": claim.get("claim"),
                        "source_sentence_id": (item.get("evidence") or {}).get("sent_id"),
                        "paper_id": (item.get("evidence") or {}).get("paper_id"),
                        "retrieval_score": (item.get("evidence") or {}).get("retrieval_score"),
                    }
                    for item in claim.get("evidence", [])[:5]
                ],
            }
            for claim in table.get("claims", [])
        ],
    }
