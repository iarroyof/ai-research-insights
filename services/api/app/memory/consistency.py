from __future__ import annotations

import hashlib
import re
from typing import Any, Iterable

from app.memory.rewards import important_terms, terms


NEGATION_TERMS = {"no", "not", "never", "without", "absent", "negative", "fails", "cannot", "can't", "doesn't"}


def _stable_id(*parts: str) -> str:
    return hashlib.sha256("\n".join(str(p) for p in parts).encode("utf-8")).hexdigest()[:24]


def _text(value: Any) -> str:
    return " ".join(str(value or "").split())


def _merge_unique(*items: Iterable[str], limit: int = 24) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for group in items:
        for item in group or []:
            value = str(item or "").strip().lower()
            if not value or value in seen:
                continue
            seen.add(value)
            out.append(value)
            if len(out) >= limit:
                return out
    return out


def correction_terms(message: str) -> dict[str, list[str]]:
    """Extract generic user steering: terms to avoid and terms to prefer.

    This is intentionally domain-agnostic. It detects natural correction
    patterns like "not X, Y", "instead of X", "rather than X", and
    "pivot from X to Y" without encoding biomedical-specific topics.
    """
    text = _text(message)
    lower = text.lower()
    avoided_chunks: list[str] = []
    preferred_chunks: list[str] = []

    for pattern in (
        r"\bpivot\s+from\s+(.+?)\s+to\s+(.+?)(?:[.;]|$)",
        r"\binstead\s+of\s+(.+?)(?:,\s*|\s+use\s+|\s+focus\s+on\s+|\s+prefer\s+)(.+?)(?:[.;]|$)",
        r"\brather\s+than\s+(.+?)(?:,\s*|\s+use\s+|\s+focus\s+on\s+|\s+prefer\s+)(.+?)(?:[.;]|$)",
        r"\bnot\s+(.+?)\s+but\s+(.+?)(?:[.;]|$)",
    ):
        for match in re.finditer(pattern, lower):
            avoided_chunks.append(match.group(1))
            preferred_chunks.append(match.group(2))

    for match in re.finditer(r"\b(not|avoid|exclude)\s+([a-z0-9 _-]{3,80})(?:[.;,]|$)", lower):
        avoided_chunks.append(match.group(2))

    return {
        "avoid_terms": _merge_unique(*(important_terms(chunk, 8) for chunk in avoided_chunks), limit=18),
        "preferred_terms": _merge_unique(*(important_terms(chunk, 8) for chunk in preferred_chunks), limit=18),
    }


def build_conversation_frame(
    *,
    existing: dict[str, Any] | None,
    question: str,
    answer: str,
    claim_support: list[dict[str, Any]],
    turn_index: int,
) -> dict[str, Any]:
    existing = dict(existing or {})
    correction = correction_terms(question)
    entailed_claims = [
        _text(item.get("claim"))
        for item in claim_support or []
        if item.get("status") == "entailed" and _text(item.get("claim"))
    ]
    contradicted_claims = [
        _text(item.get("claim"))
        for item in claim_support or []
        if item.get("status") == "contradicted" and _text(item.get("claim"))
    ]
    unsupported_claims = [
        _text(item.get("claim"))
        for item in claim_support or []
        if item.get("status") == "unsupported" and _text(item.get("claim"))
    ]

    active_terms = _merge_unique(
        correction["preferred_terms"],
        important_terms(question, 14),
        *(important_terms(claim, 8) for claim in entailed_claims[:4]),
        existing.get("active_terms") or [],
        limit=28,
    )
    avoided_terms = _merge_unique(
        correction["avoid_terms"],
        existing.get("avoided_terms") or [],
        limit=24,
    )
    corrections = list(existing.get("corrections") or [])
    if correction["avoid_terms"] or correction["preferred_terms"]:
        corrections.insert(
            0,
            {
                "turn_index": turn_index,
                "text": question[:700],
                "avoid_terms": correction["avoid_terms"],
                "preferred_terms": correction["preferred_terms"],
            },
        )

    supported = _merge_claims(entailed_claims, existing.get("supported_claims") or [], limit=12)
    contradicted = _merge_claims(contradicted_claims, existing.get("contradicted_claims") or [], limit=10)
    unsupported = _merge_claims(unsupported_claims, existing.get("unsupported_claims") or [], limit=10)

    summary_bits = []
    if active_terms:
        summary_bits.append("active=" + ", ".join(active_terms[:12]))
    if avoided_terms:
        summary_bits.append("avoid=" + ", ".join(avoided_terms[:10]))
    if supported:
        summary_bits.append("supported_claims=" + str(len(supported)))
    if contradicted:
        summary_bits.append("contradicted_claims=" + str(len(contradicted)))

    return {
        "doc_type": "conversation_frame",
        "frame_id": existing.get("frame_id") or "frame_" + _stable_id("conversation_frame", existing.get("session_id", "")),
        "turn_index": turn_index,
        "summary": "; ".join(summary_bits)[:1200],
        "active_terms": active_terms,
        "avoided_terms": avoided_terms,
        "corrections": corrections[:8],
        "supported_claims": supported,
        "contradicted_claims": contradicted,
        "unsupported_claims": unsupported,
        "latest_question_terms": important_terms(question, 16),
        "latest_answer_terms": important_terms(answer, 16),
    }


def _merge_claims(new_claims: list[str], old_claims: list[Any], *, limit: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in list(new_claims) + list(old_claims or []):
        if isinstance(raw, dict):
            text = _text(raw.get("claim") or raw.get("text"))
            doc = dict(raw)
        else:
            text = _text(raw)
            doc = {"claim": text}
        if not text:
            continue
        key = " ".join(important_terms(text, 14))
        if key in seen:
            continue
        seen.add(key)
        doc["claim"] = text[:1000]
        out.append(doc)
        if len(out) >= limit:
            break
    return out


def render_conversation_frame(frame: dict[str, Any] | None) -> str:
    if not frame:
        return ""
    lines = ["Active conversation frame:"]
    active = frame.get("active_terms") or []
    avoided = frame.get("avoided_terms") or []
    if active:
        lines.append("- Stay aligned with: " + ", ".join(str(t) for t in active[:16]))
    if avoided:
        lines.append("- Avoid returning to retired/off-topic terms unless the user explicitly asks: " + ", ".join(str(t) for t in avoided[:14]))
    supported = frame.get("supported_claims") or []
    if supported:
        lines.append("- Preserve evidence-supported claims:")
        for item in supported[:5]:
            claim = item.get("claim") if isinstance(item, dict) else str(item)
            if claim:
                lines.append(f"  - {str(claim)[:320]}")
    corrections = frame.get("corrections") or []
    if corrections:
        latest = corrections[0]
        if isinstance(latest, dict) and latest.get("text"):
            lines.append("- Latest user steering: " + str(latest["text"])[:360])
    lines.append("- If new evidence conflicts with preserved claims or user steering, state the uncertainty and ask which fact should be authoritative.")
    return "\n".join(lines)


def frame_alignment(frame: dict[str, Any] | None, answer: str) -> dict[str, Any]:
    if not frame:
        return {"frame_alignment": 0.0, "frame_drift_penalty": 0.0, "avoided_terms_hit": [], "active_terms_hit": []}
    answer_terms = set(terms(answer))
    active_terms = set(str(t).lower() for t in frame.get("active_terms") or [])
    avoided_terms = set(str(t).lower() for t in frame.get("avoided_terms") or [])
    active_hits = sorted(answer_terms & active_terms)
    avoided_hits = sorted(answer_terms & avoided_terms)
    alignment = len(active_hits) / max(1, min(len(active_terms), 12))
    drift_penalty = min(1.0, len(avoided_hits) / 4.0)
    if active_terms and not active_hits:
        drift_penalty = max(drift_penalty, 0.25)
    return {
        "frame_alignment": round(max(0.0, min(1.0, alignment)), 4),
        "frame_drift_penalty": round(drift_penalty, 4),
        "avoided_terms_hit": avoided_hits[:12],
        "active_terms_hit": active_hits[:12],
    }


def longitudinal_consistency_report(
    *,
    question: str,
    answer: str,
    claim_support: list[dict[str, Any]],
    prior_supported_claims: list[dict[str, Any]],
    frame: dict[str, Any] | None,
) -> dict[str, Any]:
    alignment = frame_alignment(frame, answer)
    current_contradictions = [item for item in claim_support or [] if item.get("status") == "contradicted"]
    unsupported = [item for item in claim_support or [] if item.get("status") == "unsupported"]
    prior_conflicts = _prior_claim_conflicts(claim_support or [], prior_supported_claims or [])
    warnings: list[dict[str, Any]] = []
    if alignment["frame_drift_penalty"] > 0:
        warnings.append(
            {
                "type": "frame_drift",
                "message": "The answer appears to drift from the active conversation frame or reused retired terms.",
                "avoided_terms_hit": alignment["avoided_terms_hit"],
                "active_terms_hit": alignment["active_terms_hit"],
            }
        )
    if current_contradictions:
        warnings.append(
            {
                "type": "source_contradiction",
                "message": "One or more current answer claims contradict available source evidence.",
                "claim_count": len(current_contradictions),
            }
        )
    if prior_conflicts:
        warnings.append(
            {
                "type": "prior_memory_conflict",
                "message": "One or more current answer claims appear inconsistent with prior evidence-supported memory.",
                "conflicts": prior_conflicts[:3],
            }
        )
    penalty = (
        0.35 * min(1.0, len(current_contradictions) / max(1, len(claim_support or [])))
        + 0.20 * min(1.0, len(unsupported) / max(1, len(claim_support or [])))
        + 0.25 * min(1.0, len(prior_conflicts) / 2.0)
        + 0.20 * alignment["frame_drift_penalty"]
    )
    return {
        **alignment,
        "current_contradicted_claim_count": len(current_contradictions),
        "current_unsupported_claim_count": len(unsupported),
        "prior_memory_conflict_count": len(prior_conflicts),
        "prior_memory_conflicts": prior_conflicts[:5],
        "warnings": warnings,
        "longitudinal_penalty": round(min(1.0, penalty), 4),
    }


def _prior_claim_conflicts(current_claims: list[dict[str, Any]], prior_claims: list[dict[str, Any]]) -> list[dict[str, Any]]:
    conflicts: list[dict[str, Any]] = []
    for current in current_claims:
        text = _text(current.get("claim"))
        if not text:
            continue
        current_terms = set(important_terms(text, 24))
        current_neg = bool(set(terms(text)) & NEGATION_TERMS)
        for prior in prior_claims:
            prior_text = _text(prior.get("claim") or prior.get("sentence_text") or prior.get("text"))
            if not prior_text:
                continue
            prior_terms = set(important_terms(prior_text, 24))
            overlap = len(current_terms & prior_terms) / max(1, min(len(current_terms), len(prior_terms)))
            prior_neg = bool(set(terms(prior_text)) & NEGATION_TERMS)
            if overlap >= 0.45 and current_neg != prior_neg:
                conflicts.append(
                    {
                        "claim": text[:500],
                        "prior_claim": prior_text[:500],
                        "reason": "high lexical overlap with opposite negation polarity",
                    }
                )
                break
    return conflicts
