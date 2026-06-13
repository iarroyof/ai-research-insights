# services/api/app/memory/intent_router.py
"""Tier-1 zero-shot intent router (P-7).

Sits between the tier-0 lexical rules (`_is_context_poor` in search_agent.py)
and the tier-2 120b context_manager (`resolve_message_intent`). It classifies a
context-poor message into one of ROUTER_INTENT_LABELS with a confidence score so
the expensive 120b call is reserved for messages that need a query *rewrite* or
that the router cannot decide confidently.

This module classifies intent ONLY — it never rewrites the query. The caller
(`plan_auto_context`) short-circuits high-confidence ``prior_context`` (which
needs no rewrite) and escalates everything else to the 120b.

Backends, tried in order:
  1. NIM primary  — small generative model via ``LLMClient(agent="router")``.
  2. MNLI fallback — zero-shot entailment via ``app.services.zero_shot.score_labels``.
Returns None only when both backends fail (caller then uses the 120b / heuristic).

Why this beats the old lexical gate (`_is_followup_reference`): that was a pure
substring match over FOLLOWUP_CONTEXT_MARKERS, so it missed typos ("previuos"),
paraphrases ("expand on your last point"), and false-positived on standalone
questions that merely contained a marker word ("previous treatments for X").
Both backends here decide semantically.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
from typing import Any

from app.clients.llm import LLMClient
from app.prompts.agent_prompts import (
    ROUTER_INTENT_HYPOTHESES,
    ROUTER_INTENT_LABELS,
    router_system_prompt,
)
from app.services import zero_shot


def _conf_threshold() -> float:
    """Minimum router confidence to act without escalating to the 120b."""
    try:
        return float(os.getenv("ROUTER_CONF_THRESHOLD", "0.6"))
    except (TypeError, ValueError):
        return 0.6


# Resolved once at import; env-overridable for eval sweeps.
ROUTER_CONF_THRESHOLD: float = _conf_threshold()

_LABEL_RE = re.compile(r"prior_context|augment_prior|new_query")


def _recent_turn_text(notes: list[dict[str, Any]] | None) -> str:
    """Most recent conversation turn(s) from notes, used as the NLI premise /
    generative context. Empty string when no working-buffer turns are present."""
    for _n in (notes or [])[:8]:
        rt = _n.get("recent_turns")
        if rt:
            tail = [str(x) for x in list(rt)[-2:]]
            return "\n".join(tail)[:800]
    return ""


def _premise(message: str, notes: list[dict[str, Any]] | None) -> str:
    tail = _recent_turn_text(notes)
    return f"{tail}\nUser: {message}" if tail else f"User: {message}"


def _parse_nim(text: str) -> dict | None:
    """Parse the NIM router output: prefer JSON {intent, confidence}, else the
    first label token found in the text."""
    if not text:
        return None
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        try:
            data = json.loads(text[start : end + 1])
            intent = str(data.get("intent", "")).strip().lower()
            if intent in ROUTER_INTENT_LABELS:
                try:
                    conf = float(data.get("confidence"))
                except (TypeError, ValueError):
                    conf = 0.85
                return {
                    "intent": intent,
                    "confidence": min(1.0, max(0.0, conf)),
                    "source": "nim",
                }
        except (ValueError, TypeError):
            pass
    match = _LABEL_RE.search(text.lower())
    if match:
        return {"intent": match.group(0), "confidence": 0.7, "source": "nim"}
    return None


async def _classify_nim(message: str, notes: list[dict[str, Any]] | None) -> dict | None:
    messages = [
        {"role": "system", "content": router_system_prompt()},
        {"role": "user", "content": _premise(message, notes)},
    ]
    try:
        # max_tokens comes from agent_models.router (Nemotron needs headroom for
        # its reasoning pass before the final JSON; too small → empty content).
        text = await LLMClient().chat_once(messages, agent="router")
    except Exception as exc:  # noqa: BLE001 — any provider error → try fallback
        print(f"[WARN] intent router NIM backend failed: {exc}")
        return None
    return _parse_nim(text)


async def _classify_mnli(message: str, notes: list[dict[str, Any]] | None) -> dict | None:
    premise = _premise(message, notes)
    labels = list(ROUTER_INTENT_LABELS)
    hyps = [ROUTER_INTENT_HYPOTHESES[label] for label in labels]
    try:
        # score_labels is synchronous (blocking httpx + provider queue); never run
        # it directly on the event loop.
        scored = await asyncio.to_thread(zero_shot.score_labels, [premise], hyps)
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] intent router MNLI backend failed: {exc}")
        return None
    if not scored or not scored[0]:
        return None
    hyp_scores = scored[0]  # {hypothesis_text: probability}
    best_label, best_conf = None, -1.0
    for label, hyp in zip(labels, hyps):
        prob = float(hyp_scores.get(hyp, 0.0))
        if prob > best_conf:
            best_label, best_conf = label, prob
    if best_label is None:
        return None
    return {"intent": best_label, "confidence": best_conf, "source": "mnli"}


async def classify_intent_zeroshot(
    message: str,
    notes: list[dict[str, Any]] | None,
) -> dict | None:
    """Classify a context-poor message's intent. NIM primary, MNLI fallback.

    Returns ``{"intent": <label>, "confidence": 0..1, "source": "nim"|"mnli"}``
    or None when both backends fail. Intent only — does not rewrite the query.
    """
    result = await _classify_nim(message, notes)
    if result is not None:
        return result
    return await _classify_mnli(message, notes)
