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


# ── Configurable constants (no hardcoded literals — ARCHITECTURE.md rule 13) ──
# All behavioural knobs are env-overridable named constants resolved once at
# import, matching the os.getenv pattern used by zero_shot.py and nli.py.

def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return max(1, int(os.getenv(name, str(default))))
    except (TypeError, ValueError):
        return default


# Minimum router confidence to act (resolve prior_context) without escalating
# to the 120b context_manager.
ROUTER_CONF_THRESHOLD: float = _env_float("ROUTER_CONF_THRESHOLD", 0.6)
# Confidence assigned when the NIM returns a valid label as JSON but omits a
# numeric confidence field.
NIM_DEFAULT_CONF: float = _env_float("ROUTER_NIM_DEFAULT_CONF", 0.85)
# Confidence assigned when the NIM output is not valid JSON but a bare label
# token is recovered from the text.
NIM_FALLBACK_CONF: float = _env_float("ROUTER_NIM_FALLBACK_CONF", 0.7)
# How many trailing turns of the working buffer to use as the premise context.
PREMISE_TURNS: int = _env_int("ROUTER_PREMISE_TURNS", 2)
# Max characters of premise context fed to either backend.
PREMISE_MAX_CHARS: int = _env_int("ROUTER_PREMISE_MAX_CHARS", 800)
# How many leading notes to scan for the working-buffer recent_turns entry.
NOTES_SCAN_LIMIT: int = _env_int("ROUTER_NOTES_SCAN_LIMIT", 8)

# Maximum lettered options considered when detecting a clarification-question
# turn. NOT env-backed on purpose: this mirrors the frontend checkbox trigger
# (services/streamlit/app.py::extract_clarification_options) and the two must
# agree on what counts as a multi-option clarification, so it stays a fixed,
# in-sync named constant rather than an independently-tunable knob.
_MAX_OPTION_LETTERS: int = 5

_LABEL_RE = re.compile(r"prior_context|augment_prior|new_query")

# Mirrors services/streamlit/app.py::extract_clarification_options. Keep the
# regex and the consecutive-from-'a' rule below in sync with that function so
# backend intent routing and the UI checkbox launch agree on the same signal.
_OPTION_RE = re.compile(r"^\s*\(?([a-z])\)?[.:\)]\s*(.+)", re.IGNORECASE)


def _text_offers_lettered_options(text: str) -> bool:
    """True when the text presents a lettered option list (a, b, c, ...).

    Same definition as the frontend's extract_clarification_options: at least
    two options, starting at 'a' and consecutive. This is the structural signal
    that the assistant asked a multiple-option clarification question.
    """
    letters: list[str] = []
    for line in (text or "").split("\n"):
        match = _OPTION_RE.match(line.rstrip())
        if match and match.group(2).strip():
            letters.append(match.group(1).lower())
    if len(letters) < 2:
        return False
    return letters[0] == "a" and all(
        ord(letters[i]) == ord(letters[i - 1]) + 1
        for i in range(1, min(len(letters), _MAX_OPTION_LETTERS))
    )


def _recent_turn_text(notes: list[dict[str, Any]] | None) -> str:
    """Most recent conversation turn(s) from notes, used as the NLI premise /
    generative context. Empty string when no working-buffer turns are present."""
    for _n in (notes or [])[:NOTES_SCAN_LIMIT]:
        rt = _n.get("recent_turns")
        if rt:
            tail = [str(x) for x in list(rt)[-PREMISE_TURNS:]]
            return "\n".join(tail)[:PREMISE_MAX_CHARS]
    return ""


def _prior_turn_offered_options(notes: list[dict[str, Any]] | None) -> bool:
    """True when the most recent assistant turn presented a lettered clarification
    option list. Reuses the same detection as the frontend checkbox launch so the
    backend router and the UI agree on what a multi-option clarification is."""
    return _text_offers_lettered_options(_recent_turn_text(notes))


# Structural hint appended to the premise when the prior turn offered options.
# Improves BOTH backends' predictive power: it raises the prior_context
# entailment for MNLI and tells the generative NIM that a short reply is most
# likely selecting among the offered options.
_OPTIONS_HINT = (
    "\n[Context signal: the assistant's previous message asked the user to choose "
    "among lettered options (a, b, c ...). A short or selecting reply most likely "
    "refers to those options rather than starting a new topic.]"
)


def _premise(message: str, notes: list[dict[str, Any]] | None) -> str:
    tail = _recent_turn_text(notes)
    hint = _OPTIONS_HINT if _prior_turn_offered_options(notes) else ""
    base = f"{tail}\nUser: {message}" if tail else f"User: {message}"
    return base + hint


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
                    conf = NIM_DEFAULT_CONF
                return {
                    "intent": intent,
                    "confidence": min(1.0, max(0.0, conf)),
                    "source": "nim",
                }
        except (ValueError, TypeError):
            pass
    match = _LABEL_RE.search(text.lower())
    if match:
        return {"intent": match.group(0), "confidence": NIM_FALLBACK_CONF, "source": "nim"}
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
