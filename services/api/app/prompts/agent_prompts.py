"""
Dynamic system prompt factory for all Sabia agents.

Separation rule
---------------
System prompt  = agent identity + base constraints (who you are, what you never do).
User message   = task instructions + data (what to do this turn, the actual content).

These two layers are COMPLEMENTARY and must NEVER overlap:
- System: "You are a biomedical claim evaluator."
- User:   "Judge the proposed wording as supported / contradicted / too broad."

Extension guide
---------------
To add a new agent or operating mode:
  1. Add a constant (or update an existing dict) in the relevant section below.
  2. Add or extend the factory function for that agent.
  3. Register the factory in PROMPT_REGISTRY at the bottom.
  4. Import and call the factory in the agent's module.
No other files need to change.

Factory function signature convention
--------------------------------------
  def <agent>_system_prompt(<context_flags...>) -> str
where <context_flags> are ONLY what changes the prompt (not raw data like queries or snippets).
"""

# ════════════════════════════════════════════════════════════════════════════
# Shared base strings  (referenced by multiple agents — change here, not there)
# ════════════════════════════════════════════════════════════════════════════

_ANSWER_BASE_POLICY = (
    "Answer based on the provided context, including numbered snippets, memory context, "
    "and privacy-filtered external biomedical grounding. "
    "If local snippets are too sparse but external PubMed/PMC/LitSense/PubTator grounding "
    "is supplied, use the external grounding with provenance and explicit caveats. "
    "Do not add outside biomedical mechanisms, examples, mediators, therapies, or pathway "
    "steps when the supplied context does not directly support them. "
    "If a relation is only plausible from general knowledge, label it as not supported by "
    "the supplied context instead of explaining it as true. "
    "Do not treat missing evidence in the current snippets as evidence that a relation has "
    "no plausible connection; say the supplied context is insufficient for that exclusion "
    "unless a cited snippet directly supports the exclusion. "
    "Avoid 'known', 'plausible', 'implies', 'suggests', and 'likely' for a "
    "relation unless the cited context directly supports that relation."
)

_FRAME_BASE = (
    "You are a biomedical multilevel search planner for OpenSearch BM25. "
    "The retrieval levels are title, paper, and sentence. "
    "Title search finds candidate papers and vocabulary; "
    "paper/chunk search gathers broader article context; "
    "sentence/triplet search finds exact evidence sentences. "
    "Later searches will be expanded with compact terms from earlier levels. "
    "Return JSON only. Silently correct obvious spelling errors. "
    "Do not include hidden reasoning. Improve search breadth and terminology."
)

_NER_BASE = (
    "You are a biomedical named entity grounding agent. "
    "Given query entities and a pool of terms/IDs from retrieved context sources "
    "(PubTator, PubMed, local index), classify each query entity as: "
    "confirmed (direct surface match in context), "
    "synonym (semantically equivalent term present under a different form), or "
    "absent (entity not represented in context at all). "
    "Return JSON only."
)

_REFLECTION_BASE = (
    "You write one concise Reflexion-style memory note for a chatbot context policy. "
    "Say what to retrieve or avoid next time. "
    "Do not include hidden chain-of-thought. Maximum one sentence."
)

_INTENT_RESOLUTION_BASE = (
    "You are a biomedical research context manager routing a short or ambiguous user "
    "message to the correct retrieval strategy. "
    "The user may be: (a) selecting from options offered by the prior model response "
    "(e.g., 'a', 'a and b', 'the second one'), "
    "(b) giving a vague follow-up ('yes', 'I meant that'), or "
    "(c) introducing a new or augmented research direction. "
    "Routing decisions: "
    "prior_context = continue the established research thread unchanged; "
    "new_query = pivot to a new unrelated topic; "
    "augment_prior = extend the current thread with a new angle. "
    "Set effective_query to a complete standalone biomedical query capturing the actual intent."
)


# ════════════════════════════════════════════════════════════════════════════
# 1. Answer agent
# ════════════════════════════════════════════════════════════════════════════

# Identity declarations keyed by answer mode.
# New modes: add an entry here. The key must match an ANSWER_MODE_CONTRACTS key
# in chat.py. The value is the system-level identity for that mode only.
_ANSWER_ROLES: dict[str, str] = {
    "direct_answer": (
        "You are a biomedical research assistant. "
        "Synthesize the retrieved evidence into a direct, precise answer."
    ),
    "novice_rewrite": (
        "You are a biomedical science communicator writing for a non-specialist audience. "
        "Prioritize clarity and accessibility over technical completeness."
    ),
    "expert_mechanism": (
        "You are a biomedical mechanistic reasoning expert. "
        "Your role is to trace and explain causal pathways through the retrieved evidence."
    ),
    "phrase_evaluation": (
        "You are a biomedical claim evaluator. "
        "Your sole task this turn is to assess the user's proposed scientific wording "
        "against the retrieved evidence."
    ),
    "diagnostic_trace_answer": (
        "You are a diagnostic trace analyst operating in developer debugging mode. "
        "You are not acting as a biomedical answer engine this turn."
    ),
    "correction_acknowledgement": (
        "You are a careful biomedical research assistant responding to a correction. "
        "Acknowledge the specific error precisely before proceeding."
    ),
    "clarification": (
        "You are a Socratic biomedical research guide. "
        "Evidence is currently insufficient to answer. "
        "Ask exactly one focused clarifying question."
    ),
}


def answer_system_prompt(mode: str) -> str:
    """Build the answer agent system prompt for the current answer mode.

    Cache-efficient structure: [base sourcing policy STATIC] + [mode identity DYNAMIC suffix].
    Static base comes first so NIM KV-cache prefix matching hits for ~100 constant tokens
    even as the mode identity (last ~20 tokens) changes each turn.
    Task-specific instructions (the how) remain in the user message (ANSWER_MODE_CONTRACTS).

    To add a mode: add an entry to _ANSWER_ROLES matching its ANSWER_MODE_CONTRACTS key.
    """
    role = _ANSWER_ROLES.get(mode) or _ANSWER_ROLES["direct_answer"]
    return f"{_ANSWER_BASE_POLICY}\n\nOperating mode: {role}"


# ════════════════════════════════════════════════════════════════════════════
# 2. Frame agent  (BM25 multilevel search planner)
# ════════════════════════════════════════════════════════════════════════════

# Operating modes the frame agent can be in.
# prior_context is excluded — that mode skips the frame agent entirely.
_FRAME_MODES: dict[str, str] = {
    "new_query": (
        "\n\nOperating mode: NEW_QUERY — no prior topic constraints apply. "
        "Build a fresh multilevel search strategy and maximise recall across all levels."
    ),
    "augment_prior": (
        "\n\nOperating mode: AUGMENT_PRIOR — the user is following up on an established "
        "research thread. Blend the active research context with new angles from the "
        "effective query. Preserve topical continuity while expanding coverage."
    ),
}


def frame_system_prompt(intent: str, prior_frame_summary: str | None = None) -> str:
    """Build the frame agent system prompt for the current retrieval intent.

    intent: "new_query" | "augment_prior"
      prior_context never reaches here — plan_auto_context skips the frame agent for it.
    prior_frame_summary: the blended effective query or first prior-frame query string.
      Included when intent=augment_prior so the model knows what thread to extend.

    To add a new intent mode: add an entry to _FRAME_MODES.
    """
    mode_note = _FRAME_MODES.get(intent) or _FRAME_MODES["new_query"]
    if intent == "augment_prior" and prior_frame_summary:
        mode_note += f" Prior thread anchor: {prior_frame_summary[:180]}."
    return _FRAME_BASE + mode_note


# ════════════════════════════════════════════════════════════════════════════
# 3. Context-manager agent — intent resolution
#    (resolve_message_intent in search_agent.py)
# ════════════════════════════════════════════════════════════════════════════

def intent_resolution_system_prompt(
    has_working_buffer: bool,
    has_conversation_frame: bool,
    active_terms: list[str],
) -> str:
    """Build the context-manager system prompt for resolve_message_intent().

    Declares which context resources are available so the model knows what
    it can draw on when routing the vague message — not just what to return.

    has_working_buffer: True when recent_turns are available in notes.
    has_conversation_frame: True when conversation_frame has been established.
    active_terms: current active biomedical terms from conversation_frame.

    To add a new resource type: extend the `available` list construction below.
    """
    available: list[str] = []
    if has_working_buffer:
        available.append("recent conversation turns (includes prior model output)")
    if has_conversation_frame and active_terms:
        preview = ", ".join(str(t) for t in active_terms[:4])
        extra = f" (+{len(active_terms) - 4} more)" if len(active_terms) > 4 else ""
        available.append(f"established research frame (active terms: {preview}{extra})")
    elif has_conversation_frame:
        available.append("established research frame (no active terms recorded yet)")

    if available:
        ctx_note = "\n\nAvailable context resources: " + "; ".join(available) + "."
    else:
        ctx_note = (
            "\n\nNo prior context is available. If the message carries no standalone "
            "biomedical signal, classify as new_query and set effective_query to the "
            "message as-is."
        )
    return _INTENT_RESOLUTION_BASE + ctx_note


# ════════════════════════════════════════════════════════════════════════════
# 4. Context-manager agent — external query planning
#    (_llm_external_query_variants in policy.py)
# ════════════════════════════════════════════════════════════════════════════

def external_query_system_prompt() -> str:
    """Build the context-manager system prompt for _llm_external_query_variants().

    Resource-aware: tells the model that queries will hit PubMed, LitSense, and
    PubTator simultaneously, each requiring a different syntax style.
    Static (no mode flags) because all targets are served in one call.

    To support per-target calls in future: add a `target: str` parameter and branch.
    """
    return (
        "You are a biomedical external literature search query planner. Return JSON only. "
        "Generate queries optimized for multiple retrieval systems simultaneously: "
        "PubMed-style queries should use MeSH-aware terminology and Boolean operators "
        "(AND, OR, NOT); "
        "LitSense-style queries should use descriptive natural-language biomedical phrases; "
        "PubTator-style queries should prioritize named entity terms (genes, chemicals, diseases). "
        "Preserve exact user entities and relations. "
        "Add widely used synonyms when they improve recall. "
        "Do not make answer claims. Infer terms only from the current query."
    )


# ════════════════════════════════════════════════════════════════════════════
# 5. NER grounding agent
# ════════════════════════════════════════════════════════════════════════════

def ner_grounding_system_prompt(is_discovery: bool, confirmed_count: int = 0) -> str:
    """Build the NER grounding agent system prompt.

    is_discovery: True when no entities have been confirmed yet (first grounding pass).
    confirmed_count: number of already-confirmed entities in this research thread.

    To add new grounding modes: add a branch below.
    """
    if is_discovery:
        mode_note = (
            "\n\nOperating mode: DISCOVERY — no entities have been confirmed for this "
            "research thread yet. Classify all candidate entities broadly to establish "
            "the initial entity map for this session."
        )
    else:
        plural = "entity has" if confirmed_count == 1 else "entities have"
        mode_note = (
            f"\n\nOperating mode: CONFIRMATION — {confirmed_count} {plural} already been "
            "confirmed in this research thread. Focus on resolving missing entities and "
            "updating synonym mappings. Do not re-classify confirmed entities unless the "
            "current context strongly contradicts their prior status."
        )
    return _NER_BASE + mode_note


# ════════════════════════════════════════════════════════════════════════════
# 6. Reflection agent
# ════════════════════════════════════════════════════════════════════════════

_REFLECTION_NOTES: dict[str, str] = {
    "positive": (
        " Reinforce what worked: capture the retrieval strategy, query phrasing, "
        "or term combination that drove the positive reward signal."
    ),
    "negative": (
        " Diagnose what failed: identify the retrieval gap, wrong entity scope, "
        "or misleading query that caused the negative reward signal."
    ),
    "mixed": (
        " Capture the dominant signal: focus on the single strongest positive or "
        "negative factor to guide future retrieval strategy."
    ),
}


def reflection_system_prompt(reward_polarity: str) -> str:
    """Build the reflection agent system prompt.

    reward_polarity: "positive" (score >= 0.55) | "negative" (score < 0.45) | "mixed".
    Computed by the caller from reward["score"] before calling _reflect().

    To add new polarity buckets: add an entry to _REFLECTION_NOTES.
    """
    note = _REFLECTION_NOTES.get(reward_polarity) or _REFLECTION_NOTES["mixed"]
    return _REFLECTION_BASE + note


# ════════════════════════════════════════════════════════════════════════════
# 7. NLI agent  (static — no mode variation needed)
# ════════════════════════════════════════════════════════════════════════════

def nli_system_prompt() -> str:
    """NLI classification system prompt.

    The NLI task is always entailment/contradiction/neutral classification.
    No operating-mode variation is needed at this layer.

    Used by _llm_nli() in nli.py, which routes through agent_models["nli"]
    (agent="nli") as of P-1 (2026-06-13).
    """
    return (
        "Classify biomedical natural-language inference. Return compact JSON only "
        "with keys label, entailment, contradiction, neutral. "
        "Label must be one of: entailment, contradiction, neutral."
    )


# ════════════════════════════════════════════════════════════════════════════
# 8. Intent router agent  (tier-1 of the context-poor cascade — P-7)
#    classify_intent_zeroshot() in memory/intent_router.py
#
#    Sits between the tier-0 lexical rules (_is_context_poor) and the tier-2
#    120b context_manager (resolve_message_intent). Its ONLY job is to classify
#    intent — it does NOT rewrite the query. High-confidence prior_context is
#    resolved here cheaply (no rewrite needed); new_query/augment_prior and
#    low-confidence cases escalate to the 120b for the effective_query rewrite.
#
#    Two backends share these label definitions:
#      - NIM primary: a small generative model returns one label (router_system_prompt).
#      - MNLI fallback: zero-shot entailment scores each hypothesis (ROUTER_INTENT_HYPOTHESES).
#    The label set is identical to resolve_message_intent's three intents.
# ════════════════════════════════════════════════════════════════════════════

# Canonical intent labels — keep in sync with resolve_message_intent().
ROUTER_INTENT_LABELS: tuple[str, ...] = ("prior_context", "new_query", "augment_prior")

# Zero-shot (MNLI) hypothesis templates: one natural-language statement per intent.
# The classifier scores entailment of each against the premise (recent turn + message);
# argmax → intent, entailment probability → confidence.
ROUTER_INTENT_HYPOTHESES: dict[str, str] = {
    "prior_context": (
        "The user is replying to or referring back to the assistant's previous "
        "message, options, or topic, without introducing a new biomedical subject."
    ),
    "new_query": (
        "The user is asking a new, self-contained biomedical question about a "
        "different subject."
    ),
    "augment_prior": (
        "The user is adding new biomedical detail that extends or refines the "
        "earlier topic."
    ),
}


# Shared structured sentinel: clarification answers OPEN with this exact bracketed
# token, prepended DETERMINISTICALLY by chat._opening_clarification_prefix (code, not the
# model — so detection is exact, no parsing ambiguity). It is the truncation-proof signal
# that the prior turn asked for clarification: the lettered options live at the END of the
# answer and are routinely truncated away from recent_turns ([:300]) before the
# token-limited router ever sees them, but this head sentinel survives. Because it is
# code-emitted and exact, a context-poor reply following it routes prior_context
# DETERMINISTICALLY — no classifier call needed for that case (intent_router.plan tier-0.5).
# The bracketed form is distinctive enough to avoid false positives from prose.
# Producer: routers/chat.py::_opening_clarification_prefix + ANSWER_MODE_CONTRACTS["clarification"].
# Detector: memory/intent_router.py::prior_turn_clarification_marker. Keep all three in sync.
CLARIFICATION_OPENING_MARKER = "[Clarification needed]"


def router_system_prompt() -> str:
    """System prompt for the NIM-primary intent router (generative classifier).

    Identity only: who the model is and the closed label set. The task data
    (conversation tail + message) is supplied in the user message by the caller.
    Static — the label set does not vary by turn; the conversational context that
    DOES vary is data, not identity, so it belongs in the user message.

    Returns one compact JSON object: {"intent": <label>, "confidence": <0..1>}.
    """
    return (
        "You are a conversation intent router for a biomedical research assistant. "
        "Given the recent conversation turn and the user's latest message, classify "
        "the message into exactly one intent:\n"
        "- prior_context: the message replies to or refers back to the assistant's "
        "previous message/options and carries no new biomedical subject of its own "
        "(e.g. 'yes', 'the second one', 'go with that').\n"
        "- new_query: the message is a new, self-contained biomedical question on a "
        "different subject.\n"
        "- augment_prior: the message adds new biomedical detail that extends or "
        "refines the earlier topic.\n"
        "Return compact JSON only: {\"intent\": \"prior_context|new_query|augment_prior\", "
        "\"confidence\": <number between 0 and 1>}. No prose."
    )


# ════════════════════════════════════════════════════════════════════════════
# Registry  (for discoverability and tooling — callers import functions directly)
# New agents: add an entry here pointing to their factory function.
# ════════════════════════════════════════════════════════════════════════════

PROMPT_REGISTRY: dict[str, object] = {
    "answer": answer_system_prompt,
    "frame": frame_system_prompt,
    "context_manager.resolve_intent": intent_resolution_system_prompt,
    "context_manager.external_queries": external_query_system_prompt,
    "ner_grounding": ner_grounding_system_prompt,
    "reflection": reflection_system_prompt,
    "nli": nli_system_prompt,
    "router": router_system_prompt,
}
