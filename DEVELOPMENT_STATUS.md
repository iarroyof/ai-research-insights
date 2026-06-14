# Sabia Development Status Register

> Linked from: [ARCHITECTURE.md](ARCHITECTURE.md)
> Last updated: 2026-06-13

---

## Current Stage

  Shape8 eval: sentinel_a CLEARED (sentinel_c cleared at 0.6724)
  Pending:     Decide whether to promote shape8 or block + diagnose.

---

## WP Status

  WP-A  PubTator entity extraction   DONE   web_search.py
  WP-B  GapSpec                      DONE   search_agent.py, policy.py
  WP-C  Snippet utility              DONE   search_agent.py
  WP-D  Per-step rewards             DONE   rewards.py, policy.py
  WP-E  Eval shape8                  DONE   reward_scorer.py, scenarios/
  WP-F  Onomasiological Bandit       DONE   policy.py, store.py, idea_index.py,
         Memory                             vocabulary_store.py

---

## Recent Changes

  2026-06-12
    - Context-poor message handling: _CONTEXT_POOR_EXACT set added;
      limit=1->2 in _is_context_poor; resolve_message_intent reads working buffer.
    - Working buffer injected into build_auto_context for context_poor messages.
    - conversation_frame fetched first in policy.plan().
    - Multi-turn coreference eval scenarios: multiturn_coreference_001,
      multiturn_context_drift_001 (YAML + JSONL seeds).
    - Coreference reward components: inter_turn_coreference (0.10),
      context_poor_resolution (0.05), conversation_continuity (0.05).
    - Full prompt audit: all 6 agents documented; NLI routing gap identified.
    - ARCHITECTURE.md written (full canonical reference).

---

## Pending Items

  P-1  NLI agent routing gap  — ✅ DONE (2026-06-13)
       _llm_nli() now passes agent="nli"; nli entry added to agent_models
       (super-49b, max_tokens=1024 — empty under tighter caps); system prompt
       from nli_system_prompt(); premise/hypothesis truncation env-backed
       (NLI_LLM_PREMISE_MAX_CHARS/NLI_LLM_HYPOTHESIS_MAX_CHARS). HF MNLI
       (memory.nli_model) remains the primary factuality authority; the LLM path
       is fallback only. Validated real: entailment 1.0, contradiction 1.0;
       4 routing unit tests + claim_support regression green.

  P-2  Shape8 promotion decision  — sentinel_a RE-CLEARED 2026-06-13 (recommend PROMOTE)
       Ran sentinel_a live (in-container, http adapter) on the P-7+P-1+P-3 config.
       Result run shape8_sentinel_a_p7p1p3: 8-scenario avg 0.7247 (> 0.7148 baseline),
       missed_injected_traps=0 (the gate) → CLEARS. cross_cancer/mdsc_treg (which
       regressed in the reverted ultra-550b experiment) now healthy (0.8175/0.7212).
       recommendations.json has forward-looking reward-shaping items (mechanism-graph
       required-node awareness; unsupported-mechanism penalization) — NOT gate blockers.
       Remaining for FULL promotion: (a) bump current_stage in reward_shape_registry.yaml,
       (b) optionally re-run sentinel_c (was 0.6724) with the new config for full rigor.
       Both are the user's milestone call.

  P-3  Reward signal for intent resolution  — ✅ DONE (2026-06-13)
       plan_auto_context records intent_resolution metadata {tier, intent, source,
       confidence, state_key, action_key, effective_query} on AutoContextPlan; it
       flows via plan.to_dict() -> auto_context payload -> observe_turn, which
       credits the (state, action) = (intentres|len=BUCKET, TIER:INTENT) pair with
       the turn reward in the existing ActionValue table (no new Q-layer). Reward
       SIGNAL is recorded; the learning loop that reads it back to bias tier choice
       is the documented next step. Tiers: tier1_router | tier2_120b | heuristic.

  P-4  answer_mode vs context-poor resolved query  — ✅ DONE (default off, 2026-06-13)
       _answer_mode now accepts resolved_query; ONLY the question-type modes
       (novice_rewrite, expert_mechanism) consider it — utterance modes (correction,
       clarification, phrase_evaluation, diagnostic) stay on the raw message.
       Gated by memory.answer_mode_consider_resolved_query (default FALSE → no
       behaviour change). Enable via env ANSWER_MODE_CONSIDER_RESOLVED_QUERY=true
       after an eval gate confirms no shape8 regression.

  P-5  Prompt caching — static-prefix ordering  — ✅ DONE (2026-06-13)
       Verified all 7 prompt factories are static-base-first (dynamic suffix last),
       so KV-cache prefix matching holds. Added test_prompt_cache_ordering.py to
       LOCK the invariant against future reordering. Latency still monitorable via
       GET /chat/memory/provider-metrics. (Ordering was already correct on main;
       this pins it.)

  P-6  Cascaded routing for context_manager cost (proposal)
       resolve_message_intent() input is ~700-1100 tokens — NOT a context-window
       problem. Bottleneck: 120b model cost + reasoning=medium tokens.
       Proposal: tier-0 (heuristic, free) → tier-1 (49b fast) → tier-2 (120b,
       reasoning=medium, only genuine multi-turn ambiguity). See session 2 notes.

---

## Shape Ladder Reference

  Stages (in order):
    correction_scope -> diagnostic_gate -> family_microfit ->
    sentinel_a       -> sentinel_c      -> PROMOTE / BLOCK

  Eval files:
    evals/lung_factuality_lab/configs/reward_shape_registry.yaml
    evals/lung_factuality_lab/src/reward_scorer.py

---

## 2026-06-12 (session 2)

### Changes: Dynamic system prompt factory (agent_prompts.py)

New file:  services/api/app/prompts/agent_prompts.py
           services/api/app/prompts/__init__.py

Patched:
  services/api/app/routers/chat.py         — answer_system_prompt(answer_mode)
  services/api/app/memory/search_agent.py  — frame_system_prompt(intent, prior_frame_summary)
                                             intent_resolution_system_prompt(has_wbuf, has_frame, active_terms)
                                             ner_grounding_system_prompt(is_discovery, confirmed_count)
  services/api/app/memory/policy.py        — external_query_system_prompt()
                                             reflection_system_prompt(reward_polarity)

Separation rule enforced:
  System prompt  = agent identity + base constraints
  User message   = task instructions + data (ANSWER_MODE_CONTRACTS unchanged)

Extension guide:
  Add entry to _ANSWER_ROLES / _FRAME_MODES / _REFLECTION_NOTES or a new factory fn.
  Register in PROMPT_REGISTRY. No other files change.

### Pending
  P-1  NLI agent routing gap (nli.py uses context_manager_provider directly)
  P-2  Shape8 promotion decision
  P-3  Reward signal for context_manager intent resolution
  P-5  Prompt caching trade-off: dynamic system prompts mean no cache hit per turn
       for 120b context_manager (reasoning=medium). Monitor latency if needed.
  P-6  Cascaded routing for context_manager cost (PROPOSAL, not yet implemented)
       Context: resolve_message_intent() input is ~700-1100 tokens — well below any
       context window limit. Cost bottleneck is model size (120b) + reasoning=medium
       tokens (~1000-2000 reasoning tokens per call, billed as output).
       Proposal: 3-tier routing
         tier-0 (no LLM): if active_terms=[] AND recent_turns=[] AND message short
                          → return prior_context heuristically (free)
         tier-1 (49b): if message contains recognisable biomedical term(s) AND
                        no strong ambiguity signal from working buffer → new_query fast
         tier-2 (120b, reasoning=medium): genuinely ambiguous multi-turn reference
       Expected savings: >60% of resolve_message_intent() calls never reach 120b.
       See P-3 for reward tracking once correct intent resolution is confirmed.

## P-7 IMPLEMENTED (2026-06-13): Tier-1 zero-shot intent router

Inserted between tier-0 lexical rules (_is_context_poor) and tier-2 120b (resolve_message_intent) in plan_auto_context.

- Backends: NIM primary (nvidia/nemotron-3-nano-30b-a3b via agent_models.router) + HF MNLI fallback (facebook/bart-large-mnli via app.services.zero_shot.score_labels, wrapped in asyncio.to_thread). Both validated against real providers.
- Behavior: high-confidence prior_context (conf >= ROUTER_CONF_THRESHOLD=0.6) short-circuits the 120b (no query rewrite needed); new_query/augment_prior and low-confidence escalate to the 120b for the effective_query rewrite.
- Files: config/default.yaml (router agent_models, max_tokens=256 — Nemotron needs reasoning headroom), prompts/agent_prompts.py (router_system_prompt + ROUTER_INTENT_HYPOTHESES + registry), memory/intent_router.py (NEW), memory/search_agent.py (cascade integration + import), tests/test_intent_router.py (NEW, 12 tests).
- Validation: 12/12 unit, 36/36 search_agent regression, real NVIDIA nano + HF MNLI smoke (the second one->prior_context 0.96 nim; EGFR question->new_query 0.96 nim).
- Note: replaces the naive lexical _is_followup_reference gate for context-poor routing; _is_followup_reference still used elsewhere (P-1/anchor robustness remains future work).

### P-7 follow-up (same day): answer-derived signal + no-hardcoding pass

- Answer-derived intent signal: intent_router._text_offers_lettered_options detects
  when the prior assistant turn asked a lettered (a/b/c) clarification question;
  _premise() appends a hint that biases the router toward prior_context. This
  MIRRORS the frontend checkbox trigger (streamlit extract_clarification_options) —
  shared contract documented in BOTH files; do not let them diverge. Does NOT
  conflict with the UI checkbox launch (frontend renders; backend only classifies).
  Real smoke: prior turn with a/b/c + reply "b" -> prior_context 0.96.
- No-hardcoding refactor: all router literals are now env-backed named constants:
  ROUTER_CONF_THRESHOLD (0.6), ROUTER_NIM_DEFAULT_CONF (0.85),
  ROUTER_NIM_FALLBACK_CONF (0.7), ROUTER_PREMISE_TURNS (2),
  ROUTER_PREMISE_MAX_CHARS (800), ROUTER_NOTES_SCAN_LIMIT (8).
  _MAX_OPTION_LETTERS is a fixed named constant (frontend-locked, not env).
- Tests: 17/17 (added option-detection + premise-hint + constant-referenced asserts).

---

## P-8: Tail-content head-truncation audit (2026-06-14)

Failure mode: important content placed at the TAIL of a message is silently lost to a
HEAD truncation (or a count cap) before a downstream consumer sees it.

- ✅ FIXED (primary): clarification OPTIONS live at the answer tail, but recent_turns are
  head-truncated to [:300] in build_auto_context → the tier-1 intent router (token-limited)
  AND the 120b resolve_message_intent (same recent_turns) silently missed them, diverging
  from the frontend (which parses the full answer for checkboxes). Fix: deterministic head
  marker CLARIFICATION_OPENING_MARKER (survives [:300]) + clarification contract now opens
  with the marker and puts lettered options last. 19/19 router tests.
  REFINEMENT (2026-06-14): marker strengthened to a bracketed sentinel "[Clarification
  needed]" (distinctive, exact, code-emitted) and PROMOTED to a deterministic tier-0.5 in
  plan_auto_context: context-poor + marker -> prior_context with NO classifier/120b
  (prior_turn_clarification_marker). The classifier is NOT removed — it's reserved for
  no-marker context-poor replies (coreference/"continue"/"yes" after a normal answer), the
  majority, since clarification answer-mode is a minority of turns. _prior_turn_is_clarification
  (marker OR lettered) stays as the softer tier-1 premise hint.
- Secondary, LOW risk, EASY if needed (not done — flagged):
  * ner_grounding ctx_entities[:40]→[:30] (search_agent ~1726/1745): silent COUNT cap; a key
    entity ranked >40 is dropped. Easy: sort query/confirmed entities to the front before the cap.
  * NLI _llm_nli premise[:1200]/hypothesis[:500] (nli.py): head-trunc of a long source
    window could cut the relevant clause. Already env-configurable (NLI_LLM_*_MAX_CHARS);
    easy to raise or take a centered window. Low risk (origin sentences are short).
  * Answer-agent render caps (policy.py text[:900]/[:700]/[:500] etc.): per-item head-trunc for
    the answer prompt. Lower impact (49b large context, synthesizes). Could append "…(truncated)"
    so the model knows. Not a silent-routing bug.

## Engineering Standards (apply to ALL agents/sessions)

- NO HARDCODING (ARCHITECTURE.md rule 13): every tunable literal is a named,
  configurable constant (env-backed via os.getenv like zero_shot.py/nli.py, or a
  config/default.yaml field). No magic numbers inline. Frontend-locked values stay
  fixed named constants with a sync comment. Record new knobs here when added.
- Two-level working memory to revisit EVERY session before changing code:
  (1) ARCHITECTURE.md (canonical system reference) + DEVELOPMENT_STATUS.md (this
  register) on blue-demon; (2) the dev agent's own memory (MEMORY.md index +
  architecture_sabia.md / project_gapspec_wps.md). Grep before adding (rule 1).

## Session-Termination Handoff (2026-06-13)

- Source of truth = blue-demon working tree; access via plink/pscp (PuTTY), creds
  ../blue-demon.txt; GitHub token ../../github_toke.txt. No gh / no credential.helper
  on blue-demon. API container ai-research-insights-api-1 bind-mounts services/api->/app.
- Local doc-artifacts mirrors (source_snapshot/, remote_edit/) are STALE — never
  trust them for code; read blue-demon.
- Before ending any session: update ARCHITECTURE.md + DEVELOPMENT_STATUS.md for any
  change, mirror into the dev agent memory, and re-state these standards so the next
  agent syncs to the docs and obeys rule 13.
