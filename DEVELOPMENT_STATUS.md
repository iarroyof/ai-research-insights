# Sabia Development Status Register

> Linked from: [ARCHITECTURE.md](ARCHITECTURE.md)
> Last updated: 2026-06-12

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

  P-1  NLI agent routing gap
       nli.py _llm_nli() uses settings.llm.context_manager_provider directly.
       Add "nli" key to agent_models + pass agent="nli" to LLMClient if a
       dedicated model config is needed for NLI.

  P-2  Shape8 promotion decision
       Current: sentinel_a cleared, sentinel_c at 0.6724.
       Action needed: run blue-demon test suite, decide promote or block+diagnose.

  P-3  Reward signal for context_manager intent resolution
       When resolve_message_intent correctly resolves intent (verified by answer
       quality), context_manager should receive positive reward via
       ActionValue/VocabularyStore. Not yet implemented.

  P-4  _answer_mode uses raw message -- not verified against context-poor resolved query
       Currently intentional (answer mode is orthogonal to retrieval resolution).
       Revisit if users report wrong answer mode for resolved queries.

  P-5  Prompt caching trade-off
       Dynamic system prompts mean no cache hit per turn for 120b context_manager
       (reasoning=medium). Monitor latency in production.

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
