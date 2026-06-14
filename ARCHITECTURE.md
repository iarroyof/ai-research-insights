# Sabia — Full Agentic Architecture & Engineering Reference

> **Development-agent canonical reference.** Read this before making any change.
> Last updated: 2026-06-13.
> For WP/milestone status see: [DEVELOPMENT_STATUS.md](DEVELOPMENT_STATUS.md)

---

## 1. System Overview

Sabia is a multi-agent biomedical research assistant:
- FastAPI backend       services/api/
- Streamlit frontend   services/streamlit/app.py
- Elasticsearch        episodic memory, ideas, triplets, policy notes, action values
- Redis                vocabulary bandit (VocabularyStore)
- NVIDIA NIM endpoints multi-agent LLM routing via LLMClient
- Dynamic prompt factory: services/api/app/prompts/agent_prompts.py

Entry point: services/api/app/routers/chat.py  /chat/  SSE endpoint.

---

## 2. Turn Execution Order (chat.py)

  1. build_auto_context()         [search_agent.py]   BM25/OpenSearch retrieval
  2. build_prompt_and_citations() [rag/context.py]    grounded task prompt + citations
  3. policy.plan()                [policy.py]         memory retrieval + context_prefix
  4. LLM stream (answer agent)                        generate answer
  5. policy.observe_turn()        [policy.py]         memory update + rewards

build_auto_context() and policy.plan() are INDEPENDENT -- not nested.
Do NOT assume one has access to the other's within-turn data.

Full prompt assembly order for the answer agent user message:
  [context_prefix from policy.plan()]
      "\n\nGrounded task prompt:\n"
  [answer_mode_context from _answer_mode_prompt()]
      "\n\nGrounded task prompt:\n"
  [system_msg + context_block + "User: {msg}" from build_prompt_and_citations()]

---

## 3. Agent Roles, Models, and Prompts

All LLM calls go through LLMClient().chat_once/chat_stream(agent=X).
Per-agent routing: _agent_provider_config(agent_name) reads
  config/default.yaml -> llm.agent_models.<name>
  -> {provider, model, max_tokens, reasoning_effort, enable_thinking}

### 3a. Answer Agent  (agent="answer")
  Model:   nvidia/llama-3.3-nemotron-super-49b-v1.5  (no reasoning)
  Called:  chat.py, chat_stream(messages, agent="answer")
  System prompt [STATIC]:
    "You are a helpful research assistant. Answer based on the provided context..."
    (hardcoded string in chat.py line ~864)
  User message [FULLY DYNAMIC, assembled per turn]:
    1. context_prefix from policy.plan():
         policy_instruction + conversation_frame render + landmarks + summaries +
         working buffer render (_render_recent) + vocabulary guide (_render_ideas) +
         episodic memory hits + triplets + web grounding + reflections
    2. answer_mode_contract (7 modes -- see section 4)
         includes puzzle state: edge_support_status, covered_nodes, missing_nodes
    3. grounded task prompt with ranked BM25 snippets from build_prompt_and_citations()
  Native history [DYNAMIC]:
    _native_history_messages() injects last K=working_buffer_turns*2 turns as native
    {role,content} messages into the messages list BEFORE the user content.
    Gives model turn-level continuity without re-sending full history in user text.
  Key invariant: system role is DYNAMIC since 2026-06-12 via answer_system_prompt(answer_mode).
    Structure: [mode identity] + [base sourcing policy]. Mode identity changes per turn.
    User message and native_history remain fully dynamic as before.

### 3b. Frame Agent  (agent="frame")
  Model:   nvidia/llama-3.3-nemotron-super-49b-v1.5  (no reasoning)
  Called:  search_agent.py llm_refine_variants()
  System prompt [DYNAMIC since 2026-06-12]:
    frame_system_prompt(intent, prior_frame_summary) -- new_query | augment_prior
  User message [DYNAMIC per turn]:
    - _eff_msg (resolved effective message, NOT raw for context-poor inputs)
    - base_variants (deterministic query variants)
    - search_frame (domain, preferred_queries, avoid_terms)
    - prior search notes from policy_notes (last 4 cached plans)
    - action_value hints (Q-table best actions for this state)
  Skipped: when resolve_message_intent returns prior_context
    (_skip_frame_refine=True -- reuses prior BM25 frame)

### 3c. Context_manager Agent  (agent="context_manager") -- TWO USES
  Model:   nvidia/nemotron-3-super-120b-a12b  (reasoning=medium)

  USE 1: resolve_message_intent()  [search_agent.py line ~889]
    Triggered: _is_context_poor(message)=True AND allow_llm_refine=True
    System prompt [STATIC]:
      "You are a biomedical research context manager. Determine the actual
       search intent of a short or ambiguous user message..."
    User message [DYNAMIC per turn]:
      - vague message text
      - conversation_frame.summary + active_terms (from frame_note in notes)
      - prior search queries (from policy_notes, recent first)
      - recent_turns (working buffer: last 3 turns injected by build_auto_context())
    Returns JSON:
      {intent: prior_context|new_query|augment_prior, effective_query, explanation}
    Effect on retrieval:
      prior_context  -> reuse prior BM25 frame, skip frame LLM
      new_query      -> _eff_msg = effective_query (new topic)
      augment_prior  -> _eff_msg = effective_query (blended query)

  USE 2: _llm_external_query_variants()  [policy.py line ~344]
    Triggered: local retrieval is sparse, need PubMed/LitSense/PubTator queries
    System prompt [STATIC]:
      "You are a biomedical literature search query planner. Return JSON only..."
    User message [DYNAMIC per turn]:
      - query (message or _eff_msg)
      - deterministic base variants
    Returns JSON: {queries: [...], note: "..."}

### 3d. Reflection Agent  (agent="reflection")
  Model:   nvidia/llama-3.3-nemotron-super-49b-v1.5  (no reasoning)
  Called:  policy.py _reflect()  after observe_turn()
  System prompt [STATIC]:
    "You write one concise Reflexion-style memory note for a chatbot context policy..."
  User message [DYNAMIC per turn]:
    - question (truncated 1000 chars)
    - answer (truncated 1000 chars)
    - reward dict
    - conflict count
  Output: one short policy note written to ES episodic memory

### 3e. NER Grounding Agent  (agent="ner_grounding")
  Model:   default (inherits nvidia_model = nemotron-super-49b)
  Called:  search_agent.py llm_ground_entities()  inside policy.plan()
  System prompt [STATIC]:
    "You are a biomedical named entity grounding agent. Given query entities
     and a pool of terms/IDs from retrieved context sources..."
  User message [DYNAMIC per turn]:
    - message (truncated)
    - query_ents (entities extracted from user query)
    - ctx_entities (up to 40 terms from all retrieved sources: PubTator + local index)
  Effect: updates GapSpec.confirmed_entities, .missing_entities, .entity_map
  Toggle: settings.memory.entity_grounding_enabled

### 3f. NLI Agent  (agent="nli" — routed via agent_models since P-1, 2026-06-13)
  Model:   nvidia/llama-3.3-nemotron-super-49b-v1.5 (agent_models.nli, max_tokens=1024
           — super-49b returns EMPTY under a tight cap; 1024 gives JSON headroom)
  Called:  nli.py _llm_nli()  (generative LLM fallback, used only when the HF MNLI
           path is unavailable; HF MNLI = memory.nli_model PubMedBERT-MNLI-MedNLI
           remains the primary factuality authority)
  System prompt [STATIC]: nli_system_prompt() from agent_prompts.py
  User message [DYNAMIC per turn]:
    - premise sentence (NLI_LLM_PREMISE_MAX_CHARS, default 1200)
    - hypothesis/claim (NLI_LLM_HYPOTHESIS_MAX_CHARS, default 500)
  P-1 FIXED: was settings.llm.context_manager_provider + hardcoded max_tokens=120;
    now passes agent="nli" so model/provider/max_tokens come from agent_models.
    Validated real: entailment fixture -> entailment 1.0, contradiction -> 1.0.

### 3g. Intent Router Agent  (agent="router")  -- tier-1 of context-poor cascade (P-7)
  Model:   nvidia/nemotron-3-nano-30b-a3b (agent_models.router, max_tokens=256 —
           Nemotron spends budget on reasoning first; <~200 yields EMPTY content)
  Fallback: HF zero-shot MNLI (facebook/bart-large-mnli) via
           app.services.zero_shot.score_labels (SYNC — wrapped in asyncio.to_thread).
  Called:  search_agent.py plan_auto_context() -> intent_router.classify_intent_zeroshot()
           ONLY when _is_context_poor(message)=True AND allow_llm_refine=True,
           BEFORE the 120b resolve_message_intent.
  System prompt [STATIC]: router_system_prompt() in agent_prompts.py.
  Returns: {intent: prior_context|new_query|augment_prior, confidence, source}.
           CLASSIFIES ONLY — does not rewrite the query.
  Cascade effect:
    prior_context AND confidence>=ROUTER_CONF_THRESHOLD (0.6)
        -> resolve as prior_context, SKIP the 120b (no rewrite needed)
    new_query/augment_prior OR low confidence OR both backends fail
        -> escalate to resolve_message_intent (120b) for the effective_query rewrite
  Answer-derived signal: _premise() appends an options hint when the prior
    assistant turn asked for clarification (_prior_turn_is_clarification). Two
    signals OR'd: (1) CLARIFICATION_OPENING_MARKER ("Clarification needed —") at
    the answer HEAD — prepended deterministically by chat._opening_clarification_prefix,
    TRUNCATION-PROOF; (2) a lettered (a/b/c) option list, mirroring the frontend
    checkbox trigger (extract_clarification_options).
    ⚠️ TRUNCATION FAILURE MODE (fixed 2026-06-14): clarification options live at the
    END of the answer, but recent_turns are head-truncated to [:300] in
    build_auto_context, so the token-limited router (and the 120b resolve_message_intent,
    same recent_turns) silently lost them. The head marker survives [:300] and is the
    robust signal; the lettered check stays aligned with the UI. The clarification
    ANSWER_MODE_CONTRACT now opens with the marker + puts options last.
    Shared contract across THREE sites — keep in sync: agent_prompts.CLARIFICATION_OPENING_MARKER,
    chat._opening_clarification_prefix/ANSWER_MODE_CONTRACTS["clarification"] (producer),
    intent_router._prior_turn_is_clarification (detector).
  All knobs env-backed: ROUTER_CONF_THRESHOLD, ROUTER_NIM_DEFAULT_CONF,
    ROUTER_NIM_FALLBACK_CONF, ROUTER_PREMISE_TURNS, ROUTER_PREMISE_MAX_CHARS,
    ROUTER_NOTES_SCAN_LIMIT (rule 13).

---

## 4. Answer Mode System (chat.py)

_answer_mode() selects mode from raw body.message + evidence_assembly state.
IMPORTANT: mode selection uses LITERAL user message, not resolved _eff_msg.
This is intentional -- answer mode is orthogonal to retrieval resolution.

  Trigger logic (priority order):
    1. correction_only_turn=True              -> correction_acknowledgement
    2. _hold_generation_for_clarification()  -> clarification
         (clarification_recommended=True AND edge_support="missing"
          AND missing_nodes present AND retrieved_count=0)
    3. "can i phrase"/"is this phrase"/"this statement" in message
                                             -> phrase_evaluation
    4. "reward model"/"diagnostic"/"debug"/"trace evidence"
                                             -> diagnostic_trace_answer
    5. "novice"/"rewrite"/"summarize"/"one paragraph"
                                             -> novice_rewrite
    6. "mechanism"/"pathway"/"explain how"/"why does"
                                             -> expert_mechanism
    7. (default)                             -> direct_answer

  ANSWER_MODE_CONTRACTS (what each mode injects into user prompt):
    direct_answer            answer from evidence; separate supported from unsupported
    novice_rewrite           compress supported puzzle edges; preserve caveats
    expert_mechanism         explain mechanism edges with evidence support labels
    phrase_evaluation        judge proposed wording (supported/contradicted/too broad)
    diagnostic_trace_answer  discuss trace/evaluator evidence only; no biomedical inference
    correction_acknowledgement  acknowledge correction; update scope
    clarification            summarize puzzle state; ask one focused clarification

  Special behaviors:
    - external_grounding_covers_puzzle AND mode=clarification
        -> override to direct_answer (user gets answer, not a question)
    - opening_clarification_prefix prepended before generation when hold=True
    - post_generation_guard runs for novice_rewrite and clarification when evidence missing

---

## 5. Context-Poor Message Handling (search_agent.py)

Problem: short/vague replies ("a and b", "yes", "the second one") have no
biomedical anchors -> BM25 gets nonsense queries -> retrieval broken.

### 5a. Detection: _is_context_poor(message)

  Step 1: exact match against _CONTEXT_POOR_EXACT frozenset (after strip+lower):
    Affirmations: yes, no, yeah, sure, ok, okay, alright, agreed, correct, ...
    Ordinals:     the first one, the second one, the third one, second one, ...
    Discourse:    I meant that, that option, all of them, both of them, ...
    (Needed because _query_anchor_terms("yes") = ["yes"] -- "yes" passes anchor
     filtering without this exact set.)

  Step 2: _is_clarification_reply() -- matches pure letter replies: "a", "b and c"

  Step 3: not _query_anchor_terms(message, limit=2) AND len(message.split()) <= 8
    (limit=2 -> important_terms gets 4 candidates; ensures proper nouns ranked 3rd
     like "Aspergillus" in "how does Aspergillus..." are detected and NOT flagged)

### 5b. Resolution Pipeline

  build_auto_context():
    a. Detect context_poor flag
    b. If context_poor OR followup: fetch conversation_frame -> inject as frame_note
    c. If context_poor: fetch store.recent_messages(session_id, 3, token_budget=4000)
         -> format as [Turn N] role: content[:300]
         -> inject as recent_turns_note into notes list

  plan_auto_context() cascade (P-7):
    tier-1: classify_intent_zeroshot() [router agent, nano NIM + MNLI fallback]
            high-confidence prior_context -> resolve here, skip the 120b
    tier-2: resolve_message_intent() [context_manager, 120b, med] only when
            tier-1 says new_query/augment_prior or is unsure
    Reads from notes: active_terms, summary, recent_queries, recent_turns
    Returns: {intent, effective_query} -> sets _eff_msg

  policy.plan():
    When context_poor: _mem_query = active_terms from conversation_frame
    ALL ES searches (episodic, ideas, triplets) use _mem_query not raw message

  Answer agent:
    native_history shows prior model output (e.g. "a) immunosuppression... b) mycotoxin...")
    _answer_mode("a and b") -> direct_answer (correct -- it IS a direct answer)
    LLM sees correct retrieved content AND prior turn context -> resolves correctly

---

## 6. Memory Levels (complete inventory -- do not add duplicates)

  Level  Name                 Key code path
  -----  -------------------  -----------------------------------------
  6.1    Working Buffer       store.recent_messages(session_id, K=8, 48000)
                              plan() -> _render_recent() -> context_prefix.
                              ALSO native_history in answer agent message list.
                              ALSO fetched in build_auto_context() (3 turns, 4000 tokens)
                              for context_poor -> resolve_message_intent().
                              Three read paths. Not a duplicate -- each path has a
                              distinct role (render / native role msgs / intent context).

  6.2    Episodic Memory      store.search_memory(session_id, _mem_query, k=16)
                              Semantic ES vector search.

  6.3    Landmarks            store.landmarks(session_id) -- key pinned facts

  6.4    Episodic Summaries   store.episodic_summaries(session_id, 3)

  6.5    Conversation Frame   store.conversation_frame(session_id)
                              {active_terms, avoided_terms, supported_claims,
                               contradicted_claims, summary}
                              FETCHED FIRST in plan() (2026-06-12 fix).
                              Do NOT add a second frame/state tracker.

  6.6    Policy Notes         store.search_policy_notes(session_id, 4)
                              Cached BM25 search plans. Used by _prior_frame_variants().

  6.7    Idea/Concept Index   store.search_ideas(session_id, _mem_query, k=8, user_id=X)
         (IdeaRecord, ES)     {idea, synonyms, parent_idea, concept_path, reward_avg,
                               frequency, scope: session_id|user_{id}|shared}
                              Rendered by _render_ideas() -> vocabulary guide in context_prefix.
                              ALREADY EXISTS (WP-F-2). Do NOT add a new concept store.

  6.8    Vocabulary Store     Redis Thompson-sampling Beta(a,b) bandit per term per scope.
         (VocabularyStore)    record_outcome(), session_top_terms(), promote_to_global_candidate()
                              ALREADY EXISTS. Do NOT add a second bandit.

  6.9    Action Values        ES Q(state,action) table.
                              best_action_value() in action_value.py.
                              Updated in observe_turn(). ALREADY EXISTS.

  6.10   Triplets (KG)        search_triplets(tenant, _mem_query, confidence_min) in plan()

  6.11   Reward Traces        store.latest_traces(session_id, 3) in plan().
                              Written by observe_turn() via reward_report().

---

## 7. Module Interaction Map

  chat.py (entry point)
    |
    +-- build_auto_context()                         [search_agent.py]
    |     +-- search_policy_notes()                  [ES: policy notes]
    |     +-- conversation_frame() -> frame_note     [if context_poor or followup]
    |     +-- recent_messages(3, 4000) -> recent_turns_note  [if context_poor]
    |     +-- plan_auto_context(message, notes)
    |     |     +-- _is_context_poor? -> resolve_message_intent()  [context_manager]
    |     |     |     reads: active_terms, summary, recent_queries, recent_turns from notes
    |     |     |     returns: intent + effective_query -> _eff_msg
    |     |     +-- deterministic_query_variants(_eff_msg)
    |     |     +-- _prior_frame_variants(notes)       [if prior_context or fallback]
    |     |     +-- llm_refine_variants(_eff_msg)      [frame agent, if NOT prior_context]
    |     +-- BM25 multilevel search (title -> paper -> sentence)
    |           -> snippets + GapSpec + step_rewards
    |
    +-- build_prompt_and_citations()                 [rag/context.py]
    |     -> grounded task prompt with ranked snippets
    |
    +-- policy.plan()                                [policy.py]
    |     +-- conversation_frame()       <- FETCHED FIRST (2026-06-12)
    |     +-- _mem_query = active_terms if context_poor else message
    |     +-- search_memory(_mem_query)             [ES: episodic]
    |     +-- landmarks()                           [ES: landmarks]
    |     +-- episodic_summaries()                  [ES: summaries]
    |     +-- latest_traces()                       [ES: reward traces]
    |     +-- search_ideas(_mem_query, user_id)     [ES: IdeaRecord]
    |     +-- action_values()                       [ES: Q-table]
    |     +-- search_triplets(_mem_query)           [ES: KG]
    |     +-- recent_messages(K=8, 48000)           [working buffer]
    |     +-- llm_ground_entities()                 [ner_grounding agent]
    |     +-- external web search if sparse         [context_manager + PubMed/LitSense]
    |     +-- _render_*() -> context_prefix
    |           _policy_instruction, render_conversation_frame, _render_landmarks,
    |           _render_summaries, _render_recent, _render_ideas, _render_memory,
    |           _render_triplets, _render_web, policy reflections
    |
    +-- _answer_mode() + _answer_mode_prompt()      -> answer_mode_contract
    +-- _native_history_messages()                  -> native chat turns
    |
    +-- LLM stream [answer agent]
    |     system: "You are a helpful research assistant..." [STATIC]
    |     messages: [system] + native_history + [{role:user, content: assembled_prompt}]
    |     assembled_prompt: context_prefix + answer_mode_contract + grounded_task_prompt
    |
    +-- policy.observe_turn()                        [policy.py]
          +-- reward_report()                        [rewards.py]
          +-- _reflect()                             [reflection agent]
          +-- update_conversation_frame()
          +-- update_landmarks()
          +-- update_idea_index(user_id)             [WP-F-2]
          +-- VocabularyStore.record_outcome()
          +-- save_policy_note()

---

## 8. GapSpec (WP-B)

Accumulates across BM25 retrieval levels:
  confirmed_entities, missing_entities, coverage_ratio, query_entities, entity_map
Used in: _external_retry_queries() -> steers retry toward missing entities.
WP-F-1: entity_synonyms substitutes bandit-selected synonyms for missing entities.
Passed: build_auto_context() -> chat.py -> policy.plan() via gap_spec= param.

---

## 9. WP Status

  See DEVELOPMENT_STATUS.md for current stage and pending tasks.

  WP-A: PubTator entity extraction (web_search.py)                   DONE
  WP-B: GapSpec (search_agent.py, policy.py)                         DONE
  WP-C: Snippet utility (search_agent.py)                            DONE
  WP-D: Per-step rewards (rewards.py, policy.py)                     DONE
  WP-E: Eval shape8 (reward_scorer.py, scenarios/)                   DONE
  WP-F: Onomasiological Bandit Memory (policy.py, store.py,
         idea_index.py, vocabulary_store.py)                         DONE

---

## 10. Eval Lab

  evals/lung_factuality_lab/scenarios/*.yaml            scenario definitions
  evals/lung_factuality_lab/data/conversations/seed/*.jsonl  seed turns
  evals/lung_factuality_lab/src/reward_scorer.py        scoring logic
  evals/lung_factuality_lab/configs/reward_shape_registry.yaml

  Shape ladder: correction_scope -> diagnostic_gate -> family_microfit ->
                sentinel_a -> sentinel_c -> promote/block.

  Multi-turn coreference scenarios (added 2026-06-12):
    multiturn_coreference_001   "a and b" option-letter resolution
    multiturn_context_drift_001 "yes, elaborate on the second one"

  Coreference reward components (active when coreference_data= passed to score_turn()):
    inter_turn_coreference   weight 0.10
    context_poor_resolution  weight 0.05
    conversation_continuity  weight 0.05

  ### Running the eval lab (2026-06-13)
  Runs INSIDE the api container (deps yaml/pydantic/urllib already present; NO httpx
  needed, NO rebuild). Eval tree is bind-mounted via docker-compose api volume
  `./evals:/lab/evals` with `PYTHONPATH=/app:/lab`. ⚠️ It is mounted at /lab/evals, NOT
  /app/evals: a mount nested under the /app bind mount gets MASKED (and deleting its
  backing dir from another container breaks it). No curl/ps/pytest/jq in the container.
    docker exec -d -w /lab ai-research-insights-api-1 sh -c \
      'python -m evals.lung_factuality_lab.src.run_batch \
         --config evals/lung_factuality_lab/configs/generated_sentinel_a.yaml \
         --assistant http --endpoint http://localhost:8080/chat/ \
         --api-key "$API_KEY" --tenant-id eval-lab \
         --out evals/lung_factuality_lab/runs/<name> --request-timeout 300 > /tmp/eval.log 2>&1'
  Internal uvicorn port=8080; chat route POST /chat/; HttpChatAdapter (urllib) needs the
  FULL chat URL; API_KEY is in the container env. ~12 min/scenario (real NVIDIA + HF NLI).
  Completion = recommendations.json at run root. Gate = failure_summary.missed_injected_traps==0.
  Output persists to host ./evals/.../runs/ via the mount (no docker cp needed).

  Last run shape8_sentinel_a_p7p1p3 (P-7+P-1+P-3 config, 2026-06-13): 8-scenario avg
  0.7247 (> 0.7148 baseline), missed_injected_traps=0 → sentinel_a RE-CLEARED. Per-scenario:
  caf_ecm 0.6045, citation_drift 0.6478, correction_scope 0.675, cross_cancer 0.8175,
  expert_hgf_met 0.8416, expert_tam_cd8 0.8048, hypoxia 0.6853, mdsc_treg 0.7212. The
  scenarios that regressed in the reverted ultra-550b experiment (cross_cancer, mdsc_treg)
  are now healthy.

---

## 11. Critical Engineering Rules

  1.  Grep before adding any new store/bandit/memory. Everything already exists.
  2.  plan() and build_auto_context() are independent -- not nested.
  3.  conversation_frame is fetched FIRST in plan() (since 2026-06-12).
  4.  _mem_query exists -- never use raw message for ES searches when context_poor.
  5.  Working buffer is NOT the same as episodic memory. Both exist, both needed.
  6.  VocabularyStore.enabled() must be checked before any bandit call.
  7.  _render_ideas() produces a vocabulary guide -- do not add a parallel renderer.
  8.  Agent routing is per-call (agent= param) NOT per-session.
  9.  _eff_msg is used for retrieval. Raw message is used for _answer_mode().
      These are intentionally separate -- do not conflate.
  10. NLI agent is routed via agent_models["nli"] (agent="nli") since P-1.
      HF MNLI (memory.nli_model) remains the primary NLI/factuality authority;
      the agent="nli" LLM path is a fallback only. See section 3f.
  11. Do NOT start local GPU services: llm, worker-gpu, models-init, rebel-extractor.
  12. Do NOT restart Docker, containerd, prune networks, stop unrelated containers,
      or reboot host without explicit user approval.
  13. NO HARDCODING. Every tunable/behavioural literal (threshold, default
      confidence, truncation length, batch size, retry count, model id, scan
      limit, etc.) MUST be a named, configurable constant — env-backed via
      os.getenv (matching zero_shot.py/nli.py) or a config/default.yaml field
      parsed in config.py. No magic numbers inline. Exception: a value that must
      stay in lock-step with another component (e.g. _MAX_OPTION_LETTERS mirrors
      the frontend) is a fixed NAMED constant with a comment explaining why it is
      not independently tunable. Any agent adding code follows this rule and
      records new knobs in DEVELOPMENT_STATUS.md.

---

## 12. Changelog

  2026-06-13 (batch 2 — post-merge follow-ups):
    - P-3 DONE: intent-resolution reward attribution. AutoContextPlan carries
      intent_resolution {tier, intent, source, confidence, state_key, action_key,
      effective_query}; observe_turn credits (intentres|len=BUCKET, TIER:INTENT)
      in the existing ActionValue table with the turn reward.
    - P-4 DONE (default off): _answer_mode question-type modes (novice/mechanism)
      may consider the resolved query for context-poor replies, gated by
      memory.answer_mode_consider_resolved_query. Utterance modes stay raw-message.
    - P-5 DONE: verified all factories static-prefix-first; added
      test_prompt_cache_ordering.py to lock the KV-cache invariant.
    - Streamlit: "Use server-side per-agent model routing" now DEFAULT ON; help/
      caption updated to include router + nli agents.
    - Synced blue-demon main with merged origin/main (PR #1 -> 0aca9f9).

  2026-06-13:
    - P-7 tier-1 zero-shot intent router (section 3g): nano NIM primary + HF MNLI
      fallback, inserted before the 120b in plan_auto_context. prior_context
      short-circuits the 120b; rewrite intents escalate. New file
      memory/intent_router.py; router agent_models entry; router_system_prompt +
      ROUTER_INTENT_HYPOTHESES in agent_prompts.py.
    - Answer-derived intent signal: prior-turn lettered-options detection mirrors
      the frontend checkbox trigger (shared contract) and biases prior_context.
    - Engineering rule 13 (NO HARDCODING) established; all router knobs env-backed.
    - Validated: 17/17 router unit tests, 36/36 search_agent regression, real
      NVIDIA nano + HF MNLI smoke (incl. options-hint path: "b" -> prior_context).
    - P-1 FIXED (section 3f): _llm_nli now routes via agent_models["nli"]
      (agent="nli"), system prompt from factory, env-backed truncation; nli entry
      added to agent_models (super-49b, max_tokens=1024). HF MNLI stays primary.
    - Git: baseline of all uncommitted 2026-06 work committed (05803c0) + merged
      to main (1df7386); P-7 + P-1 on feat/zero-shot-intent-router (PR #1).

  2026-06-12 (session 2):
    - Dynamic system prompt factory: app/prompts/agent_prompts.py
    - answer_system_prompt: 7 mode-specific identities + base sourcing policy
    - frame_system_prompt: NEW_QUERY vs AUGMENT_PRIOR mode declaration
    - intent_resolution_system_prompt: adapts to available context resources
    - external_query_system_prompt: resource-aware (PubMed/LitSense/PubTator)
    - ner_grounding_system_prompt: DISCOVERY vs CONFIRMATION mode
    - reflection_system_prompt: positive/negative/mixed reward polarity
    - Separation enforced: system=identity, user=task instructions
    - Extension guide: add to factory dict + PROMPT_REGISTRY, nothing else
    - 25/25 regression tests passed

  2026-06-12 (session 1):
    - Context-poor handling: _is_context_poor, _CONTEXT_POOR_EXACT, resolve_message_intent
    - Working buffer injected into resolve_message_intent (store.recent_messages 3 turns)
    - conversation_frame fetched FIRST in policy.plan() (previously after ES searches)
    - _mem_query derived from active_terms for ES searches when context_poor
    - Multi-turn coreference eval scenarios + reward components added
    - Bug fix: _is_context_poor limit=1->2 (proper nouns ranked 3rd now detected)
    - Bug fix: yes/second/I meant that added to _CONTEXT_POOR_EXACT
    - Full prompt audit: every agent prompt documented, dynamic vs static verified
    - ARCHITECTURE.md created (this file)
    - DEVELOPMENT_STATUS.md created (linked below)
