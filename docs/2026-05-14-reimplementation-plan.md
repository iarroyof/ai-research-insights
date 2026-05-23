# 2026-05-14 Detailed Reimplementation Plan

This plan supersedes the first implementation direction by incorporating the latest clarification:

> Triplets are valuable for retrieval, graphing, candidate selection, memory compression, and idea indexing, but the primary factuality/reward check should compare source/database PubMed sentences against atomic answer claims using biomedical NLI.

The plan is intentionally detailed. Future agents must not silently omit implementation gaps or defer hard pieces without marking them as partial/missing/deferred.

## 0. Planning Principles

### 0.0 Immediate Priority Update - No-Selection Chat Auto-Context

As of 2026-05-18, the next implementation priority is to make `/chat/` useful even when the user has not selected or pinned context.

Required behavior:

```text
User asks a question with no selected context
  -> memory/context policy still loads session state
  -> search-planning agent proposes term-based ES queries
  -> ES sentence/triplet retrieval runs BM25-first across refined query variants
  -> retrieved source sentences/papers become answer context and citation candidates
  -> answer generation uses this auto-context
  -> observe_turn records the search strategy, query variants, results, rewards, and notes
  -> Q-like action-value telemetry scores how the search strategy worked, not the exact terms
```

Compatibility constraints:

- Keep existing selected/pinned context behavior unchanged when the user selects evidence.
- Keep BM25/term retrieval primary.
- Use the hosted context-manager LLM only as a query/refinement and note-writing assistant; deterministic query generation must remain the fallback.
- Search policy learning remains inference-time telemetry only: no weight updates.
- The action-value table should bucket search behavior patterns such as query count, term breadth, use of biomedical synonyms, and result density; it should not key on exact query terms.
- The search agent should maintain compact textual notes about successful or failed search patterns so later turns can improve retrieval.

Implemented first slice on 2026-05-18:

- `app.memory.search_agent` now creates deterministic biomedical query variants and optionally refines them with the hosted context-manager LLM.
- `/chat/` auto-builds context when no items are pinned, feeds the retrieved ES snippets through the existing prompt/citation path, and preserves pinned-context behavior when items are selected.
- `ContextPolicy.observe_turn()` records auto-context search plans, adds auto-context snippets to reward context, updates a separate Q-like search action-value bucket, and persists textual search-policy notes.
- Streamlit chat works without selected context and exposes search notes in session diagnostics.
- This remains compatible with the original memory-management plan: it is BM25-first, has deterministic fallback, and does not train model weights.

Implemented second slice on 2026-05-18:

- Auto-context retrieval is now a structured multilevel search action:
  - title level: paper-title-heavy search to identify candidate papers and useful terminology;
  - paper/chunk level: broader article-context search;
  - sentence/triplet level: exact source-sentence evidence search.
- The search agent extracts compact feedback terms from earlier level hits and expands later level queries with those terms.
- The hosted context-manager LLM prompt now knows the level roles and that later searches are self-informed by earlier results.
- `/chat/` exposes `levels` and `level_reports` in auto-context citation metadata.

Implemented term-steering slice on 2026-05-19:

- The search agent now infers a compact biomedical search frame for ambiguous cancer/TME questions.
- In cancer/TME context, `functional synergy` is bridged toward mechanistic synergy, crosstalk, stromal/immune/metabolic cooperation, hypoxia, angiogenesis, ECM remodeling, EMT, immune evasion, CAF/TAM/Treg/MDSC, and lung cancer/NSCLC terminology.
- Mathematical/pharmacological synergy meanings such as combination index, CI value, dose response, drug synergy, CTCAE, adverse events, and toxicity are avoided unless the user explicitly asks for drug-combination/pharmacological context.
- Feedback terms from prior search levels are filtered so caption/generic terms such as `figure`, `show`, and `define` do not steer later searches.

Implemented general evidence-assembly slice on 2026-05-22:

- Auto-context refinement now treats early search hits as candidate evidence, not automatically trusted vocabulary. Feedback terms from a result must stay anchored to the active query/refinement frame or come from a structured relation before they steer the next retrieval level.
- This gate is intentionally general. It blocks noisy multilevel vocabulary drift for broad questions across the local corpus or external sources without assuming the user is asking about cancer, tumor microenvironment, or one mechanism family.
- Auto-context plans now include an `evidence_assembly` summary with information-need shape, query ambiguity, retrieved level coverage, distinct-paper count, refinement-quality counters, and clarification guidance for underspecified evidence puzzles.
- The answer prompt receives a concise evidence-assembly block that instructs it to use retrieved snippets as pieces, avoid inventing unsupported bridges, state a supported partial structure when edges are missing, and ask one focused clarification when the relation remains underspecified.
- Runtime reward traces now expose evidence-assembly quality and query-refinement drift counters. Accepted/rejected feedback telemetry can be reward-shaped in the live lab without leaking exact user terms into Q-like search state keys.
- Remaining work is iterative: add result-ranking calibration across local/external sources, richer candidate-query/clarification loops, and live lab reward-shaping over representative broad questions rather than one topic-specific regression.

Implemented candidate-frame and live-lab continuation on 2026-05-22:

- Ambiguous no-selection queries now search candidate evidence frames through the normal multilevel variant loop. Runtime plans expose frame ids, per-frame result counts, candidate nodes, covered/missing nodes, relation-evidence counts, and evidence-puzzle edge status.
- If the evidence puzzle is ambiguous and needs clarification, `/chat/` streams a plain textual opening clarification prefix inside the answer before model-generated continuation. This is not a UI selector.
- Reward traces now penalize unsupported bridge claims when the evidence puzzle reports partial or missing edges. The lab HTTP adapter preserves citations/memory-debug SSE metadata so live turn scores can include evidence-assembly quality and bridge-safety components.
- The lab now includes `biomedical_ambiguous_evidence_assembly_001` and `configs/evidence_assembly_microfit.yaml` for general evidence-assembly shaping outside one lung-cancer mechanism family.
- Saved-answer replay and live family-fit runs are recorded in `configs/reward_shape_registry.yaml` as Shape7 candidate evidence. Keep Shape7 candidate-only until a stratified guard clears the broader runtime search/prompt changes.

Implemented cross-domain follow-up evidence-assembly continuation on 2026-05-22:

- Specialized search frames are now chosen from the current user turn. Search policy notes remain useful for strategy and explicit follow-up continuity, but an old successful TME note no longer forces a later cross-domain biomedical query into a TME bridge.
- Explicit follow-up turns such as a request to develop previously suggested frameworks can reuse compact prior query frames from saved search-plan notes. This keeps a supported frame available when the follow-up text itself is referential.
- Multilevel feedback now rejects low-value correspondence/author metadata, patient-physician dialogue boilerplate, and cross-domain case-treatment vocabulary when the active search is an analogy or inspired-strategy probe and the hit lacks the target domain.
- The evidence-assembly prompt now forbids naming a candidate therapy, agent, framework, pathway, or experiment unless supplied context supports that named candidate.
- The live evidence-assembly microfit includes `cross_domain_fungal_therapy_assembly_001`, seeded from the cancer/fungal-infection analogy trace, so future shaping observes clarification, frame continuity, and unsupported-framework pressure across multiple turns.
- Claim extraction now uses alias boundary matching so words such as `metabolic`, `vitamin`, and `myofibroblast` do not spuriously activate `MET/c-MET`, `TAM`, or broad CAF scoring.
- Explicit follow-ups can carry prior query-frame nodes into evidence-puzzle assembly. A hard clarification guard now prevents answer generation from continuing when the evidence puzzle says clarification is required and edge support is missing or partial.
- Latest live microfit status: `shape7d` remains the best observed two-scenario evidence-assembly microfit (`avg_reward=0.9017`, failure count `1`). `shape7j` is candidate-only (`avg_reward=0.8987`, failure count `1`) because it is slightly below `shape7d` despite fixing a clean single-run seed. Do not promote Shape7 until a stratified guard clears.

Implemented seeded evaluation/reward-shaping slice on 2026-05-19:

- `evals/lung_factuality_lab` is now the modular evaluation laboratory for lung-cancer factuality, conversation drift, claim support, injected traps, recommendations, and regression-plan generation.
- Seeded conversations are used as fixed multi-turn tests; generated variants can perturb those seeds.
- Reward shaping was run on seeded batches and improved dummy-adapter average reward from `0.3514` to `0.4469` while reducing missed injected traps to `0`.
- The evaluator was corrected to distinguish a resisted false premise from a missed trap, to use alias-aware mechanism graph matching, to avoid `MET` false positives in `metabolic`, and to keep scenario IDs in batch recommendations/regressions.
- This lab should be used before changing memory, factuality, or search policies: run the seeded batch, inspect `conversation_trace.json`, `failure_board.json`, `recommendations.md`, and `regression_plan.yaml`, then decide the next implementation change.

Implemented large generated corpus benchmark slice on 2026-05-19:

- The generated `lung_factuality_large_corpus_v1` artifact has been integrated as a first large fixed benchmark.
- The lab now supports 120 generated conversations and 840 user turns spanning HGF/MET, TAM/CD8, hypoxia/HIF/PD-L1, MDSC/Treg, CAF/ECM stiffness, cross-cancer transfer, correction/scope, and citation-drift families.
- `configs/generated_batch_runs.yaml` runs the full generated benchmark.
- The benchmark was used to shape evaluator/reward behavior:
  - generated evidence and trap banks are loaded by default;
  - generated missing trap objects are synthesized from turn metadata;
  - mechanism and entity matching includes aliases for generated biomedical concepts;
  - cross-domain transfer detection handles negated/corrective language;
  - curated wrong variants take precedence over generic scope drift;
  - generated wrong-answer replay fixtures match by base scenario and variant.
- Final local and hosted dummy large benchmark: average reward `0.88`, `missed_injected_traps=0`.
- Final local and hosted generated wrong-answer replay benchmark: average reward `0.74`, `missed_injected_traps=0`.
- The next evaluation step is a slower target-chatbot run against live `/chat/`, using the same corpus, after deciding how much hosted-provider traffic is acceptable.

Live reward-shaping ladder update on 2026-05-21:

- Live reward shaping now uses `evals/lung_factuality_lab/configs/reward_shape_registry.yaml` as the source of truth for accepted, candidate, and blocked reward/evaluator shapes.
- A later agent may diagnose one live conversation or one saved-live replay trace before a code change, but one conversation is not enough to accept a shape.
- After a narrow test-backed change, use saved-answer replay to isolate the evaluator/reward delta, run the relevant live family microfit, then run a stratified live sentinel guard before broader Stage 2 or Stage 3 work.
- Current ladder assets:
  - family fits: `configs/generated_microfit_correction_scope.yaml`, `configs/generated_microfit_tam_cd8.yaml`;
  - guards: `configs/generated_sentinel_a.yaml`, `configs/generated_sentinel_b.yaml`, `configs/generated_sentinel_c.yaml`;
  - protected holdout: `configs/generated_semantic_drift_holdout_v1.yaml`.
- Replay-only reward improvements are diagnostic. Live endpoint fit and guard evidence decide promotion, and rejected shapes must remain blocked in the registry so the loop does not revisit them.

### 0.1 Primary Product Goal

Build a research chatbot that answers from disease-specific PubMed/context corpora while maintaining context over long sessions, minimizing forgetting, and giving traceable, evidence-grounded answers.

The chatbot must behave like a memory-aware research assistant:

- knows what source sentences were available;
- can retrieve relevant source sentences and triplets;
- can judge whether each answer claim is supported, contradicted, or unsupported;
- can explain uncertainty;
- can ask the user to resolve disputed facts;
- can improve its future context selection through inference-time reward traces, not weight updates.

### 0.2 Core Technical Rule

Use this division of labor:

```text
Original database/source sentences = authoritative evidence units.
Biomedical NLI = factuality/support/contradiction reward signal.
Triplets = retrieval, structure, memory compression, candidate selection, graph, idea-index.
LLM = generation, lightweight reflection, claim normalization, fallback evaluator when no NLI service exists.
OpenSearch = term-first memory and evidence index.
Optional vectors = fallback when term retrieval is weak.
```

### 0.3 Why This Change Matters

Triplets alone are too lossy for truth checking. They may lose:

- negation;
- uncertainty/speculation;
- causal direction;
- experimental population;
- intervention/outcome role;
- species/cell line;
- disease subtype;
- dose/time;
- section context;
- comparison group;
- whether the original sentence reported, hypothesized, reviewed, or denied a fact.

Therefore, triplets should identify candidate evidence, but the final factuality reward should use the original source sentence/window as premise.



### 0.4 Priority Update - Higher-Confidence Semantic Drift Detection Model Panel

As of 2026-05-20, higher-confidence semantic drift detection is added to the implementation plan. This must be developed as a modular evaluation-and-runtime panel, not as a single keyword rule or a single LLM judgment.

Current confirmed runtime baseline:

- Hosted mode remains active.
- Main chat/context-manager LLM uses NVIDIA API.
- HF API is available for biomedical NLI and zero-shot classification.
- Current configured NLI model is `pritamdeka/PubMedBERT-MNLI-MedNLI`, so the active NLI baseline is MedNLI/MNLI-finetuned PubMedBERT, not a BioNLI-specific model.
- BM25/term search remains primary.
- Local embeddings, vector indexing, and trained RL weights remain deferred.
- `evals/lung_factuality_lab` is the required test harness for this work.

Goal:

```text
Detect semantic drift across answers, retrieval, citations, and memory state with traceable confidence, while avoiding false penalties for valid caveats, uncertainty statements, and properly caveated cross-domain background.
```

The model panel should classify drift at these levels:

1. Turn-level answer drift: the answer leaves the user's current biomedical scope.
2. Conversation-level drift: user corrections and scope constraints are not preserved across turns.
3. Retrieval/search drift: query refinements and retrieved source sentences drift away from the biomedical question.
4. Memory drift: unsupported, contradicted, stale, or off-scope claims remain active or are promoted.
5. Citation/evidence drift: evidence from another disease/domain is presented as direct lung-cancer proof.

#### Phase 1 - Highest-impact implementation, no new model endpoint required

Priority: highest. Implement first because it can run now in hosted mode with existing NVIDIA and HF API infrastructure.

Phase 1 components:

1. `scifact_verifier` module
   - Add a SciFact-style structured verifier over answer claims, retrieved evidence sentences, active memory items, and search-result snippets.
   - This is prompt-instructed LLM verification, not fine-tuning.
   - Required labels:
     - `supported`
     - `refuted`
     - `not_enough_info`
     - `supported_with_scope_caveat`
     - `out_of_scope`
     - `citation_drift`
     - `memory_drift`
     - `retrieval_drift`
   - Required output fields:
     - claim text;
     - evidence sentence ids;
     - evidence scope: lung-cancer-specific, general oncology, other-cancer transfer, clinical-trial, unrelated;
     - drift type;
     - confidence;
     - short rationale;
     - recommended memory action: promote, keep_active_with_caveat, keep_inactive, demote, mark_contradicted.

2. Runtime prompt integration
   - Add SciFact-style instructions to the memory/context-manager agent, not to the answer-generation prompt alone.
   - The memory agent must verify claims before promoting or reusing them.
   - The search agent must use verifier feedback to avoid drifting into high-frequency but irrelevant terms.

3. Deterministic scope/evidence gates
   - Add a lightweight gate before LLM judgment using existing extracted entities, scenario scope, source metadata, and correction state.
   - Detect obvious mismatches:
     - lung cancer vs other cancer;
     - TME biology vs drug approval/pricing/treatment timeline;
     - mechanistic biology vs pharmacological/mathematical synergy;
     - claim evidence without disease/mechanism overlap;
     - rejected false premise being repeated as if still plausible.
   - The gate should not make final high-confidence decisions alone except for explicit excluded-topic drift.

4. Existing MedNLI baseline use
   - Keep `pritamdeka/PubMedBERT-MNLI-MedNLI` as the baseline NLI member.
   - Use it for premise/hypothesis support/refute scores after comparability gating.
   - Do not expect this model alone to solve literature-grounded mechanistic drift.

5. Zero-shot drift classifier
   - Use current HF zero-shot path for candidate drift labels such as:
     - `lung cancer TME mechanism`
     - `general oncology background`
     - `other cancer transfer evidence`
     - `clinical treatment recommendation`
     - `drug approval or pricing`
     - `not enough evidence`
     - `retrieval/search drift`
   - Treat zero-shot as a weak panel signal, not authoritative truth.

6. Lab integration
   - Add drift diagnostics to `conversation_trace.json`, `failure_board.json`, and `turn_scores.json`.
   - Add reward components:
     - `semantic_scope_alignment`;
     - `correction_frame_adherence`;
     - `evidence_scope_match`;
     - `memory_scope_safety`;
     - `retrieval_scope_quality`.
   - Add regression scenarios from existing generated families first:
     - HGF/MET false premise and correction;
     - TAM/CD8 immunosuppression direction;
     - hypoxia immune escape;
     - MDSC/Treg suppression;
     - CAF/ECM stiffness;
     - cross-cancer transfer;
     - citation drift;
     - correction scope TME-only.

Phase 1 success criteria:

- Seeded and generated wrong-answer replay still have `missed_injected_traps=0`.
- Live Stage 1 average reward does not regress relative to the current accepted `shape4` registry entry.
- Drift diagnostics identify why a turn is drift, caveated background, or valid uncertainty.
- False positives on rejected false premises and caveated transfer statements do not increase.

#### Phase 2 - Model-panel experiments in the lab

Priority: high. Begin after Phase 1 trace schema and verifier outputs are stable.

Phase 2 compares available model-panel options using replay first, then live endpoint runs.

Panel candidates to test:

1. MedNLI baseline: `pritamdeka/PubMedBERT-MNLI-MedNLI`
   - Role: general medical NLI baseline.
   - Current active model.
   - Expected strength: clinical/medical entailment.
   - Expected weakness: mechanistic literature drift and disease-scope transfer.

2. SciFact-style LLM verifier
   - Role: literature claim verification behavior without fine-tuning.
   - Expected strength: support/refute/no-info reasoning with evidence rationales.
   - Expected weakness: possible LLM over-judgment; must be bounded by evidence ids and deterministic gates.

3. SciFact/claim-verification HF models where hosted inference is available
   - Candidate family:
     - `shidey/deberta-v3-mednli-scifact-open-sentence-nli`
     - `MilosKosRad/DeBERTa-v3-large-SciFact`
   - Role: scientific claim verification signal.
   - Test only if accessible through hosted HF API or a dedicated endpoint.
   - Do not promote until compared on replay and live traces.

4. BioNLI model candidates
   - Candidate family:
     - `Bam3752/PubMedBERT-BioNLI-LoRA` or any accessible BioNLI-derived classifier.
   - Role: mechanistic polarity, role-flip, and adversarial mechanism perturbation signal.
   - Expected strength: HGF/MET direction, TAM/CD8 role direction, mechanism chain perturbations.
   - Expected weakness: model availability and calibration risk.

5. NLI4CT candidates
   - Candidate family:
     - `domenicrosati/ClinicalTrialBioBert-NLI4CT` or similar NLI4CT endpoint.
   - Role: clinical-trial robustness and cancer-adjacent entailment signal.
   - Use only for clinical-trial evidence drift, not general lung TME mechanism truth.
   - Do not make it authoritative for lung-cancer biology.

6. Zero-shot drift classifier
   - Role: low-cost weak signal for drift categories and evidence scope.
   - Continue using as auxiliary panel evidence.

Experiment protocol:

```text
For each candidate panel configuration:
  1. Run seeded replay and generated wrong-answer replay.
  2. Run saved-answer replay on prior live traces.
  3. Run Stage 1 live only if replay improves without higher false positives.
  4. Run Stage 2 live only if Stage 1 improves over current accepted shape.
  5. Register every shape in reward_shape_registry.yaml.
  6. Block rejected shapes so the loop cannot revisit them.
```

Required panel configurations:

- `panel_a`: deterministic gate + MedNLI baseline + SciFact-style LLM verifier.
- `panel_b`: `panel_a` + zero-shot drift classifier.
- `panel_c`: `panel_b` + SciFact HF model when available.
- `panel_d`: `panel_b` + BioNLI candidate when available.
- `panel_e`: `panel_b` + NLI4CT candidate only for clinical-trial drift tests.

Promotion rule:

- Replay-only improvement is not sufficient.
- A panel becomes accepted only if live Stage 1 and at least the relevant Stage 2 subset improve or remain stable versus the current accepted registry shape.
- If a panel improves a subset but regresses another subset, create a gated specialist panel rather than replacing the global panel.

Phase 2 success criteria:

- `reward_shape_registry.yaml` records every panel attempt and blocks rejected variants.
- Drift false positives decrease on caveated transfer and rejected false-premise turns.
- Drift false negatives decrease on off-topic clinical/pricing/timeline turns.
- Memory promotion/demotion decisions become traceable in the lab outputs.

#### Phase 3 - Full confidence benchmark and production promotion

Priority: later. Start only after Phase 1 and Phase 2 produce stable results.

Phase 3 actions:

1. Run full generated Stage 3 benchmark: 120 conversations / 840 turns.
2. Run live Stage 3 only if hosted-provider traffic is acceptable.
3. Add external metrics export for panel latency, retries, disagreement, and confidence calibration.
4. Add optional dedicated endpoints only for panel models that proved useful in Phase 2.
5. Integrate final drift scores into runtime memory promotion/demotion and search-agent notes.
6. Add UI diagnostics for drift evidence:
   - drift type;
   - evidence ids;
   - panel votes;
   - confidence;
   - memory action;
   - search action-value impact.

Phase 3 success criteria:

- Full benchmark improves over the accepted current shape without increasing severe false positives.
- Runtime drift diagnostics are explainable to the user/developer.
- Memory state no longer promotes contradicted/off-scope claims.
- Search notes reflect drift feedback without keying policy on exact query terms.

Deferred until Phase 3 or later:

- Local embedding/vector search as a primary drift signal.
- Trained RL/model weights.
- Lung-cancer-specific NLI fine-tuning.
- Dedicated BioNLI/NLI4CT/SciFact endpoints unless Phase 2 proves clear value.


## 1. Research And User-Signal Basis For Priorities

### 1.1 Research Signals

The systematic-review tooling literature emphasizes:

- search;
- screening;
- data extraction;
- synthesis;
- quality assessment;
- reporting;
- usability;
- transparency;
- traceability;
- evaluation.

Bolaños et al. 2024 specifically discuss AI-supported SLR workflows and identify knowledge graphs, usability, and standardized evaluation/transparency as major challenges and directions.

SciLit proposes an integrated research workflow that recommends relevant papers, extracts highlights, and suggests reference/citation sentences based on context and keywords.

Elicit-style research assistants are valued because they search scholarly corpora and produce research-backed summaries, not just generic summaries.

### 1.2 User/Product Signals From Public Discussions

Recurring "nice to have" or "actually useful" research-AI features:

1. Answers with citations and page/source locations.
2. Ability to extract relevant information from many papers.
3. Chat with selected documents or selected source sets.
4. Dynamic RAG where the user controls included sources per round.
5. Evidence tables with extracted fields, findings, limitations, sample size, method, and citation.
6. Citation/semantic graph exploration.
7. Knowledge-base memory that scales without becoming a mess.
8. Highlight extraction and source sentence suggestions.
9. Research boards/notebooks that keep context organized.
10. Trustworthy "no hallucination" behavior and source-grounded answers.

These signals make the following features highest priority:

- evidence-grounded factuality and citation traceability;
- evidence tables;
- selected-source context control;
- idea-index/knowledge memory;
- semantic/citation graph navigation;
- feedback/correction loop.

Fork/thread workflows are valuable but should come later.

## 2. Target Architecture

### 2.1 Runtime Flow

```text
User question
  -> privacy/safety preprocessing
  -> context manager state load
  -> source sentence retrieval
  -> triplet retrieval/graph expansion
  -> idea-index retrieval
  -> optional web retrieval if allowed
  -> context assembly with token budget
  -> answer generation
  -> answer claim extraction
  -> evidence candidate selection per claim
  -> premise/hypothesis comparability gate
  -> biomedical NLI factuality scoring
  -> reward computation
  -> warnings/corrections
  -> memory/landmark/idea-index/trace update
```

### 2.2 Memory Tiers

Tier 1: Working Buffer

- Recent exact turns.
- Current user question.
- Current selected source sentences.
- Current active landmarks.
- Must be token-budget constrained.

Tier 2: Episodic Session Summary

- Rolling summary of session goals.
- Accepted facts.
- Disputed facts.
- User preferences.
- Open questions.
- Correction decisions.
- Not currently implemented; high priority after factuality.

Tier 3: Source Evidence Memory

- PubMed/database sentences and windows.
- Citation metadata.
- Paper metadata.
- MeSH terms/disease tags.
- Section labels.
- Existing project has sentence/triplet retrieval pieces but not a claim-evidence table.

Tier 4: Triplet/Graph Memory

- REBEL/OpenIE triplets.
- Biomedical entity probability filters.
- Source sentence pointer.
- Graph neighborhoods.
- Relation/path expansion.

Tier 5: Idea-Index

- Aggregate ideas/concepts.
- Frequency, recency, co-occurrence.
- Parent/child hierarchy.
- Linked entities/triplets/source sentences.
- Reward association.
- Transfer candidates.
- Not currently implemented.

Tier 6: Policy Trace Memory

- State, action, reward, evidence, selected/rejected context.
- Future Q-table/contextual-bandit training data.
- Partially implemented as policy traces.

## 3. Phase 1 - Evidence-Grounded Factuality Reimplementation

Priority: highest.

### 3.0 Mandatory Module Test Gate

Every module in this phase must be tested immediately after implementation. Do not build higher-level chat behavior on top of an untested module.

Required gate:

```text
module implemented
  -> unit tests
  -> mocked integration tests
  -> real-provider smoke test where applicable
  -> status tracker update
  -> only then integrate into next layer
```

Provider order:

- For main chat and context-manager LLM calls: NVIDIA hosted API first, local models only after GPU approval.
- For biomedical NLI: Hugging Face Inference API for `pritamdeka/PubMedBERT-MNLI-MedNLI` first, local model only as fallback.
- For local models: run the exact same test suite and compare output shape, latency, and failure modes against the NVIDIA/HF paths.

### 3.1 Replace Primary NLI Unit

Current partial behavior:

- answer triplets are compared to retrieved triplets/source sentences.

Required behavior:

```text
premise = source/database sentence or small source sentence window
hypothesis = atomic answer claim
```

Implementation tasks:

1. Add `app.memory.claims`.
2. Split assistant answer into candidate answer sentences.
3. Convert answer sentences into atomic claims.
4. Preserve original answer sentence text for display.
5. For each claim, select evidence source sentence candidates.
6. Run comparability gate.
7. Run NLI only on comparable candidates.
8. Aggregate support state per claim.

Atomic claim schema:

```json
{
  "claim_id": "stable hash",
  "answer_sentence_id": "stable hash",
  "answer_sentence": "...",
  "claim_text": "...",
  "claim_type": "biomedical_fact|method|recommendation|uncertainty|citation_statement|other",
  "entities": [],
  "relations": [],
  "negation": false,
  "speculation": "none|possible|likely|uncertain",
  "requires_citation": true
}
```

Claim extraction strategy:

- Start deterministic:
  - sentence split on punctuation and biomedical abbreviations carefully;
  - skip non-factual social text;
  - keep sentence as one claim if atomic enough.
- Add LLM/reflection optional step later:
  - decompose compound sentence into atomic claims;
  - output JSON only;
  - fallback to deterministic sentence-level claims.

### 3.2 Evidence Candidate Selection

Candidate sources:

1. Source sentences already included in the prompt.
2. Pinned snippets selected by user.
3. Retrieved memory/source sentences.
4. Same-paper neighboring sentence windows.
5. Triplet-linked source sentences.
6. BM25 search over disease-specific corpus.
7. Vector fallback when BM25 low confidence.
8. Optional web only if enabled and clearly marked external.

2026-05-18 implementation note: vector fallback now exists in `app.search.os_client.os_hybrid_query()` as an optional OpenSearch kNN fallback when a caller supplies `filters.query_vector`; BM25 remains primary. Query-vector generation and vector index/template validation are still not wired, so auto-context does not yet invoke vector fallback automatically.

2026-05-18 implementation note: hosted BioNLI now supports pair batching through `app.memory.nli.classify_nli_batch()`. Claim support and answer-triplet scoring use the batched path by default, so factuality/reward validation can scale better while embeddings/vector work remains deferred.

Candidate evidence schema:

```json
{
  "evidence_id": "stable hash",
  "source": "prompt_context|pinned|memory|triplet|bm25|vector|web",
  "paper_id": "...",
  "pmid": "...",
  "pmcid": "...",
  "title": "...",
  "section": "...",
  "sent_id": "...",
  "sentence_text": "...",
  "window_text": "...",
  "mesh_terms": [],
  "disease_terms": [],
  "retrieval_score": 0.0,
  "triplet_links": [],
  "was_in_model_prompt": true
}
```

Important:

- `was_in_model_prompt` must be tracked.
- If answer uses evidence not in prompt but found afterward, label separately as "post-hoc evidence".
- Reward should favor support from prompt-provided evidence over post-hoc evidence.

### 3.3 Premise/Hypothesis Comparability Gate

NLI is invalid if the premise and hypothesis are not about the same factual proposition. Add a gate before NLI.

Gate output:

```json
{
  "comparable": true,
  "score": 0.0,
  "reasons": [],
  "blocking_mismatch": "none|disease|entity|relation|population|species|cell_line|intervention|outcome|directionality|negation|temporality|section_context"
}
```

Deterministic features:

- lexical/entity overlap;
- MeSH/disease overlap;
- triplet subject/object overlap;
- biomedical entity class overlap;
- negation mismatch;
- relation/action verb overlap;
- same paper/section proximity;
- retrieval score.

LLM optional gate:

- Use local/NVIDIA context manager only if deterministic gate is uncertain.
- Prompt must return JSON only.
- Do not expose chain-of-thought.

Gate decision:

- `score >= 0.65`: run NLI.
- `0.35 <= score < 0.65`: run NLI but mark low comparability; weak reward.
- `< 0.35`: do not run NLI; retrieve better evidence.

### 3.4 Biomedical NLI Service

Current:

- `nli_provider=hf_api` default after the latest update.
- Hugging Face Inference API support should be the first real NLI path.
- `nli_provider=http` is scaffolded.
- `nli_provider=heuristic` is fallback only.

Required:

- Prefer remote Hugging Face Inference API first for `pritamdeka/PubMedBERT-MNLI-MedNLI`.
- Deploy an HTTP/local service later only if needed for throughput, cost, privacy, or offline operation.
- Candidate models:
  - PubMedBERT fine-tuned on MedNLI/MNLI-style data;
  - BioLinkBERT/BioClinicalBERT NLI variants;
  - DeBERTa/BioBERT variants if biomedical calibration is acceptable.

Hugging Face API configuration:

```yaml
memory:
  nli_provider: hf_api
  nli_model: pritamdeka/PubMedBERT-MNLI-MedNLI
  hf_api_token: ${HF_API_TOKEN:-}
  hf_api_base_url: https://router.huggingface.co/hf-inference/models
```

Token installation:

- Local token file:

```text
/mnt/d/UserFolders/Documents/hf_huggingface_token.txt
```

- Do not print the token.
- Parse only a token-like value, normally `hf_...`.
- Write it to blue-demon project `.env`:

```text
HF_API_TOKEN=<token>
NLI_PROVIDER=hf_api
```

- Target `.env`:

```text
/home/iarroyof/sabia/ai-research-insights/.env
```

- Docker Compose already passes `HF_API_TOKEN` and `NLI_PROVIDER` to the API service.

Expected implementation behavior:

- If `HF_API_TOKEN` is present, call HF Inference API.
- If HF request fails due to temporary model loading, use `wait_for_model`.
- Use SEP-string input for the hosted text-classification pipeline:

```text
hypothesis [SEP] premise
```

- If HF input format is rejected, retry alternate premise/hypothesis serialization.
- If no token exists, fall back to heuristic with warning, not silent success.
- Local model loading is fallback and requires explicit approval if it uses GPU.

Service API:

```http
POST /predict
{
  "premise": "...",
  "hypothesis": "...",
  "model": "..."
}
```

Response:

```json
{
  "label": "entailment|contradiction|neutral",
  "scores": {
    "entailment": 0.0,
    "contradiction": 0.0,
    "neutral": 0.0
  },
  "model": "...",
  "version": "...",
  "calibration": "uncalibrated|temperature_scaled|isotonic"
}
```

Infrastructure constraints:

- Do not deploy on GPU until user permits.
- CPU service is acceptable for initial low-throughput validation if model size permits.
- NVIDIA hosted LLM can be used as fallback evaluator, but should not replace calibrated NLI.
- HF API must be tested before local NLI.

Module tests:

- token cleaning: parse only `hf_...` from token file/prose;
- label mapping: contradiction/entailment/neutral and `LABEL_0/1/2`;
- response shape parsing: flat and nested list;
- fallback input format after 400/422;
- no-token fallback warning;
- synthetic premise/hypothesis fixtures.

### 3.5 Factuality Aggregation

For each claim:

```json
{
  "claim_id": "...",
  "claim": "...",
  "best_label": "entailed|contradicted|unsupported|not_comparable|not_checked",
  "best_entailment": 0.0,
  "max_contradiction": 0.0,
  "best_evidence_id": "...",
  "candidate_count": 0,
  "prompt_supported": true,
  "needs_user_confirmation": false
}
```

Reward contribution:

- entailed by prompt evidence: strong positive;
- entailed only by post-hoc evidence: small positive, maybe retrieval miss penalty;
- neutral/unsupported: moderate penalty if factual claim required support;
- contradiction: strong penalty and warning;
- not comparable: retrieval/premise-selection penalty, not factuality penalty;
- not checked: uncertainty penalty.

### 3.6 Output To User

Add optional SSE events:

- `evidence_table`
- `claim_support`
- `consistency_warning`
- `memory_debug`

Do not overwhelm user by default. Provide compact warning:

```text
Some answer claims were not fully supported by the selected source sentences. I found a possible contradiction for claim X; please confirm which source/fact should be treated as authoritative.
```

## 4. Phase 2 - Evidence Table And Citation Traceability

Priority: highest, because public user signals strongly value referenced answers and extraction from many papers.

### 4.1 Evidence Table Backend

Create a normalized evidence table object per answer:

```json
{
  "answer_id": "...",
  "session_id": "...",
  "turn_index": 12,
  "claims": [
    {
      "claim": "...",
      "status": "entailed|contradicted|unsupported|not_comparable",
      "source_sentence": "...",
      "source_window": "...",
      "pmid": "...",
      "pmcid": "...",
      "paper_title": "...",
      "section": "...",
      "sent_id": "...",
      "nli_label": "...",
      "nli_scores": {},
      "comparability": {},
      "was_in_prompt": true
    }
  ]
}
```

Store in OpenSearch as:

- `doc_type=evidence_table`
- linked to `policy_trace`
- linked to `session_id`
- linked to answer message.

### 4.2 Citation Requirements

When answering biomedical factual questions:

- every factual claim should either cite a source or be marked as synthesis/uncertain;
- citations should refer to source snippets with stable IDs;
- unsupported claims should be rewritten or caveated in a future version.

### 4.3 UI-Ready Payload

Even before UI work, backend should produce:

- compact evidence table;
- downloadable JSON;
- source sentence references;
- warning flags;
- claim status.

## 5. Phase 3 - Idea-Index / Concept-Frequency Tree

Priority: high. The user specifically requested this and it supports long-session memory.

### 5.1 Purpose

The idea-index is not just keyword tags. It is an aggregate, term-index-like structure over recurring ideas/meaning/concepts.

It should answer:

- Which concepts recur in this session?
- Which concepts are central to current user goals?
- Which concepts are associated with high/low reward answers?
- Which concepts co-occur?
- Which concepts form parent/child hierarchies?
- Which concepts should be kept in prompt vs long-term memory?

### 5.2 Idea Document Schema

```json
{
  "doc_type": "idea",
  "idea_id": "stable canonical hash",
  "tenant": "...",
  "session_id": "...",
  "scope": "session|tenant|global_policy",
  "label": "biomedical NLI factuality",
  "canonical_terms": ["factuality", "NLI", "entailment"],
  "aliases": ["claim support", "source verification"],
  "definition": "...",
  "frequency": 12,
  "turn_frequency": 7,
  "last_seen": "...",
  "first_seen": "...",
  "recency_score": 0.0,
  "reward_avg": 0.0,
  "reward_delta": 0.0,
  "importance": 0.0,
  "parent_idea_id": "...",
  "path": ["research_assistant", "factuality", "biomedical_nli"],
  "depth": 2,
  "children": [],
  "co_occurs": [
    {"idea_id": "...", "count": 4, "pmi": 0.0}
  ],
  "linked_entities": [],
  "linked_triplets": [],
  "linked_source_sentences": [],
  "linked_policy_actions": [],
  "contradiction_count": 0,
  "correction_count": 0
}
```

### 5.3 Update Logic

After each turn:

1. Extract candidate ideas from:
   - user question;
   - answer claims;
   - retrieved evidence;
   - triplets;
   - landmarks;
   - corrections.
2. Normalize terms:
   - lowercase;
   - lemmatize where available;
   - map aliases;
   - biomedical entity normalization if available.
3. Merge into existing idea docs by:
   - exact canonical key;
   - alias match;
   - high lexical overlap;
   - optional embedding fallback.
4. Increment frequency/turn frequency.
5. Update recency.
6. Update reward association:
   - if idea was in selected context and answer reward high, increase usefulness;
   - if idea led to contradiction/unsupported claim, increase caution.
7. Update co-occurrence edges for ideas in same turn/context.
8. Update hierarchy:
   - deterministic parent from MeSH/disease/category when possible;
   - LLM-assisted parent suggestion only if uncertain.

### 5.4 Retrieval Boost

Context policy should score ideas:

```text
idea_utility =
  bm25_or_term_overlap
  + log(1 + frequency)
  + recency_weight
  + reward_avg
  + landmark_overlap
  + correction_authority_boost
  - contradiction_risk
  - redundancy_penalty
```

Use idea-index to decide:

- keep in prompt;
- retrieve source evidence;
- retrieve triplet graph;
- compress to summary;
- ask clarification.

## 6. Phase 4 - Context Scheduler Reimplementation

Priority: high.

### 6.1 Replace Ad-Hoc Context Prefix With Structured ContextPlan

Current `ContextPlan` should be extended:

```json
{
  "state": {},
  "actions": [],
  "selected_context": [],
  "rejected_context": [],
  "token_budget": {},
  "evidence_candidates": [],
  "triplet_candidates": [],
  "idea_candidates": [],
  "warnings": [],
  "debug": {}
}
```

### 6.2 State Features

State must include:

- session id;
- turn index;
- current user question;
- current task type;
- active disease/MeSH terms;
- recent sentiment;
- open questions;
- accepted facts;
- disputed facts;
- selected sources;
- available token budget;
- memory retrieval confidence;
- source retrieval confidence;
- triplet retrieval confidence;
- idea-index focus;
- previous reward trend.

### 6.3 Action Types

Actions:

- keep recent turn;
- evict recent turn;
- summarize segment;
- retrieve source sentences;
- retrieve triplets;
- retrieve idea-index;
- retrieve web;
- ask clarification;
- warn about inconsistency;
- use NLI check;
- store correction;
- update landmark.

Every action should be logged:

```json
{
  "action_type": "...",
  "reason": "...",
  "input_ids": [],
  "output_ids": [],
  "token_cost": 0,
  "expected_utility": 0.0,
  "actual_reward": null
}
```

### 6.4 Token Budgeting

Budget categories:

- system instruction;
- user current question;
- recent buffer;
- active landmarks;
- source evidence;
- triplets;
- idea-index;
- correction/disputed facts;
- web snippets;
- response allowance.

Policy:

- source evidence and current user question are highest priority;
- corrections and disputed facts override old memory;
- triplets are compact guides, not evidence replacement;
- web only when local evidence sparse and allowed;
- old low-reward memory should be compressed or left in ES.

## 7. Phase 5 - Reward And Policy Learning Without Weight Updates

Priority: medium-high after factuality/evidence foundation.

### 7.1 Reward Rubric

Reward fields:

```json
{
  "answer_relevance": 0.0,
  "prompt_evidence_support": 0.0,
  "posthoc_evidence_support": 0.0,
  "nli_entailment": 0.0,
  "nli_contradiction": 0.0,
  "unsupported_claim_penalty": 0.0,
  "citation_coverage": 0.0,
  "user_sentiment_delta": 0.0,
  "clarification_quality": 0.0,
  "retrieval_precision": 0.0,
  "retrieval_recall_proxy": 0.0,
  "latency_penalty": 0.0,
  "token_cost_penalty": 0.0,
  "correction_penalty": 0.0,
  "overall": 0.0
}
```

### 7.2 Q-Table / Action-Value Table

Yes, action-value table means Q-table-like structure:

```text
Q(state_features, action) -> expected future reward
```

Do not implement full Q-learning immediately. First collect traces.

After enough traces:

1. Define state buckets:
   - task type;
   - evidence density;
   - memory confidence;
   - contradiction risk;
   - token pressure;
   - user correction history.
2. Define action buckets:
   - retrieve_more_sources;
   - retrieve_triplets;
   - use_web;
   - ask_clarification;
   - compress_context;
   - cite_more;
3. Compute empirical average reward per bucket.
4. Use epsilon-greedy or UCB contextual bandit for policy exploration.
5. Keep human-visible logs.

### 7.3 Transferable Policy Patterns

Store high-value patterns separately:

```json
{
  "doc_type": "policy_pattern",
  "scope": "tenant|global",
  "state_signature": {},
  "action_sequence": [],
  "reward_avg": 0.0,
  "support_count": 0,
  "failure_modes": [],
  "privacy_class": "safe_to_transfer|tenant_only|session_only"
}
```

Cross-tenant transfer must avoid leaking user/session/source-specific content.

## 8. Phase 6 - Compression, Eviction, And Episodic Summaries

Priority: medium.

### 8.1 Segmenting Conversation

Segment by:

- topic shift;
- user goal shift;
- evidence set shift;
- correction/dispute;
- completed task.

### 8.2 Summary Types

- session goal summary;
- evidence summary;
- disputed facts summary;
- accepted facts summary;
- user preference summary;
- policy reflection summary.

### 8.3 Promotion/Demotion Rules

Promote to working prompt:

- current question evidence;
- corrections;
- active disputed facts;
- high-frequency/high-reward ideas;
- open questions.

Demote to ES memory:

- old low-use turns;
- repeated explanations;
- stale retrievals;
- low-reward context.

Compress:

- multi-turn resolved discussions;
- repeated evidence;
- long answer text after claims/evidence extracted.

## 9. Phase 7 - UI/API Features Prioritized By User Value

### 9.1 Evidence Table UI

Highest UI priority.

Show:

- answer claim;
- source sentence;
- citation;
- NLI label;
- support score;
- contradiction warning;
- "confirm correction" action.

### 9.2 Source Control / Research Board

Allow user to:

- select papers/sentences;
- include/exclude sources for current answer;
- save source set as board;
- ask across selected source set.

### 9.3 Citation And Semantic Graph

Integrate:

- citation graph;
- triplet graph;
- idea graph;
- evidence graph.

Do not make graph decorative. It must support concrete workflows:

- find related mechanisms;
- inspect contradictions;
- discover missing evidence;
- navigate from answer claim to source neighborhood.

### 9.4 Correction Workflow

When warning appears:

1. Show claim and conflicting/source evidence.
2. Ask user: "Which fact should be treated as authoritative?"
3. Save answer to correction endpoint.
4. Add correction landmark.
5. Re-answer or continue.

### 9.5 Forks/Threads

Deferred until after the above.

Forks are useful for:

- exploring alternative interpretations;
- resolving contradictions without polluting main state;
- trying answer variants.

But they add state complexity and should be implemented after evidence/corrections are reliable.

## 10. Implementation Order

### Milestone A - Grounded Factuality Core

1. Add claim extraction module.
2. Add source evidence candidate module.
3. Add comparability gate.
4. Rework `nli.py` to score answer claims vs source sentences.
5. Extend reward report.
6. Store evidence tables.
7. Add SSE payloads.
8. Validate with synthetic examples.

Acceptance:

- For an answer with two claims, system returns claim support records.
- Unsupported claim is marked unsupported, not contradicted.
- Contradicted claim produces warning.
- Source sentence provenance is preserved.

Test gate:

- Unit tests for each new module pass before integration.
- Mocked OpenSearch/HF API tests pass.
- HF API smoke test passes when token is available.
- NVIDIA LLM smoke test passes for context-manager prompt if LLM decomposition/gating is used.

### Milestone B - Biomedical NLI Service

1. Add `services/biomedical-nli` or equivalent.
2. First implement/test HF API provider fully.
3. Add Dockerfile for local fallback service only after HF path works.
4. Add CPU-safe config by default.
5. Add model env var.
6. Add `/predict`.
7. Add smoke test fixture.
8. Calibrate thresholds on small hand-labeled set.

Acceptance:

- API can call `memory.nli_provider=hf_api`.
- API can call `memory.nli_provider=http`.
- Service returns entailment/neutral/contradiction scores.
- Does not start GPU unless explicitly configured.

### Milestone C - Idea-Index

1. Add idea schema.
2. Add idea updater after each turn.
3. Add co-occurrence tracking.
4. Add hierarchy fields.
5. Add idea retrieval boost in context policy.
6. Add debug view/payload.

Acceptance:

- Repeated concepts increase frequency.
- Active ideas affect context selection.
- Correction-related ideas get high importance.

### Milestone D - Scheduler Refactor

1. Make context plan structured and auditable.
2. Track selected and rejected context.
3. Add token budget accounting.
4. Add source priority policy.
5. Add policy trace schema migration.

Acceptance:

- Every context item has selection reason and token cost.
- Trace says why evidence/triplet/memory/web was selected or rejected.

### Milestone E - UI/UX Research Features

1. Evidence table UI.
2. Source selection/research board UI.
3. Correction save UI.
4. Graph navigation UI.
5. Memory/debug UI for developers.

Acceptance:

- User can inspect where an answer came from.
- User can correct a fact.
- User can constrain answers to selected sources.

### Milestone F - Policy Learning

1. Export trace/reward replay.
2. Define Q/action-value buckets.
3. Add empirical reward table.
4. Add safe contextual-bandit selection.
5. Add per-tenant/global transfer controls.

Acceptance:

- Policy can report which actions historically improved reward.
- No private content leaks across tenants.

### Milestone G - Forks/Threads

1. Add conversation branch schema.
2. Add fork from warning/dispute.
3. Add merge/correction semantics.
4. Add UI for return to main conversation.

Acceptance:

- User can explore disputed fact in fork.
- Main conversation state remains stable.

## 11. Testing Strategy

### 11.0 No Propagation Rule

Testing must happen bottom-up. A failing or untested lower-level module must not be used as if reliable by a higher-level module. Mark it Partial or Blocked in the status tracker.

### 11.1 No-GPU Tests

- Python syntax/AST checks.
- Unit tests for claim extraction.
- Unit tests for comparability gate.
- Unit tests for NLI aggregation with mocked service.
- Unit tests for idea-index updates.
- `docker compose config`.
- Direct NVIDIA hosted API test only if needed.
- Hugging Face NLI API smoke test when `HF_API_TOKEN` is available.

### 11.2 Integration Tests

Use small fixtures:

- source sentence entails answer claim;
- source sentence contradicts answer claim;
- source sentence is neutral but shares entities;
- source sentence same disease but wrong population;
- answer claim has no support.

Provider order:

1. NVIDIA hosted API for chat/context-manager roles.
2. Hugging Face API for biomedical NLI.
3. Local models only after explicit GPU approval.

For NVIDIA models, test:

- main chat model path;
- context-manager path;
- non-streaming `chat_once`;
- streaming generation path if needed;
- unsupported optional argument fallback;
- empty final `content` when reasoning consumes budget;
- minimum token budget required for non-empty final content.

For HF NLI model, test:

- known entailment pair;
- known contradiction pair;
- neutral pair;
- wrong evidence candidate pair;
- unavailable/missing token fallback.

### 11.3 Calibration Set

Create hand-labeled CSV:

```csv
premise,hypothesis,label,comparability,notes
```

Start with 100 examples from actual PubMed sentence corpus.

Metrics:

- contradiction precision;
- entailment precision;
- unsupported detection;
- false warning rate;
- calibration curve.

## 12. Risks And Guardrails

## 11.4 2026-05-19 Replay Evaluation Notes

Implemented:

- Added deterministic saved-conversation replay using existing memory documents as seeds.
- Added reward/lifecycle comparison between a legacy recent-message buffer and the current memory lifecycle policy.
- Added constrained-token simulations to evaluate compression/eviction/promotion behavior where the lifecycle policy should matter most.

Partially implemented:

- The default-budget replay is non-regressive, not yet better: current policy matched baseline reward/support while using slightly fewer context tokens.
- The constrained-budget replay shows improvement because extractive compression keeps usable context that the legacy recent-buffer baseline drops.
- These simulations are proxy evaluations over saved answers, not full hosted-LLM answer regeneration.

Not implemented:

- Labeled expected-answer evaluation.
- Disease-specific factuality calibration corpus.
- Hosted LLM replay that regenerates answers under each memory policy.

Blocked:

- No current infrastructure blocker. End-to-end answer-quality scoring needs a labeled corpus and slower provider-backed replay.

Tests run:

- Default token budget: reward delta `0.0000`, context support delta `0.0000`, current tokens `293.6` vs baseline `300.8`.
- Token budget 160: reward delta `+0.0540`, context support delta `+0.1355`.
- Token budget 120: reward delta `+0.0391`, context support delta `+0.0683`.
- Token budget 80: reward delta `+0.0341`, context support delta `+0.0475`.
- Hosted no-GPU API test suite: `Ran 64 tests`, `OK`, `skipped=3`.

## 11.5 2026-05-19 Longitudinal Consistency Addition

Implemented:

- Added a conversation-frame memory object that carries active topic terms, avoided/retired terms, user steering, and evidence-supported claims across turns.
- Prior entailed claims from evidence tables are now reused as memory evidence candidates for later turns, allowing the existing claim-support/BioNLI path to compare new answer claims against managed memory.
- Reward now includes frame alignment, frame drift, prior-memory conflict, and longitudinal penalty fields.
- Chat/UI debug can surface the conversation frame and longitudinal consistency warnings.

Partially implemented:

- This is a deterministic first pass. It is intentionally conservative and auditable.
- It improves factual consistency plumbing, but it is not yet a fully learned consistency controller.

Not implemented:

- Higher-confidence semantic drift detection. Planned later as a bounded classifier that compares the active conversation frame, user corrections, and generated answer; it should output drift type, confidence, evidence terms/spans, and a warning/no-warning decision calibrated on labeled examples.
- Backfill old sessions into conversation frames.
- Labeled longitudinal consistency benchmark.
- Full contradiction resolution UX.

Blocked:

- No immediate blocker. Better semantic drift detection needs labeled examples or a bounded hosted-LLM classifier.

Deferred semantic-drift plan:

1. Create a small labeled set from saved conversations with examples of on-topic continuation, scope drift, entity drift, user-correction violation, and acceptable broadening.
2. Add a mocked deterministic interface first: `classify_semantic_drift(frame, question, answer)`.
3. Add an optional hosted context-manager LLM classifier only after the deterministic tests pass; keep it bounded to JSON fields and no chain-of-thought.
4. Feed drift confidence into `longitudinal_consistency`, reward penalties, SSE warnings, and simulation reports.
5. Track false positives separately so the system does not over-warn when answers legitimately broaden the topic.

Tests run:

- Focused hosted tests: `Ran 15 tests`, `OK`.
- Full hosted no-GPU API suite after integration: `Ran 69 tests`, `OK`, `skipped=3`.

## 11.6 2026-05-19 Lung Cancer Evaluation Lab

Implemented:

- Added `evals/lung_factuality_lab/` as a standalone iterative evaluation lab.
- The lab follows this pipeline:
  - scenario specification;
  - synthetic conversation generation;
  - assistant adapter answer generation;
  - claim extraction;
  - evidence/gold-claim and mechanism-graph comparison;
  - decomposed reward scoring;
  - conversation trace writing;
  - failure diagnosis;
  - structured recommendations;
  - regression plan generation.
- Outputs are designed to remain stable across implementation commits, so evaluator modules can change while trace/report contracts remain comparable.
- Added commands:
  - `run_single`;
  - `run_batch`;
  - `compare_runs`;
  - `regression_planner`.

Partially implemented:

- Initial assistant adapters include `dummy` and `http/target_chatbot`.
- The dummy adapter and fixture evaluator are validated; live target-chatbot batch replay should be the next lab step.
- The initial scenarios cover the requested minimal useful batch but are not yet a full benchmark.

Not implemented:

- CI across commits.
- Large labeled scenario corpus.
- Provider-backed semantic drift classifier.
- OpenSearch/source-document retrieval inside the lab.

Blocked:

- No hosted no-GPU blocker.
- Target-chatbot evaluation at scale requires an explicit provider-cost/latency budget.

Tests run:

- Lab unit tests: `Ran 7 tests`, `OK`.
- Local CLI smoke covered single scenario, batch, run comparison, and regression planner.
- Hosted API-image validation via `scripts/compose-hosted.sh run`: `Ran 7 tests`, `OK`.
- Hosted API-image batch smoke wrote `evals/lung_factuality_lab/runs/batch_smoke`.

## 11.7 2026-05-19 Seed Conversations And Generated Variants

Implemented:

- Added seed JSONL conversations as first-class data assets.
- Added scenario `conversation_file` links so scenarios define purpose and conversations define user turns.
- Added user false-premise perturbation bank and assistant wrong-answer bank.
- Added wrong-answer replay adapter to test evaluator/reward failure detection without depending on the live chatbot.
- Added deterministic generated variants through `--variant-index`.

Partially implemented:

- Variant generation is still template-based, not provider-generated.
- Wrong-answer replay covers core failures, not every gold claim.

Not implemented:

- Large generated conversation corpus.
- Target-chatbot seed-conversation batch.
- CI/regression automation across commits.

Blocked:

- No hosted no-GPU blocker.
- Large live replay needs provider-cost/latency budget.

Tests run:

- Local tests: `Ran 11 tests`, `OK`.
- Hosted API-image tests: `Ran 11 tests`, `OK`.
- Local and hosted CLI smoke covered seed loading, wrong-answer replay, generated variants, and batch reports.

Risk: NLI false positives.

- Mitigation: comparability gate, threshold calibration, show evidence, ask user.

Risk: Triplet extraction errors.

- Mitigation: triplets are candidate selectors, not truth.

Risk: Post-hoc evidence laundering.

- Mitigation: distinguish evidence that was in prompt from evidence found after answer.

Risk: NVIDIA token leakage.

- Mitigation: never print `.env`; parse token but do not log it; keep `.env` out of Git.

Risk: GPU contention.

- Mitigation: no local model-loading tests unless user permits.

Risk: Overcomplicated agent loop.

- Mitigation: deterministic first, auditable traces, LLM only where useful and fallback-safe.

Risk: User trust loss due to opaque warnings.

- Mitigation: compact warnings with source sentence and correction button.

## 13. Open Questions For User

1. Should the first real biomedical NLI service run CPU-only initially, even if slower, to avoid GPU contention?
2. Do you want answer factuality checking to block final answers, or only warn and lower reward?
3. Should evidence table be returned by default in the chat SSE stream, or only when `expose_memory_debug=true`?
4. What disease-specific corpus should be used first for calibration examples?
5. Do you want PubMed source sentences indexed separately from triplet source sentences if they are not already in the same index?
6. What tenant/privacy rules should govern future cross-session policy transfer?

## 14. References

- Bolaños F., Salatino A., Osborne F., Motta E. "Artificial intelligence for literature reviews: opportunities and challenges." Artificial Intelligence Review, 2024.
- OECD, "Elicit: Language models as research tools."
- Cohan et al./SciLit authors, "SciLit: A Platform for Joint Scientific Literature Discovery, Summarization and Citation Generation", arXiv:2306.03535.
- Public user discussions from Reddit on AI research tools, PDF citation tools, dynamic RAG, knowledge management, and research-note workflows. Treat as product-signal evidence, not scientific evidence.
