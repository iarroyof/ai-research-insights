# 2026-05-14 Development Status Tracker

This file tracks exact implementation status. Future agents must update it when changing code. Do not delete incomplete items. Do not convert "partial" to "done" unless the behavior is implemented, configured, and minimally validated.

## Status Legend

- Done: implemented and locally/remote syntax or config validated.
- Partial: code exists but does not fully satisfy requirement.
- Scaffold: config or placeholder exists but operational behavior is absent.
- Missing: not implemented.
- Blocked: technically blocked by infrastructure, dependencies, GPU availability, or missing decision.
- Deferred: intentionally later priority.

## Requirement Status Matrix

| Requirement | Status | Current Implementation | Remaining Work |
|---|---:|---|---|
| Pre-generation context manager for `/chat/` | Partial | `ContextPolicy.plan()` runs before LLM generation and prepends memory/triplet/web context. No-selection chat now also invokes an auto-context search planner before prompt construction. | Make policy claim/evidence aware, more token-budget aware, and measurable. Add clearer planner output schema. |
| No-selection chat auto-context | Partial | `/chat/` can answer without selected items by calling `app.memory.search_agent`, generating deterministic biomedical query variants, optionally refining them with the hosted context-manager LLM, and running a structured multilevel search over title, paper/chunk, and sentence/triplet levels. Later levels are expanded with compact feedback terms learned from earlier level hits only when those hits stay anchored to the active refinement frame or provide structured relations. Ambiguous evidence-synthesis queries now expose candidate evidence frames, node/edge puzzle telemetry, and a plain textual opening clarification in the streamed answer when context edges are missing. The current user turn now controls specialized search-frame selection, explicit follow-up turns can reuse prior search-plan frames, and low-value metadata/dialogue/cross-domain case-treatment feedback is filtered before it steers later levels. Retrieved context is used for answer context/citations; search plans now include evidence-assembly and refinement-quality telemetry; runtime reward traces and the live lab score that telemetry; a separate search-strategy action-value bucket is updated; textual search-policy notes are persisted. Streamlit exposes the feature with an auto-search toggle and search-note diagnostics. | Add stronger paper-level metadata coverage across all indices, calibrate local/external result ranking and search rewards in the live lab beyond the evidence-assembly microfits, extend supported-frame continuity beyond explicit referential follow-ups, wire automatic query-vector generation for semantic fallback after low BM25 confidence, add note summarization/compaction, and add browser/e2e UI coverage. |
| Multi-turn chat continuation | Partial | `/chat/` accepts/returns `session_id`, reuses recent session memory, and now passes prior user/assistant turns as native chat role messages before the current prompt. Streamlit stores `chat_session_id`, renders a persistent transcript, sends the session id on follow-up turns, and has a New chat reset. | Add session list/resume/delete endpoints, UI session browser, dedicated history retrieval endpoint, and browser/e2e UI tests. |
| Same model acts as assistant and context manager | Partial | Local provider uses same LLM base/model. | Validate in running stack. Improve prompt separation for policy vs answer generation. |
| NVIDIA hosted context-manager option | Partial | `LLMClient.chat_once(provider="nvidia")`, env/config/Compose pass-through, token installed in blue-demon `.env`. `LLM_CHAT_PROVIDER` now lets main chat streaming use NVIDIA without local LLM serving. | End-to-end project-container client test hung in Docker CLI before a visible container appeared; retest when Docker stable. Add model capability profiles. |
| NVIDIA unexpected-argument fallback | Partial | Retries without optional payload controls for 400/422. | Add structured logging and per-model capability cache. |
| Short/fast working context | Partial | Recent session messages loaded from OpenSearch memory. | Use token-aware recent buffer and pin current-turn context. |
| Long/slower ES memory | Partial | `<tenant>_chat_memory` index stores messages, landmarks, traces, corrections. | Add index templates, migrations, shard/mapping review, retention policy. |
| Term-based search primary | Partial | Memory search uses multi_match fields and boosted `terms`. Triplet search is multi_match. | Add explicit BM25 tuning, analyzers, term statistics, idea frequencies. |
| Higher-confidence semantic drift detection model panel | Planned | A phased plan now prioritizes deterministic scope/evidence gates, SciFact-style structured LLM verification, current MedNLI baseline use, zero-shot drift labels, and later SciFact/BioNLI/NLI4CT model-panel experiments. `reward_shape_registry.yaml` prevents revisiting rejected reward/evaluator shapes. | Implement Phase 1 modules and lab trace fields first; run seeded/generated replay and Stage 1/2 live validation before Phase 3 full benchmark. |
| Semantic/vector search fallback | Partial | `app.search.os_client.os_hybrid_query()` remains BM25-first and now has an optional OpenSearch kNN/vector fallback that activates only when a query vector is supplied and BM25 results are sparse or below a configured score threshold. Vector results are appended after BM25 and marked `retrieval_mode=vector_fallback`. | Add hosted embedding/query-vector generation, index templates for vector fields, confidence calibration, and integration into auto-context only after BM25 confidence is low. |
| DuckDuckGo web search | Partial | DuckDuckGo Instant Answer API called after privacy redaction when enabled and local evidence sparse. Off by default. Streamlit now exposes the user toggle and chat citations disclose returned web context snippets. | Add full result ranking/source trust, privacy policy, per-tenant toggle, cache, and richer UI disclosure. |
| PubMed/PMC, PubTator, and LitSense sparse external retrieval | Partial | When optional sparse external context is enabled, `ContextPolicy.plan()` now privacy-redacts the query, retrieves PubMed abstracts with PMC top-up through NCBI E-utilities, reserves distinct slots for PubTator 3 semantic results and LitSense 2.0 reranked sentences/passages when available, and leaves DuckDuckGo as the final no-biomedical-result fallback. PubMed/PMC, PubTator, and LitSense result normalization preserve PMID/PMCID links for chat citation payloads. | Add rate-limit/cache controls for NCBI calls, richer source ranking and abstract/sentence selection, PubTator relation-search strategies, LitSense ranking calibration, and broader live query coverage. |
| Query redaction | Partial | Regex redacts email, phone, URL, IP, secrets, paths. | Add biomedical PHI/PII patterns, user/org names, configurable blocklist. |
| Triplet retrieval enrichment | Partial | `search_triplets()` is used in context policy. | Use triplets to select source sentence windows and graph-neighborhood evidence. |
| Triplet truth/contradiction checking | Partial | Heuristic same-entity/different-relation/polarity detector. | Demote to candidate-discovery role; use source-sentence NLI for factuality. |
| Origin sentence use | Partial | NLI hook reads `sentence_text`/`origin_sentence` from triplets. New `app.memory.evidence` normalizes prompt, pinned, memory/source, and triplet-linked sentences into evidence candidates with paper/source provenance. | Wire source-sentence retrieval broadly enough that every runtime candidate includes stable provenance from ingestion, not just normalized best-effort fields. |
| Biomedical NLI reward | Partial | `memory/nli.py` supports `hf_api`/heuristic/LLM/http providers; reward has NLI factuality fields. HF API for `pritamdeka/PubMedBERT-MNLI-MedNLI` is the preferred first real provider. `classify_nli_batch()` now batches hosted HF premise/hypothesis pairs through the provider queue/retry budget, and claim support plus answer-triplet scoring use the batch path. `HF_API_TOKEN` is present in blue-demon `.env`; non-secret HF/NLI runtime keys are also set there and passed through Compose/config. | Calibrate thresholds on corpus. Local model is fallback only. |
| Zero-shot classification hosted API | Partial | `app.services.zero_shot.score_labels()` supports `ZERO_SHOT_PROVIDER=hf_api`, uses the HF router with `facebook/bart-large-mnli`, parses both pipeline-style and element-style responses, batches multiple texts per hosted request through `ZERO_SHOT_HF_API_BATCH_SIZE`, retries retryable cold-start/rate-limit/server failures with exponential backoff and `Retry-After` support, records in-process provider metrics, uses shared provider queue/retry-budget controls, and lazy-loads local `transformers` only for `ZERO_SHOT_PROVIDER=local`. Compose passes zero-shot provider/model/API/retry settings. BioNLI pair batching is now implemented separately in `app.memory.nli`. | Add external production metrics export. Local path remains full/local-ML image fallback only. |
| Premise/hypothesis comparability gate | Partial | `app.memory.comparability` provides deterministic pre-NLI gating with mismatch reasons for disease, entity, relation, species/population, cell line, negation, temporality, and weak section context. Fixture tests cover entailment, contradiction-with-negation, neutral/not-comparable, and wrong-evidence skip. | Add biomedical NER/normalization, intervention/outcome role detection, directionality parsing, and optional NVIDIA/LLM JSON gate only for uncertain cases. |
| Atomic answer claim extraction | Partial | `app.memory.claims` splits answer text into candidate sentences, deterministically atomizes simple same-subject compound claims, preserves original answer sentence, extracts lightweight entities/relations/negation/speculation, and marks citation need. | Add biomedical abbreviation coverage, robust clause parsing, and optional LLM JSON atomization with strict fallback. |
| Source sentence vs answer claim NLI | Partial | `app.memory.claim_support` runs NLI only after `app.memory.comparability` marks a source sentence/claim pair comparable, aggregates per-claim statuses, and builds an evidence-table payload. No chat-route dependency was added yet. | Integrate into post-answer chat trace after source retrieval is mature; calibrate thresholds on project corpus; keep triplets as candidate retrieval only. |
| Reward shaping without training | Partial | Reward trace with lexical/context/sentiment/conflict/NLI fields plus claim-support citation coverage, contradiction, unsupported penalties, evidence table debug payload, action key, and state key. | Calibrate weights, uncertainty, and trend summaries against corpus/user feedback. |
| Replay buffer for future learning | Partial | Policy trace docs stored. | Define export schema and Q/bandit training data format. |
| Q-table/action-value table | Partial | `app.memory.action_value` implements a Q-like incremental action-value estimate keyed by deterministic state/action buckets; `MemoryStore.update_action_value()` persists per-session or shared docs. Auto-context search now writes a separate non-lexical search state/action bucket so it can score how to search without keying on exact terms. | Add export/analysis tooling, exploration policy, decay, and UI/debug surfacing. |
| Conversation landmarks | Partial | Current focus, open question, latest reward state, correction landmark. | Add explicit conversation-state schema: goals, constraints, accepted facts, disputed facts, pending tasks. |
| Idea-index / concept-frequency tree | Partial | `app.memory.idea_index` extracts recurring ideas from important terms and selected biomedical phrases, normalizes a small deterministic biomedical synonym set such as PD-1/PDL1/NSCLC, tracks frequency, reward average, co-occurrence, parent/child links, concept paths, synonyms, recency/importance, stores `doc_type=idea`, retrieves idea hits in `ContextPolicy.plan()`, and exposes debug output in API/Streamlit. | Expand biomedical normalization with real MeSH/UMLS-style synonym sources, add robust hierarchy construction beyond deterministic phrase parents, tenant-shared transfer policy, and migration/index-template handling for existing indices. |
| Compression/eviction/promotion | Partial | `app.memory.lifecycle` adds deterministic token estimation, token-aware working-set selection, episodic summary document construction, and promote/working/episodic/evicted state classification. `ContextPolicy.plan()` uses a token-budgeted recent buffer and episodic summaries. `ContextPolicy.observe_turn()` writes per-turn episodic summaries and requests lifecycle state updates. `MemoryStore` persists summary/state/token fields. | Add LLM or stronger biomedical summarizer, rolling multi-turn compression windows, retention/deletion policy, user-visible debug endpoint, and corpus-calibrated promotion/eviction thresholds. Running API process was not restarted in this slice. |
| User correction endpoint | Partial | `POST /chat/memory/correction` stores correction as high-importance landmark. | Add UI, link to conflict/NLI evidence, support user-confirmed fact status. |
| User warning workflow | Partial | SSE `warning` and `consistency_warning` emitted. | UI rendering, "confirm actual fact" prompt, correction save action. |
| Fork/thread workflow | Deferred | None. | Implement after factuality/evidence and correction loop are mature. |
| Shared cross-session/tenant policy memory | Scaffold | `shared_policy_enabled` config only. | Design privacy boundaries and transfer schema. |
| Evidence table | Partial | `app.memory.claim_support.build_evidence_table()` creates an OpenSearch-compatible `doc_type=evidence_table` object with per-claim support/evidence rows. `MemoryStore.add_evidence_table()` persists it. Chat debug SSE can emit compact `evidence_table` payloads after `observe_turn()`. `GET /chat/memory/evidence-tables` retrieves stored tables for diagnostics, and Streamlit exposes active-session diagnostics. | Improve source-sentence coverage. Add index template/migration before production volume. |
| Citation-backed answer quality | Partial | Existing RAG citations preserved; memory metadata appended. | Require every biomedical factual claim to map to source evidence when possible. |
| UI selected-source control | Partial | Backend has pinned item support. Streamlit chat now works with no selected source by default through auto-context search and still preserves pinned-context behavior when items are selected. | Add richer selected source set controls, dynamic RAG scope, debug view, and e2e UI tests. |
| Literature/citation graph | Partial | Triplet graph exists in project; not integrated into chat memory plan. | Integrate citation graph, semantic graph, and evidence graph navigation. |

## Validation Status

Done:

- 2026-05-23 post-deploy Shape7j guard continuation:
  - committed and deployed hosted chat memory/evaluation/UI integration (`b37b048`), stricter grounded-answer policy (`38736d5`), biomedical acronym query expansion (`043b1c0`), and prior-frame reuse for style follow-ups (`1db6e89`);
  - Streamlit GUI refactor does not affect lab experiments directly because the lab calls `/chat/` over HTTP; it was deployed with API/worker updates and exposes the same chat/session/debug options to users;
  - hosted deploy used `scripts/compose-hosted.sh`; GPU-profile services remained off; API health passed after restarts;
  - validation before deploy: hosted API suite `Ran 92 tests`, `OK`, `skipped=3`; hosted lab suite `Ran 32 tests`, `OK`; Streamlit `py_compile` passed;
  - validation after search/follow-up fixes: hosted API suite `Ran 93 tests`, `OK`, `skipped=3`; hosted lab suite `Ran 32 tests`, `OK`; focused search/chat tests `Ran 21 tests`, `OK`;
  - Shape7j Sentinel C was attempted as a full live batch but the all-at-once run blocked before writing files, so live guard execution was switched to scenario-by-scenario runs;
  - first CAF/ECM Sentinel C scenario remains a blocker: baseline post-deploy run failed `6/7` turns with `failure_count=13`; stricter prompt-only run regressed to `7/7` failed turns with `failure_count=18`; acronym search expansion improved one trap but still failed `5/7` turns with `failure_count=16`; prior-frame reuse improved follow-up behavior and reduced `failure_count` to `10` but still failed `6/7` turns;
  - decision: do not promote Shape7j and do not run the rest of Sentinel C until the CAF/ECM guard family is fixed or explicitly accepted as a known holdout.
- 2026-05-22 cross-domain follow-up evidence-assembly continuation:
  - search-frame selection now requires current-turn topic signals, so stale TME search notes do not force later biomedical analogy queries into the TME bridge;
  - explicit follow-up questions can carry prior saved search-plan queries as a bounded prior supported search frame;
  - feedback refinement rejects patient-physician dialogue boilerplate, correspondence/author metadata, and case-treatment feedback for cross-domain analogy/inspired-strategy probes when the hit lacks the target domain;
  - the evidence-assembly answer guidance now prevents named candidate therapies, agents, frameworks, pathways, or experiments from being introduced without supplied support;
  - added the seeded live-lab scenario `cross_domain_fungal_therapy_assembly_001` to the evidence-assembly microfit for the cancer/fungal-infection analogy trace;
  - live microfit `evidence_assembly_live_microfit_shape7d`: average reward `0.9017`, failure count `1`; fungal cross-domain seed cleared with reward `1.0`;
  - later follow-up/prior-frame and hard-clarification guard experiments are recorded as candidate-only: single ambiguous rerun `shape7i` cleared (`avg_reward=1.0`, failure count `0`); two-scenario guarded microfit `shape7j` scored `avg_reward=0.8987`, failure count `1`, slightly below `shape7d`, but inspection showed the runtime behavior is preferable because it suppresses unsupported sparse-evidence elaboration and the remaining failure is an evaluator false positive on a snippet-constraint boundary sentence;
  - decision: keep `shape7j` guarded runtime behavior as the candidate path because its trigger is evidence-edge confidence, not a metabolic/pH-specific rule; do not promote Shape7 until the boundary evaluator fix and stratified guard clear;
  - implemented the boundary evaluator fix for general evidence-assembly meta-constraints such as "constrained by the provided snippets"; saved-answer replay of `shape7j` cleared with `avg_reward=1.0`, failure count `0`;
  - a fresh live endpoint rerun cleared the ambiguous sparse-evidence scenario (`avg_reward=1.0`, failure count `0`); the full two-scenario live rerun through a one-off container was stopped after hanging on the second scenario, and no experiment process remains active;
  - focused hosted no-GPU search/chat tests passed: `Ran 18 tests`, `OK`;
  - final focused hosted no-GPU search/chat tests passed after follow-up and hard-clarification guard changes: `Ran 19 tests`, `OK`;
  - hosted lung factuality lab suite passed after alias-boundary claim-extractor fix: `Ran 31 tests`, `OK`.
- 2026-05-22 candidate-frame evidence-assembly and reward-shaping continuation:
  - auto-context now searches and traces candidate evidence frames, records per-frame result counts and evidence-puzzle node/edge status, and ranks/filters generic feedback terms before later query levels;
  - ambiguous evidence puzzles stream a deterministic plain-text opening clarification prefix inside the answer instead of requiring a UI selector;
  - runtime reward reports penalize unsupported bridge claims when assembled evidence edges are missing or partial;
  - live lab HTTP answers now retain citation/memory-debug SSE metadata, turn scores can include evidence-assembly quality and bridge-safety components, and evidence-boundary/clarification claims are treated as safe lab judgments in evidence-assembly turns;
  - added lab scenario `biomedical_ambiguous_evidence_assembly_001` and `configs/evidence_assembly_microfit.yaml`;
  - live endpoint smoke returned HTTP 200 and opened with the textual clarification prefix while the evidence puzzle reported `edge_support_status=missing`;
  - evidence-assembly microfit reward shaping: initial live Shape7 family fit `avg_reward=0.3891`, `failure_count=6`; saved-answer replay after boundary shaping `avg_reward=0.5048`, `failure_count=4`; live Shape7b `avg_reward=0.5758`, `failure_count=3`; final code-matched Shape7c `avg_reward=0.5666`, `failure_count=5`;
  - Shape7 remains candidate-only in `reward_shape_registry.yaml`; the broader stratified guard is still required before broader Stage 2 work.
- 2026-05-22 general auto-context evidence-assembly continuation:
  - generalized multilevel feedback refinement so unanchored early hits do not inject their vocabulary into later queries; regression coverage uses a broad metabolic/pathway-style prompt with irrelevant questionnaire-title feedback rather than a cancer-specific avoid rule;
  - auto-context plans now expose evidence assembly telemetry: information-need shape, ambiguity bucket, level coverage, distinct-paper count, accepted/rejected feedback counters, and clarification guidance;
  - the live answer prompt receives evidence-assembly guidance to avoid asserting unsupported bridges between retrieved evidence pieces and to ask a focused clarification when the relation remains underspecified;
  - runtime reward reports now include evidence-assembly quality and search-query drift counters so live reward shaping can distinguish useful evidence assembly from noisy refinement;
  - focused hosted no-GPU search/reward/chat tests passed: `Ran 17 tests`, `OK`;
  - broader hosted no-GPU API suite passed: `Ran 83 tests`, `OK`, `skipped=3`.
- 2026-05-22 LitSense sparse external retrieval continuation:
  - added privacy-filtered LitSense 2.0 sentence retrieval with passage top-up alongside the existing PubMed/PMC and PubTator biomedical fallback path;
  - sparse external context merging now reserves a distinct LitSense result slot when PubMed/PubTator/LitSense all have non-duplicate provenance, while DuckDuckGo stays last when biomedical external retrieval returns nothing;
  - LitSense results preserve PMID/PMCID provenance, source granularity, rerank score, section, and annotation payload for downstream context/citation diagnostics;
  - focused hosted no-GPU retrieval and chat coverage passed: `Ran 13 tests`, `OK`;
  - broader hosted no-GPU API suite passed: `Ran 81 tests`, `OK`, `skipped=3`;
  - real hosted-container LitSense smoke returned two current LitSense sentence results with NCBI provenance for `lung cancer tumor microenvironment HGF MET`;
  - after the Shape6 `sentinel_c` log showed artifact copy completion and no guard process remained, API alone was restarted through `scripts/compose-hosted.sh restart api`; `/health` returned `{"status":"ok"}`.
- 2026-05-21 PubMed/PMC and PubTator sparse external retrieval continuation:
  - added privacy-filtered PubMed/PMC retrieval through NCBI E-utilities; PubMed abstracts are primary and PMC XML results top up an underfilled external-context budget;
  - added privacy-filtered PubTator 3 semantic search through `pubtator3-api/search`, normalizing PubTator highlight/entity markup before context selection;
  - sparse external context now preserves PubMed/PMC abstract grounding while reserving room for a distinct PubTator semantic result, with DuckDuckGo kept as the final fallback only when biomedical external retrieval returns no results;
  - added an empty-E-utilities result regression after a live sparse-query smoke exposed an empty XML parse edge case;
  - focused hosted no-GPU retrieval/chat tests passed after that regression: `Ran 10 tests`, `OK`;
  - broader hosted no-GPU API suite passed: `Ran 78 tests`, `OK`, `skipped=3`;
  - real hosted-container smokes returned current PubMed/PMC records and PubTator records for `lung cancer tumor microenvironment`;
  - after `shape5_live_sentinel_b` completed, API was restarted through `scripts/compose-hosted.sh restart api`, `/health` returned `{"status":"ok"}`, and a live `/chat/` smoke with sparse external retrieval returned PubTator 3 context snippets for the pH/parasite/cancer query.
- 2026-05-21 live reward-shaping and DuckDuckGo context-source continuation:
  - future agents are now directed from `docs/START_HERE_FOR_NEXT_AGENT.md` to the lung factuality lab README and `configs/reward_shape_registry.yaml` before continuing live reward shaping;
  - `evals/lung_factuality_lab/README.md` and the detailed plan document the one-conversation diagnosis, saved-answer replay, live family microfit, stratified sentinel guard, protected holdout, and blocked-shape registry ladder;
  - Streamlit chat now exposes a privacy-filtered DuckDuckGo context toggle, and chat citation payloads disclose returned `web_context` snippets when the sparse-memory web path is enabled;
  - focused hosted no-GPU DuckDuckGo/chat tests passed: `Ran 4 tests`, `OK`;
  - broader hosted no-GPU API suite passed after the DuckDuckGo changes: `Ran 72 tests`, `OK`, `skipped=3`;
  - real DuckDuckGo Instant Answer smoke from the hosted image reached the public endpoint: the longer `lung cancer tumor microenvironment` query returned zero sparse snippets, while the simpler `lung cancer` query returned two snippets.
- 2026-05-21 reward-shaping continuation after Shape5 guard completion:
  - `shape5_live_sentinel_b` completed with aggregate average reward `0.3935`, failure count `103`, and all eight guard scenario directories plus aggregate trace/failure/recommendation artifacts copied back;
  - Shape6 citation-scope shaping now treats explicit evidence-transfer guidance as citation-scoping rather than unsupported mechanism prose and skips mechanism-graph completeness for that guidance;
  - focused Shape6 claim-judging regression passed and the full hosted lung factuality lab suite passed: `Ran 29 tests`, `OK`;
  - Shape6 saved-answer replay on the weak citation-drift sentinel trace improved average reward from `0.2190` to `0.3739` and reduced failure count from `18` to `12`;
  - Shape6 live single citation-drift diagnostic improved average reward to `0.5011` with failure count `10`;
  - added `configs/generated_microfit_citation_drift.yaml`; saved Stage 2 citation-drift microfit replay improved average reward from `0.3688` to `0.4874` and reduced failure count from `56` to `47`;
  - the live Shape6 citation-drift family microfit completed with average reward `0.3978` and failure count `50`, improving over its Shape4 Stage 2 subset baseline `0.3688` / `56`;
  - Shape6 `sentinel_c` live guard is now running under `/tmp/run_shape6_sentinel_c.sh`; consult `configs/reward_shape_registry.yaml` before any broader Stage 2 rerun.
- 2026-05-19 lung factuality seeded conversation evaluation/reward-shaping slice was implemented and validated:
  - `evals/lung_factuality_lab` now carries seed conversation scenarios through full simulation, failure boards, recommendations, and regression plans;
  - shaped evaluator logic now recognizes resisted false-premise traps instead of counting them as missed failures;
  - mechanism graph matching now uses biomedical aliases for required nodes such as `MET/c-MET`, `M2-like polarization`, `CD8 T cells`, `immunosuppression`, and `HIF`;
  - claim judging now avoids false HGF/MET inversions from words like `metabolic`, flags cross-cancer direct-proof transfer, and keeps broader scope-drift diagnostics;
  - reward penalties were strengthened for `cross_domain_transfer` (`0.55`), `mechanistic_chain_break` (`0.40`), and `overgeneralization` (`0.20`);
  - batch failure items now preserve `scenario_id`, so recommendations and generated regression tests point to exact seed scenarios;
  - seeded dummy baseline to shaped comparison: average reward improved from `0.3514` to `0.4469` across 8 scenarios (`+0.0955`);
  - final shaped dummy seeded batch: `total_turns=16`, `failed_turns=11`, `failure_count=14`, `missed_injected_traps=0`;
  - final shaped wrong-answer-replay seeded batch: average reward `0.4194`, `total_turns=16`, `failed_turns=11`, `failure_count=13`, `missed_injected_traps=0`.
- 2026-05-19 local lung factuality lab tests after reward shaping:
  - command: `python -m unittest discover -s evals/lung_factuality_lab/tests -p 'test_*.py'`;
  - result: `Ran 15 tests`, `OK`.
- 2026-05-19 hosted API-image lung factuality lab validation:
  - command: `./scripts/compose-hosted.sh run --rm --no-deps -v "$PWD/evals:/app/evals:ro" api python -m unittest discover -s /app/evals/lung_factuality_lab/tests -p "test_*.py"`;
  - result: `Ran 15 tests`, `OK`;
  - hosted seeded dummy smoke command wrote to `evals/lung_factuality_lab/runs/seed_after_dummy_smoke`;
  - hosted smoke result: average reward `0.4469`, `total_turns=16`, `failed_turns=11`, `failure_count=14`, `missed_injected_traps=0`;
  - `scripts/compose-hosted.sh ps` remained up and no GPU-profile services were started.
- 2026-05-19 large generated lung factuality corpus was integrated and benchmarked:
  - imported `lung_factuality_large_corpus_v1.zip` from `C:\Users\nachi\Downloads\lung_factuality_large_corpus_v1.zip`;
  - corpus size: 8 scenario families, 120 generated conversations, 840 user turns;
  - imported generated conversations, generated scenario registry, generated gold claims, generated mechanism graphs, generated user-false-premise bank, generated wrong-answer bank, verification report, and corpus generator script under `evals/lung_factuality_lab/`;
  - added `configs/generated_batch_runs.yaml` for the full generated benchmark;
  - loader now merges generated evidence/scenario/trap files, normalizes generated `required_nodes` into `required_mechanism_nodes`, preserves generated `scenario_id`/`conversation_id`/`variant_index`/`tags`, and derives target mechanism graphs from generated gold-claim links;
  - wrong-answer replay now supports generated list-style banks and base-scenario plus variant matching;
  - shaped evaluator/reward logic for the large corpus:
    - generated traps are synthesized from turn metadata when not present in the explicit trap bank;
    - forbidden-claim matching is stricter and recognizes resisted false premises;
    - generated biomedical aliases were added for MDSC/Treg, HIF/PD-L1, ECM stiffness, CAF heterogeneity, and cross-cancer scope;
    - curated wrong variants are judged before generic scope drift;
    - negated citation-scope language such as `not direct proof` and `transfer hypothesis` is not counted as cross-domain transfer.
- 2026-05-19 local large generated benchmark results:
  - initial after-import dummy benchmark: average reward `0.2235`, `missed_injected_traps=318`;
  - final shaped dummy benchmark: average reward `0.88`, `scenario_count=120`, `total_turns=840`, `failed_turns=120`, `failure_count=120`, `missed_injected_traps=0`, reward delta `+0.6565`;
  - initial after-import generated wrong-answer replay benchmark: average reward `0.1924`, `missed_injected_traps=340`;
  - final shaped generated wrong-answer replay benchmark: average reward `0.74`, `scenario_count=120`, `total_turns=840`, `failed_turns=240`, `failure_count=355`, `missed_injected_traps=0`, reward delta `+0.5476`.
- 2026-05-19 hosted API-image large generated benchmark validation:
  - command: `./scripts/compose-hosted.sh run --rm --no-deps -v "$PWD/evals:/app/evals:ro" api python -m unittest discover -s /app/evals/lung_factuality_lab/tests -p "test_*.py"`;
  - result: `Ran 16 tests`, `OK`;
  - hosted dummy large smoke result: average reward `0.88`, `scenario_count=120`, `total_turns=840`, `failed_turns=120`, `failure_count=120`, `missed_injected_traps=0`;
  - hosted generated wrong-answer replay large smoke result: average reward `0.74`, `scenario_count=120`, `total_turns=840`, `failed_turns=240`, `failure_count=355`, `missed_injected_traps=0`;
  - no containers were restarted and no GPU-profile services were started.
- 2026-05-18 no-selection chat auto-context first slice was implemented and validated:
  - added `app.memory.search_agent` for deterministic biomedical query variants, hosted context-manager LLM refinement, BM25-first ES sentence/triplet retrieval, result deduplication, and non-lexical search state/action keys;
  - `/chat/` now auto-retrieves context when `items=[]` and `allow_auto_context=true`, while leaving selected/pinned context behavior unchanged;
  - auto-context snippets are passed into prompt/citation construction, reward/evidence context, and citation metadata;
  - `ContextPolicy.observe_turn()` records the search plan in traces, updates a separate search-strategy action-value bucket, and persists compact textual search-policy notes;
  - `MemoryStore` now persists and retrieves `search_policy_note` docs;
  - `GET /chat/memory/search-notes` exposes active-session search notes for diagnostics;
  - Streamlit chat now has an auto-search toggle for no-pinned-context chat and includes Search notes in session diagnostics.
- 2026-05-18 no-GPU hosted unittest suite after auto-context work:
  - command: `./scripts/api-host-test.sh -m unittest discover -s tests -p "test_*"`;
  - result: `Ran 48 tests`, `OK`, `skipped=3`.
- 2026-05-18 focused auto-context/multiturn tests passed:
  - command: `./scripts/api-host-test.sh -m unittest tests.test_memory_search_agent tests.test_chat_auto_context tests.test_chat_multiturn`;
  - result before final planner-budget change: `Ran 6 tests`, `OK`;
  - command after final planner-budget change: `./scripts/api-host-test.sh -m unittest tests.test_memory_search_agent tests.test_chat_auto_context`;
  - result: `Ran 5 tests`, `OK`.
- 2026-05-18 live hosted no-selection chat smoke passed:
  - API and Streamlit were restarted individually through `scripts/compose-hosted.sh restart api streamlit`, then API alone after the final parser/budget adjustment;
  - local API `/health` returned HTTP 200 and public ngrok `/health` returned HTTP 200;
  - a live `/chat/` request with `items=[]` and `allow_auto_context=true` streamed answer tokens, emitted final session metadata, and returned auto-context metadata with `used_llm=true`, `result_count=8`, and LLM-refined query labels;
  - `scripts/compose-hosted.sh ps` showed hosted services up and no GPU-profile services running.
- 2026-05-18 semantic/vector fallback first slice was implemented and validated:
  - `os_hybrid_query()` keeps BM25 primary and only appends OpenSearch kNN fallback hits when a caller supplies `filters.query_vector` and BM25 is sparse or below `fallback_min_score`;
  - vector fallback tries configured vector fields such as `embedding`, `vector`, `vec`, and `sentence_vector`, dedupes against BM25 hits, and annotates `retrieval_mode`;
  - no embedding generation has been wired yet, so this is dormant until query vectors and indexed vectors exist.
- 2026-05-18 no-GPU hosted unittest suite after vector fallback work:
  - command: `./scripts/api-host-test.sh -m unittest discover -s tests -p "test_*"`;
  - result: `Ran 50 tests`, `OK`, `skipped=3`.
- 2026-05-18 focused vector fallback tests passed:
  - command: `./scripts/api-host-test.sh -m unittest tests.test_search_vector_fallback`;
  - result: `Ran 2 tests`, `OK`.
- 2026-05-18 post-vector hosted validation:
  - API was restarted individually through `scripts/compose-hosted.sh restart api`;
  - local API `/health` returned HTTP 200;
  - `scripts/compose-hosted.sh ps` showed hosted services up and no GPU-profile services running.
- 2026-05-18 auto-context multilevel structured search was implemented and validated:
  - `app.memory.search_agent` now plans title, paper/chunk, and sentence/triplet search levels instead of only running query variants through one sentence path;
  - title search is intended to find candidate papers and vocabulary, paper/chunk search gathers broader article context, and sentence/triplet search finds exact evidence sentences;
  - compact terms extracted from earlier level results are fed into later level queries and recorded in `level_reports`/`feedback_terms`;
  - `app.search.os_client.os_multilevel_query()` searches level-specific OpenSearch index candidates and fields, with title-heavy, paper/chunk-heavy, and sentence/triplet-heavy field boosts;
  - `/chat/` citation metadata now includes auto-context `levels` and `level_reports`.
- 2026-05-18 no-GPU hosted unittest suite after multilevel search work:
  - command: `./scripts/api-host-test.sh -m unittest discover -s tests -p "test_*"`;
  - result: `Ran 52 tests`, `OK`, `skipped=3`.
- 2026-05-18 focused multilevel search tests passed:
  - command: `./scripts/api-host-test.sh -m unittest tests.test_memory_search_agent tests.test_search_vector_fallback tests.test_chat_auto_context`;
  - result: `Ran 9 tests`, `OK`.
- 2026-05-18 live multilevel no-selection chat smoke passed:
  - API was restarted individually through `scripts/compose-hosted.sh restart api`;
  - local API `/health` returned HTTP 200 and public ngrok `/health` returned HTTP 200;
  - a live `/chat/` request with `items=[]` and `allow_auto_context=true` streamed answer tokens and returned auto-context metadata with `levels=['title','paper','sentence']`, level result counts `title=2`, `paper=2`, `sentence=3`, `result_count=7`, and `used_llm=true`;
  - API logs showed no `Traceback` or `ERROR` lines in the checked tail.
- 2026-05-18 hosted BioNLI pair batching was implemented and validated:
  - added `classify_nli_batch()` and `_hf_api_nli_batch()` in `app.memory.nli`;
  - hosted HF BioNLI now sends multiple premise/hypothesis pairs per request when `NLI_HF_API_BATCH_SIZE` is greater than 1, while preserving the existing `classify_nli()` single-pair API;
  - batch calls still use the shared `hf_biomed_nli` provider queue, shared retry budget, retryable-status handling, `Retry-After`, and provider metrics;
  - `score_answer_triples()` now batches selected evidence/claim pairs;
  - `assess_claim_support()` now uses batched NLI when the default classifier is used, while still honoring injected single-pair or batch test functions;
  - config/default adds `nli_hf_api_batch_size` / `NLI_HF_API_BATCH_SIZE`.
- 2026-05-18 no-GPU hosted unittest suite after BioNLI batching:
  - command: `./scripts/api-host-test.sh -m unittest discover -s tests -p "test_*"`;
  - result: `Ran 55 tests`, `OK`, `skipped=3`.
- 2026-05-18 focused BioNLI batching tests passed:
  - command: `./scripts/api-host-test.sh -m unittest tests.test_memory_nli_batch tests.test_memory_claim_support tests.test_memory_hf_nli_smoke`;
  - result: `Ran 8 tests`, `OK`, `skipped=1`.
- 2026-05-18 real HF BioNLI batch smoke passed from hosted image:
  - two premise/hypothesis pairs were sent through `classify_nli_batch()`;
  - result summary: entailment pair returned entailment about `0.9972`, contradiction pair returned contradiction about `0.9982`;
  - no secrets were printed.
- 2026-05-18 post-BioNLI batching hosted validation:
  - API was restarted individually through `scripts/compose-hosted.sh restart api`;
  - local API `/health` returned HTTP 200;
  - public ngrok `/health` returned HTTP 200;
  - `scripts/compose-hosted.sh ps` showed hosted services up and no GPU-profile services running.
- 2026-05-19 search-agent biomedical term steering was implemented and validated:
  - ambiguous cancer/TME uses of `functional synergy` now create deterministic bridge queries for mechanistic synergy, TME crosstalk, stromal/immune/metabolic cooperation, CAF/TAM/Treg/MDSC, hypoxia, angiogenesis, ECM remodeling, EMT, immune evasion, NSCLC/LUAD/LUSC-style lung cancer terms;
  - mathematical/pharmacological synergy terms such as `combination index`, `CI value`, dose response, drug synergy, CTCAE, adverse events, and toxicity are treated as avoid terms when the user is asking a cancer-biology/TME question rather than a drug-combination question;
  - feedback-term extraction now filters caption/generic noise such as `figure`, `show`, `define`, `study`, and malformed singularizations such as `squamou`/`continuou`;
  - idea normalization no longer strips terminal `s` from words ending in `ous`, preserving terms such as `squamous` and `continuous`;
  - auto-context plans now include `search_frame` and `skipped_off_topic_count` in `/chat/` citation metadata.
- 2026-05-19 no-GPU hosted unittest suite after search steering:
  - command: `./scripts/api-host-test.sh -m unittest discover -s tests -p "test_*"`;
  - result: `Ran 59 tests`, `OK`, `skipped=3`.
- 2026-05-19 focused search steering tests passed:
  - command: `./scripts/api-host-test.sh -m unittest tests.test_memory_search_agent tests.test_memory_idea_action`;
  - result: `Ran 15 tests`, `OK`;
  - command after metadata exposure: `./scripts/api-host-test.sh -m unittest tests.test_chat_auto_context tests.test_memory_search_agent`;
  - result: `Ran 9 tests`, `OK`.
- 2026-05-19 live search-steering smoke:
  - API was restarted individually through `scripts/compose-hosted.sh restart api`;
  - local API `/health` returned HTTP 200 and public ngrok `/health` returned HTTP 200;
  - a live `/chat/` request for `functional synergy` in aggressive lung carcinoma streamed answer tokens and completed with final session metadata.
- 2026-05-16 HF zero-shot hosted request batching was implemented and validated:
  - `_score_labels_hf_api()` now batches multiple texts into one HF router request when `ZERO_SHOT_HF_API_BATCH_SIZE` or `ZERO_SHOT_BATCH_SIZE` is greater than 1;
  - single-text calls preserve the existing string `inputs` payload shape for compatibility;
  - batch response parsing supports a list of per-input pipeline-style responses;
  - forced `ZERO_SHOT_HF_API_BATCH_SIZE=1` still sends one request per text.
- 2026-05-16 no-GPU hosted unittest suite after zero-shot batching:
  - command: `timeout 180 ./scripts/api-host-test.sh -m unittest discover -s tests -p "test_*"`;
  - result: `Ran 43 tests`, `OK`, `skipped=3`.
- 2026-05-16 focused zero-shot/provider queue tests after batching:
  - command: `timeout 120 ./scripts/api-host-test.sh -m unittest tests/test_zero_shot_hf_api.py tests/test_provider_queue.py`;
  - result: `Ran 11 tests`, `OK`.
- 2026-05-16 real HF zero-shot batch smoke passed from hosted image:
  - command shape: two texts, three labels, one batched HF request path;
  - result: `zero-shot-hf-batch-ok [(0.8951, 0.0184), (0.0002, 0.9544)]`.
- 2026-05-16 post-restart hosted validation after zero-shot batching:
  - API was restarted individually through `scripts/compose-hosted.sh restart api`;
  - local API health returned `{"status":"ok"}`;
  - public ngrok `/health` returned HTTP 200 with `{"status":"ok"}`;
  - hosted stack remained up with no GPU-profile services running.
- 2026-05-16 ngrok autorestart was implemented and validated:
  - added `scripts/run-ngrok-tunnel.sh`, a locked per-user supervisor loop for `https://bayleigh-juxtapositional-shirleen.ngrok-free.dev` -> Caddy port `8080`;
  - installed a user crontab `@reboot /home/iarroyof/sabia/ai-research-insights/scripts/run-ngrok-tunnel.sh`;
  - replaced the one-off ngrok process with the supervised runner;
  - killed the ngrok child process once and confirmed the supervisor restarted it with a new child PID;
  - public tunnel health returned HTTP 200 with `{"status":"ok"}` after restart.
- 2026-05-16 HF hosted-provider queue/retry-budget first slice was implemented and validated:
  - added `app.services.provider_queue` with bounded concurrency, queue-timeout metrics, and shared retry-budget accounting;
  - HF zero-shot now runs through the shared provider slot and consumes shared retry budget before retrying retryable failures;
  - HF BioNLI now runs through the async shared provider slot and consumes shared retry budget before retrying retryable failures;
  - queue events and retry-budget exhaustion are recorded through existing provider metrics.
- 2026-05-16 no-GPU hosted unittest suite after provider queue work:
  - command: `timeout 180 ./scripts/api-host-test.sh -m unittest discover -s tests -p "test_*"`;
  - result: `Ran 41 tests`, `OK`, `skipped=3`.
- 2026-05-16 focused provider queue/zero-shot tests passed:
  - command: `timeout 120 ./scripts/api-host-test.sh -m unittest tests/test_provider_queue.py tests/test_zero_shot_hf_api.py`;
  - result: `Ran 9 tests`, `OK`.
- 2026-05-16 post-restart hosted validation after provider queue work:
  - API was restarted individually through `scripts/compose-hosted.sh restart api`;
  - local API health returned `{"status":"ok"}`;
  - public ngrok `/health` returned HTTP 200 with `{"status":"ok"}`;
  - hosted stack remained up with no GPU-profile services running.
- 2026-05-16 idea-index hierarchy/normalization first slice was implemented and validated:
  - `app.memory.idea_index` now normalizes a deterministic biomedical synonym set for concepts including PD-1, PDL1, NSCLC, lung carcinoma/lung cancer, platelet(s), and T cells;
  - idea docs now include `normalized_idea`, `parent_idea`, `child_ideas`, `synonyms`, and `concept_path`;
  - phrase extraction now captures selected biomedical phrases such as `platelet aggregation`, `immune checkpoint`, `checkpoint inhibitor`, `lung cancer`, `PD-1`, `PD-L1`, and `non-small cell lung cancer`;
  - idea ranking now considers normalized ideas, synonyms, concept paths, children, and co-occurrence;
  - OpenSearch mappings/search fields were extended for the new idea-index fields.
- 2026-05-16 no-GPU hosted unittest suite after idea-index hierarchy work:
  - command: `timeout 180 ./scripts/api-host-test.sh -m unittest discover -s tests -p "test_*"`;
  - result: `Ran 38 tests`, `OK`, `skipped=3`.
- 2026-05-16 focused idea/action tests after hierarchy work:
  - command: `timeout 120 ./scripts/api-host-test.sh -m unittest tests/test_memory_idea_action.py`;
  - result: `Ran 6 tests`, `OK`;
  - coverage includes synonym normalization, parent/child edges, and synonym-aware ranking.
- 2026-05-16 post-restart hosted stack validation after idea-index hierarchy work:
  - API was restarted individually through `scripts/compose-hosted.sh restart api`;
  - API health returned `{"status":"ok"}`;
  - `scripts/compose-hosted.sh ps` showed hosted services up and no GPU-profile services running.
- 2026-05-16 memory diagnostics endpoints and UI panel were implemented and validated:
  - `GET /chat/memory/ideas` returns tenant/session-scoped idea-index debug docs;
  - `GET /chat/memory/action-values` returns Q-like action-value telemetry, optionally filtered by `session_id` and `state_key`;
  - `GET /chat/memory/evidence-tables` returns stored evidence-table docs, optionally filtered by `session_id`;
  - Streamlit chat tab now has a Session diagnostics expander for the active `chat_session_id` with Ideas, Action values, and Evidence tables buttons;
  - API and Streamlit services were restarted individually through `scripts/compose-hosted.sh`; GPU-profile services remained off.
- 2026-05-16 no-GPU hosted unittest suite after diagnostics endpoints:
  - command: `timeout 180 ./scripts/api-host-test.sh -m unittest discover -s tests -p "test_*"`;
  - result: `Ran 36 tests`, `OK`, `skipped=3`.
- 2026-05-16 diagnostics endpoint focused tests passed:
  - command: `timeout 120 ./scripts/api-host-test.sh -m unittest tests/test_memory_debug_endpoints.py`;
  - result: `Ran 3 tests`, `OK`.
- 2026-05-16 live diagnostics endpoint smoke passed after API restart:
  - `/chat/memory/ideas`, `/chat/memory/action-values`, and `/chat/memory/evidence-tables` each returned HTTP 200 with compact counts using tenant/API-key headers;
  - API health returned `{"status":"ok"}`;
  - `scripts/compose-hosted.sh ps` showed hosted services up and no GPU-profile services running.
- 2026-05-16 multi-turn chat continuation was implemented and validated:
  - `/chat/` now converts recent selected memory docs from the active `session_id` into native OpenAI-style `user`/`assistant` role messages before the current prompt;
  - `/chat/` emits `native_history_message_count` in memory debug metadata when debug is enabled;
  - Streamlit chat tab now keeps `chat_session_id` in `st.session_state`, renders prior user/assistant messages, sends the session id on follow-up turns, and provides a New chat reset;
  - API and Streamlit services were restarted individually through `scripts/compose-hosted.sh restart api` and `scripts/compose-hosted.sh restart streamlit`; GPU-profile services remained off.
- 2026-05-16 no-GPU hosted unittest suite after multi-turn chat work:
  - command: `timeout 180 ./scripts/api-host-test.sh -m unittest discover -s tests -p "test_*"`;
  - result: `Ran 33 tests`, `OK`, `skipped=3`.
- 2026-05-16 focused multi-turn chat regression passed:
  - command: `timeout 120 ./scripts/api-host-test.sh -m unittest tests/test_chat_multiturn.py`;
  - result: `Ran 1 test`, `OK`;
  - the test stubs the LLM and context policy, sends turn 1, reuses the returned session id on turn 2, and asserts that turn 2 includes turn 1 as native chat messages.
- 2026-05-16 post-restart hosted stack validation after multi-turn UI/API work:
  - API health returned `{"status":"ok"}`;
  - `scripts/compose-hosted.sh ps` showed hosted services up;
  - no GPU-profile services were started.
- 2026-05-16 compression/eviction/promotion first slice was implemented and validated without restarting the hosted stack:
  - added `app.memory.lifecycle` with token estimation, budgeted working-set selection, episodic summary construction, and memory-state classification;
  - `MemoryStore.recent_messages()` can now enforce a token budget while preserving high-value evidence-supported or landmark-like context;
  - `MemoryStore.add_episodic_summary()` persists deterministic per-turn summaries;
  - `MemoryStore.update_memory_lifecycle()` updates `memory_state`, state reason, and token counts for recent memory docs;
  - `ContextPolicy.plan()` includes token-aware recent context plus recent episodic summaries and records working-buffer diagnostics in plan metadata;
  - `ContextPolicy.observe_turn()` creates an episodic summary after reward/claim support assessment and triggers memory lifecycle updates;
  - this is still partial: no LLM summarizer, no retention/deletion scheduler, no dedicated lifecycle diagnostics endpoint, and thresholds are not corpus-calibrated.
- 2026-05-16 no-GPU hosted unittest suite after memory lifecycle work:
  - command: `timeout 180 ./scripts/api-host-test.sh -m unittest discover -s tests -p "test_*"`;
  - result: `Ran 32 tests`, `OK`, `skipped=3`.
- 2026-05-16 focused lifecycle/policy tests passed:
  - command: `timeout 120 ./scripts/api-host-test.sh -m unittest tests/test_memory_lifecycle.py tests/test_memory_policy_observe.py`;
  - result: `Ran 4 tests`, `OK`.
- 2026-05-16 syntax/config validation after lifecycle work:
  - AST parse passed for changed Python files using `PYTHONDONTWRITEBYTECODE=1`;
  - hosted Compose helper service graph validation passed with `scripts/compose-hosted.sh config --services`;
  - hosted service graph still excludes `llm`, `models-init`, `rebel-extractor`, and `worker-gpu`.
- 2026-05-15 hosted Compose stack was brought up and validated after splitting CPU worker dependencies:
  - `worker-cpu` no longer builds from full `services/api/requirements.txt`; `services/worker/Dockerfile` defaults to `services/api/requirements.hosted.txt`;
  - `worker-gpu` keeps full local-ML requirements through explicit GPU-profile build args;
  - `PYTHONPATH=/app` was added for worker services so `app.tasks.celery_app` imports correctly;
  - `worker-cpu` is running and connected to Redis on `cpu.default` and `cpu.ingest`;
  - hosted helper scripts now set `COMPOSE_FILE=docker-compose.yml` so `docker-compose.override.yml` does not silently add stale host-port mappings.
- 2026-05-15 live hosted stack status:
  - `ai-research-insights-api-1` up on `18081`;
  - `ai-research-insights-caddy-1` up on `8080`/`8443`;
  - `ai-research-insights-opensearch-1` up on `19200`;
  - `ai-research-insights-neo4j-1` up on `7474`/`7687`;
  - `ai-research-insights-redis-1`, `postgres-1`, `minio-1`, `grobid-1`, `corenlp-1`, `corenlp-adapter-1`, `streamlit-1`, and `worker-cpu-1` up;
  - no GPU-profile services are running.
- 2026-05-15 live API validations:
  - `GET http://127.0.0.1:18081/` and `GET http://192.168.241.149:18081/` return HTTP 200 with app metadata instead of Internal Server Error;
  - `GET /favicon.ico` returns HTTP 204;
  - `GET http://127.0.0.1:18081/health` returned `{"status":"ok"}`;
  - missing `X-Tenant-Id` on a protected route now returns JSON HTTP 400 `{"detail":"Missing X-Tenant-Id"}` instead of an unhandled middleware exception/HTTP 500;
  - authenticated `GET /chat/memory/provider-metrics` returned `{"tenant":"default","metrics":{}}` in the fresh API process;
  - one live SSE `/chat/` request using NVIDIA hosted chat completed and emitted `memory_debug`, tokens, citations, reward, evidence table, and final events.
- 2026-05-15 provider retry/metrics hardening:
  - HF zero-shot and HF BioNLI record compact in-process provider metrics through `app.services.provider_metrics`;
  - `GET /chat/memory/provider-metrics` exposes those counters without prompts, responses, or secrets;
  - HF BioNLI retry/backoff now mirrors zero-shot behavior for retryable status codes, timeout/network failures, and `Retry-After`.
- 2026-05-15 latest hosted API images after live Compose validation:
  - `ai-research-insights-api:hosted-verify` and `ai-research-insights-api:latest` both point to image ID `cabc9f2d83cb`;
  - image size remains `464MB`.
- 2026-05-15 no-GPU unittest suite after idea/action/factuality/provider hardening:
  - `Ran 29 tests` after adding public-route and tenant-middleware regression coverage;
  - `OK`;
  - `skipped=3`.
- 2026-05-15 real hosted-provider smokes after provider metrics:
  - HF zero-shot returned `biomedical=0.8951`, `finance=0.0184` and recorded `hf_zero_shot` metrics;
  - HF BioNLI returned `entailment=0.9991` and recorded `hf_biomed_nli` metrics.
- 2026-05-15 idea-index and action-value modules were added and wired:
  - `app.memory.idea_index`;
  - `app.memory.action_value`;
  - `ContextPolicy.plan()` retrieves and renders high-value recurring ideas;
  - `ContextPolicy.observe_turn()` updates idea docs and action-value docs from reward traces.
- 2026-05-15 source-sentence factuality wiring was added:
  - `ContextPolicy.observe_turn()` extracts answer claims, gathers evidence candidates from selected/pinned/source/triplet context, runs claim support, builds/stores evidence tables, and includes debug payloads in traces;
  - chat debug SSE emits `evidence_table`;
  - claim contradiction support emits a controlled `consistency_warning`.
- 2026-05-15 service-isolation hardening was added and validated:
  - project published ports are configurable through env vars: `CADDY_HTTP_PORT`, `CADDY_HTTPS_PORT`, `API_HOST_PORT`, `OPENSEARCH_HOST_PORT`, `NEO4J_HTTP_PORT`, `NEO4J_BOLT_PORT`;
  - current running unrelated stack uses host ports `5432`, `6379`, and `9200`; AI Research Insights defaults use `8080`, `8443`, `18081`, `19200`, `7474`, and `7687`;
  - `scripts/check-compose-isolation.sh` was added as a read-only preflight that detects host-port conflicts and foreign containers attached to `ai_research_insights_net`;
  - preflight result: `isolation-check-ok network=ai_research_insights_net ports=8080 8443 18081 19200 7474 7687`.
- 2026-05-15 HF zero-shot retry/backoff was implemented and validated:
  - retryable status codes: `408`, `409`, `425`, `429`, `500`, `502`, `503`, `504`;
  - timeout/network errors retry up to `ZERO_SHOT_HF_API_MAX_RETRIES`;
  - backoff is configurable through `ZERO_SHOT_HF_API_RETRY_BACKOFF_SEC`;
  - `Retry-After` response header is honored when present;
  - no container/service restart is required for the code path, only the API image/container that uses it must be rebuilt/recreated when deployed.
- 2026-05-15 latest hosted image after retry/isolation work:
  - `ai-research-insights-api:hosted-verify` and `ai-research-insights-api:latest` both point to image ID `846db5575c0c`;
  - image size is `464MB`;
  - hosted build context was `19.84kB` in the final rebuild.
- 2026-05-15 no-GPU project tests passed after retry/isolation work:
  - `Ran 20 tests`;
  - `OK`;
  - `skipped=3`.
- 2026-05-15 real HF zero-shot retry-path smoke passed from hosted image:
  - `zero-shot-hf-retry-path-ok [('biomedical', 0.8951265215873718), ('finance', 0.0184012558311224)]`.
- 2026-05-15 latest hosted/no-GPU verification passed after zero-shot HF API integration:
  - `ai-research-insights-api:hosted-verify` rebuilt successfully from `services/api/Dockerfile.hosted`;
  - `ai-research-insights-api:latest` was promoted to the same image ID, `d688b4b607c5`;
  - image size is `464MB`;
  - hosted build context remained small (`16.96kB` in the final cached rebuild).
- 2026-05-15 zero-shot classification hosted path was implemented and validated:
  - `app.services.zero_shot` now supports `ZERO_SHOT_PROVIDER=hf_api`;
  - local `transformers` zero-shot loading is lazy and remains a full/local-ML image fallback only;
  - HF response parsing supports both `{"labels": [...], "scores": [...]}` and `[{"label": "...", "score": ...}]` shapes;
  - unit tests were added for HF payload construction, response mapping, element-style response parsing, nested element-style response parsing, and local fallback dispatch;
  - real HF router smoke from the hosted image passed: `zero-shot-hf-ok [('biomedical', 0.8951265215873718), ('finance', 0.0184012558311224)]`.
- 2026-05-15 no-GPU project tests passed against the promoted hosted image:
  - command: `./scripts/api-host-test.sh -m unittest discover -s tests -p "test_*"`;
  - result: `Ran 18 tests`, `OK`, `skipped=3`.
- 2026-05-15 Compose-hosted one-shot API import passed with current settings attribute names:
  - `compose-hosted-api-ok nvidia hf_api`.
- 2026-05-15 isolated hosted API container health check passed without host networking:
  - `api-health 200 {'status': 'ok'}`.
- 2026-05-15 unrelated containers were checked after the hosted build/test/smoke work and were still running:
  - `set-attention` up;
  - `webpunch-ai-feedback-insights-queue_worker-1` up;
  - `webpunch-ai-feedback-insights-redis-1` healthy;
  - `webpunch-ai-feedback-insights-postgres-1` up;
  - `webpunch-ai-feedback-insights-opensearch-1` healthy.
- 2026-05-15 remote repo target was re-confirmed as the authoritative implementation checkout:
  - `/home/iarroyof/sabia/ai-research-insights`;
  - local `/mnt/d/UserFolders/Documents/GitHub/ai-research-insights-doc-artifacts/source_snapshot/` is only the artifact mirror.
- 2026-05-15 Docker daemon is active and the unrelated `webpunch` stack is running after the earlier Docker restart:
  - `webpunch-ai-feedback-insights-app-1`;
  - `webpunch-ai-feedback-insights-queue_worker-1`;
  - `webpunch-ai-feedback-insights-redis-1`;
  - `webpunch-ai-feedback-insights-postgres-1`;
  - `webpunch-ai-feedback-insights-opensearch-1`.
- 2026-05-15 remote `docker compose config` still passes without starting services.
- 2026-05-15 hosted Compose helper service graph still excludes GPU/local model-serving services:
  - included: `api`, `worker-cpu`, `opensearch`, `postgres`, `redis`, `neo4j`, `minio`, `corenlp`, `corenlp-adapter`, `grobid`, `streamlit`, `caddy`;
  - excluded from default hosted mode: `llm`, `models-init`, `rebel-extractor`, `worker-gpu`.
- 2026-05-15 local-GPU Compose helper service graph still includes GPU services only when the explicit `gpu` profile helper is used.
- 2026-05-15 currently running containers were checked after the timed Docker probe; only the unrelated `webpunch` stack was running.
- 2026-05-15 approved Docker maintenance was performed without host reboot and without starting AI Research Insights GPU services:
  - removed stopped stale `ai-research-insights-*` containers from three months ago;
  - removed unused stale AI Research Insights Docker networks;
  - restarted/reset Docker and containerd when Docker entered a defunct `dockerd`/stuck runtime state;
  - restored unrelated containers that were running before maintenance.
- 2026-05-15 blue-demon Docker default runtime was changed from `nvidia` to `runc`; hosted/no-GPU AI Research Insights paths also force `runc` explicitly:
  - `docker-compose.yml` sets `runtime: ${DOCKER_CPU_RUNTIME:-runc}` for non-GPU services;
  - GPU-profile services use `runtime: ${DOCKER_GPU_RUNTIME:-nvidia}`;
  - `scripts/api-host-test.sh` now runs Docker with `--runtime "${DOCKER_CPU_RUNTIME:-runc}"`;
  - `scripts/compose-hosted.sh` and `scripts/compose-local-gpu.sh` export explicit runtime defaults.
- 2026-05-15 small non-GPU Docker runtime probes pass when forcing `runc`:
  - `docker run --rm --runtime runc --network none redis:7-alpine redis-cli --version`;
  - `docker run --rm --runtime runc --network host redis:7-alpine redis-cli --version`;
  - `docker run --rm --runtime runc redis:7-alpine redis-cli --version`.
- 2026-05-15 hosted Compose config validates after the runtime patch; generated config shows non-GPU services using `runtime: runc`.
- 2026-05-15 hosted Compose service graph still excludes GPU/local model-serving services after the runtime patch.
- 2026-05-15 unrelated containers were restored and rechecked:
  - `set-attention` is up;
  - `webpunch-ai-feedback-insights-app-1` is up;
  - `webpunch-ai-feedback-insights-redis-1` recovered to healthy;
  - `webpunch-ai-feedback-insights-opensearch-1` recovered to healthy;
  - `webpunch-ai-feedback-insights-postgres-1` is up.
- 2026-05-15 hosted/no-GPU API build path was simplified and verified:
  - Docker daemon default runtime was changed from `nvidia` to `runc`; the `nvidia` runtime remains available explicitly for GPU-profile services;
  - `.dockerignore` now excludes logs/artifacts/caches so the hosted build context is small;
  - `services/api/Dockerfile.hosted` was added as the default hosted API Dockerfile;
  - `services/api/requirements.hosted.txt` was added and excludes torch, transformers, sentence-transformers, spaCy, SciSpaCy, and downloaded local model packages;
  - hosted Compose defaults to `API_DOCKERFILE=services/api/Dockerfile.hosted`;
  - local-GPU Compose defaults to `API_DOCKERFILE=services/api/Dockerfile`.
- 2026-05-15 `app.services.zero_shot` now lazy-loads `transformers`; hosted API startup no longer requires local zero-shot ML dependencies.
- 2026-05-15 clean hosted/no-GPU build passed:
  - command: `docker build --no-cache --progress=plain -t ai-research-insights-api:hosted-verify -f services/api/Dockerfile.hosted .`;
  - build context for the hosted Dockerfile was small (`7.57kB` in the verified run);
  - resulting image was promoted to `ai-research-insights-api:latest`;
  - verified image size: `464MB`.
- 2026-05-15 hosted/no-GPU API image smoke passed:
  - `docker run --rm --runtime runc --network none --env-file .env -e APP_CONFIG=/app/config/default.yaml --entrypoint python ai-research-insights-api:hosted-verify -c "import app.main; ..."` returned `hosted-image-import-ok nvidia hf_api`.
- 2026-05-15 default no-GPU API test helper passed against the promoted hosted image:
  - `Ran 14 tests`;
  - `OK`;
  - `skipped=3`.
- 2026-05-15 hosted Compose one-shot API container passed:
  - `./scripts/compose-hosted.sh run --rm --no-deps api python -c "from app.config import settings; import app.main; ..."` returned `compose-hosted-api-ok nvidia hf_api`.
- 2026-05-15 unrelated containers were checked after build/test verification:
  - `set-attention` up;
  - `webpunch-ai-feedback-insights-opensearch-1` healthy;
  - `webpunch-ai-feedback-insights-postgres-1` up;
  - `webpunch-ai-feedback-insights-queue_worker-1` up;
  - `webpunch-ai-feedback-insights-redis-1` healthy.
- Local Python syntax checks passed for changed Python files.
- Remote AST syntax checks passed.
- Remote `docker compose config` passed.
- Direct NVIDIA hosted API test returned HTTP 200 and content when `max_tokens=512`.
- Remote `.env` was updated with NVIDIA env vars without printing the key.
- Compose config contains NVIDIA env var names for API service.
- Local HF token file location is known: `/mnt/d/UserFolders/Documents/hf_huggingface_token.txt`.
- Compose config contains `NLI_PROVIDER` and `HF_API_TOKEN` names for API service.
- HF API token/model smoke test passed on blue-demon using `https://router.huggingface.co/hf-inference/models/pritamdeka/PubMedBERT-MNLI-MedNLI`; SEP-string input is required for the hosted text-classification pipeline.
- HF sanity fixtures returned expected dominant labels:
  - entailment case: entailment `0.9987`;
  - contradiction case: contradiction `0.9978`.
- Local mirror Python syntax checks passed for new factuality modules:
  - `app.memory.claims`;
  - `app.memory.evidence`;
  - `app.memory.comparability`;
  - `app.memory.claim_support`;
  - updated `app.memory.store`.
- Remote Python syntax checks passed for new factuality modules and tests.
- Remote non-GPU unittest discovery passed for memory factuality tests:
  - `Ran 12 tests`;
  - `OK`;
  - `skipped=1` for the opt-in HF smoke unittest when token env was not exported to that test process.
- Remote direct HF API smoke test passed without local model loading:
  - entailment fixture returned label `entailment` score `0.9987`;
  - contradiction fixture returned label `contradiction` score `0.9978`.
- Remote `docker compose config` passed after the new files were added; no services were started.
- Blue-demon `.env` now has the hosted-provider runtime keys:
  - `LLM_CHAT_PROVIDER=nvidia`;
  - `CONTEXT_MANAGER_PROVIDER=nvidia`;
  - `NLI_PROVIDER=hf_api`;
  - `NLI_MODEL`;
  - `HF_API_BASE_URL`;
  - `HF_API_TIMEOUT_SEC`;
  - `NLI_MIN_ENTAILMENT`;
  - `NLI_CONTRADICTION_THRESHOLD`;
  - `NVIDIA_MAX_TOKENS`.
- `HF_API_TOKEN` is present in blue-demon `.env` with a valid `hf_...` shape; the token value was not printed.
- `config/default.yaml` now reads HF/NLI and hosted LLM settings from env placeholders instead of fixed YAML literals.
- `docker-compose.yml` now passes the needed HF/NLI/LLM env vars into the API service.
- GPU/local-model services are now behind the explicit Compose profile `gpu`:
  - `llm`;
  - `models-init`;
  - `rebel-extractor`;
  - `worker-gpu`.
- The default API service dependency list no longer includes `llm`, `models-init`, `rebel-extractor`, or `worker-gpu`.
- Added helper scripts:
  - `scripts/compose-hosted.sh` for hosted NVIDIA/HF mode with no GPU profile;
  - `scripts/compose-local-gpu.sh` for explicit local GPU mode.
- Hosted Compose helper config validation passed; default service set excludes GPU/local model-serving services.
- Local-GPU helper config validation passed; `--profile gpu` includes `llm`, `models-init`, `rebel-extractor`, and `worker-gpu`.
- Sanitized env/config validation passed without printing secrets.
- Direct NVIDIA hosted chat smoke test passed from blue-demon `.env` without local model serving.
- Direct HF hosted NLI smoke test passed from blue-demon `.env` after env-key integration:
  - entailment fixture returned label `entailment` score `0.9987`;
  - contradiction fixture returned label `contradiction` score `0.9978`.

Blocked/Not completed:

- Full long-lived hosted `docker compose up -d --build` validation was run on 2026-05-15 after the user allowed this project stack to be started. GPU-profile services were not started.
- Full API runtime test was not run because GPU services are in use and should not be started.
- Docker runtime was previously infrastructure-risky after the earlier daemon restart, but the hosted/no-GPU API image path now builds and runs with explicit `runc`. Continue using `timeout` and do not restart Docker/containerd, prune networks, stop unrelated containers, or reboot without explicit user approval.
- Resolved 2026-05-15: previous `ai-research-insights-api:latest` image operations blocked before useful output because the hosted/no-GPU image path was still using the full local-ML dependency set and Docker default runtime was `nvidia`. The hosted path now builds and runs through `Dockerfile.hosted` with `runc`.
- Previous blocked state, retained for audit:
  - `./scripts/api-host-test.sh -m unittest discover -s tests -p "test_*"` timed out with no output;
  - `./scripts/compose-hosted.sh run --rm --no-deps api python -c ...` created `ai_research_insights_net` but timed out before the API command ran;
  - `./scripts/compose-hosted.sh build api` blocked at Docker CLI before BuildKit output and was stopped;
  - no Codex/API test containers or client processes were left running afterward.
- Remaining caveat: full/local-ML API image still uses `services/api/Dockerfile` and `services/api/requirements.txt`, which install torch/local ML dependencies and should be used only for local-ML/GPU work or after a separate CPU/GPU dependency split.
- 2026-05-15 unrelated `webpunch-ai-feedback-insights-queue_worker-1` remained in `Restarting (1)` after Docker maintenance. A targeted `timeout 60 docker restart webpunch-ai-feedback-insights-queue_worker-1` also timed out. Other monitored `webpunch` services and `set-attention` were up, with Redis/OpenSearch healthy. This is unrelated to AI Research Insights but must be disclosed because the maintenance interrupted Docker.
- 2026-05-15 `timeout 90 ./scripts/api-host-test.sh -m unittest discover -s tests -p "test_*"` produced no test output and timed out with exit code 124.
- 2026-05-15 five stale Codex smoke-test containers were found in `Created` state:
  - `codex-no-rm-smoke`;
  - `codex-rm-smoke`;
  - `codex-debug-create-smoke`;
  - `codex-api-create-smoke`;
  - `codex-docker-create-smoke`.
- 2026-05-15 a narrow cleanup using `timeout 45 docker rm` for only those `codex-*` smoke containers completed, but only after a noticeable delay. This reduces clutter but does not prove Docker runtime is healthy.
- Earlier Docker symptoms that must not be ignored:
  - bridge-network container creation hung before container creation;
  - `docker system df` hung;
  - `docker compose run --rm --no-deps api ...` executed the command once but hung during cleanup/removal;
  - later tiny `docker run` checks timed out again;
  - Docker logs showed stale sandbox/endpoint cleanup warnings and endpoint delete retries.
- Do not rerun Docker daemon restarts, containerd restarts, Docker network pruning, or host reboot without explicit user approval because live-restore is disabled and unrelated running containers can be interrupted.
- Project-container NVIDIA client test via `docker run` hung in Docker CLI before visible container output. Direct HTTPS succeeded, so issue is not the NVIDIA endpoint.
- Real local biomedical NLI service is not deployed; local model is fallback only.
- HF/API provider retry metrics are in-process only; they reset on API restart and are not yet exported to Prometheus/OpenTelemetry.
- HF batching/rate-limit queueing is not implemented.
- Idea-index is frequency/co-occurrence/reward based, but not yet a true hierarchical concept tree or MeSH-normalized biomedical ontology.
- Action-value memory is a Q-like incremental estimate table, but not yet a learned RL policy, exploration strategy, or offline trainer.
- Compression/eviction/promotion remains mostly missing; recent buffer, memory retrieval, idea retrieval, and evidence tracing are now wired, but there is no summarizer-driven demotion loop yet.
- Semantic/vector fallback remains missing for local ES retrieval.
- Fork/thread contradiction workflow remains deferred.
- Host Python on blue-demon lacks `httpx` and `pydantic_settings`, so real-provider smoke was run with stdlib HTTPS instead of importing the full API stack.
- `docker compose run --no-deps --rm api ...` and direct `docker run --rm ... ai-research-insights-api ...` both hung before a visible container appeared in `docker ps`; the stuck CLI processes were stopped. No GPU/model-serving container was observed starting during these attempts.

## New Factuality Core Failure Modes Documented 2026-05-14

- `app.memory.claims`: deterministic sentence splitting may still split incorrectly on uncommon biomedical abbreviations; compound atomization only handles simple same-subject conjunctions; entity/relation extraction is lightweight and not a biomedical NER substitute.
- `app.memory.evidence`: provenance is preserved when present, but missing ingestion metadata cannot be invented; `window_text` falls back to sentence text if no window is supplied; duplicate detection is hash-based over available fields.
- `app.memory.comparability`: deterministic gates can under-match synonyms and normalized biomedical entities; negation mismatch is detected but does not block NLI when the proposition is otherwise comparable so contradiction fixtures can be checked; intervention/outcome and causal directionality remain heuristic.
- `app.memory.claim_support`: unsupported is returned when a citation-requiring claim has no comparable source sentence; NLI is skipped for non-comparable pairs; provider failures will propagate unless the caller supplies a fallback `nli_func` or uses `app.memory.nli.classify_nli` provider fallback.
- Evidence table trace: schema is backend/debug ready but not yet wired into chat SSE or UI by default; persistence uses the existing chat memory index and should get an explicit index template before production volume.

## Next Highest-Priority Work Items

1. Reimplement factuality reward around answer atomic claims vs source/database sentences, not triplet-vs-triplet.
2. Add premise/hypothesis comparability gate.
3. Add evidence table trace schema and response payload.
4. Implement idea-index aggregate docs.
5. Add UI/debug surfaces for warnings, evidence, and corrections.
6. Add compression/eviction/promotion.
7. Add contextual-bandit/Q-table after enough traces exist.
8. Add fork/thread contradiction workflow last.

## Mandatory Module Test Gates

Future agents must enforce this rule:

```text
No higher-level feature may depend on a newly implemented module until that module has:
1. unit tests,
2. mocked integration tests,
3. one real-provider smoke test where applicable,
4. documented failure modes,
5. this status tracker updated.
```

This is required to avoid propagating technical debt, hidden API mismatches, and logical errors into higher-level chat behavior.

### Provider Test Order

For LLM-dependent modules:

1. Test with NVIDIA hosted API first.
2. Test both roles:
   - main chat model path;
   - context-manager/policy model path.
3. Record model capability profile:
   - supports `max_tokens`;
   - supports `temperature`;
   - supports or rejects `reasoning_effort`;
   - supports or rejects `extra_body`;
   - returns final answer in `message.content` or spends budget in `reasoning_content`;
   - minimum token budget needed for non-empty final content.
4. Only test local models after the user explicitly permits local GPU use.
5. Compare local model output shape, latency, streaming behavior, and empty-content edge cases against the NVIDIA path.

For biomedical NLI:

1. Preferred first real provider: Hugging Face Inference API for `pritamdeka/PubMedBERT-MNLI-MedNLI`.
2. Local model is fallback, not primary, until GPU/CPU deployment is explicitly approved.
3. HTTP/local NLI service is useful later for throughput/cost/privacy, but it must pass the same fixtures as HF API.
4. Heuristic NLI is only a fallback for development and must not be treated as reliable factuality.

### Required Tests By Module

| Module | Required Unit Tests | Required Integration/Smoke Tests |
|---|---|---|
| LLM client | payload building, key cleaning, fallback after 400/422, empty-content handling | NVIDIA hosted API with main model and context model; later local model |
| HF NLI client | token cleaning, label mapping, flat/nested response parsing, 400/422 fallback format | HF Inference API call against `pritamdeka/PubMedBERT-MNLI-MedNLI` when token is available |
| Claim extraction | sentence splitting, compound claims, non-factual text filtering | answer paragraph fixture to atomic claims |
| Evidence candidate selection | source ranking, prompt vs posthoc flag, provenance preservation | fixture source index or mocked OpenSearch |
| Comparability gate | disease/entity/relation/population/negation mismatches | fixtures where NLI must be skipped |
| NLI aggregation | entailment/contradiction/neutral aggregation, unsupported handling | mocked HF/NLI service |
| Reward aggregation | weights, contradiction penalty, unsupported penalty, citation coverage | full claim/evidence fixture |
| Idea-index | frequency, recency, co-occurrence, hierarchy update | multi-turn fixture |
| Context scheduler | token budget, selected/rejected context, action logging | mocked memory/evidence retrieval |
| Correction endpoint | validation, storage, landmark update | API-level test with mocked OpenSearch |

## Update 2026-05-19: Factuality Reward Calibration And Memory Lifecycle Simulation

Implemented:

- Added calibrated reward signals in `app.memory.rewards`:
  - `domain_alignment` for question/answer/context biomedical term alignment.
  - `off_topic_penalty` for mechanistic cancer/TME questions drifting into mathematical/pharmacological synergy terms.
  - `context_support_score`, which scores both global context support and the best supporting item so adding episodic context does not falsely reduce support when the exact evidence is still present.
- Added memory lifecycle improvements in `app.memory.lifecycle`:
  - query-aware `memory_priority()`;
  - query-aware `select_working_set(..., query_text=...)`;
  - extractive compression fallback when a memory item exceeds the remaining token budget;
  - preservation priority for landmarks, corrections, and evidence-supported facts.
- Wired the live context policy to pass the current user query into recent working-set selection.
- Added deterministic replay harness:
  - `app.memory.evaluation`;
  - `tools/simulate_memory_quality.py`.
- Added tests in `tests/test_memory_evaluation.py` for reward off-topic calibration, best-item context support, compression, lifecycle priority, and simulation reporting.

Partially implemented:

- Compression/eviction/promotion is now functional for working-buffer selection and constrained-token replay, but there is still no LLM summarizer-driven demotion loop and no background compaction scheduler.
- Reward calibration is improved for mechanistic-vs-pharmacological drift and context support, but it is still a proxy scorer until a hand-labeled disease-specific calibration set is available.
- Saved-conversation simulation is deterministic and uses saved assistant answers. It measures whether memory selection gives better support for answers; it does not regenerate answers with the hosted LLM.

Not implemented:

- Hand-labeled 100-example factuality calibration CSV.
- Calibration curves/threshold search for BioNLI entailment, contradiction, and unsupported labels.
- External metrics export for reward/lifecycle evaluation.
- Learned RL policy or model weights; Q-like telemetry remains the active path.

Blocked:

- No blocker for current no-GPU hosted path.
- Full end-to-end answer-quality improvement measurement is blocked on a slower hosted-LLM replay harness and a labeled expected-answer set.

Tests run:

- Focused hosted API tests: `Ran 17 tests`, `OK`.
- Full hosted no-GPU API suite: `Ran 64 tests`, `OK`, `skipped=3`.
- Saved-conversation replay, default token budget:
  - source docs: 89;
  - sessions: 14;
  - evaluated sessions: 4;
  - turn pairs: 5;
  - baseline reward: 0.2262;
  - current reward: 0.2262;
  - reward delta: 0.0000;
  - baseline/current context support: 0.2326 / 0.2326;
  - current average context tokens: 293.6 vs baseline 300.8.
- Saved-conversation constrained replay:
  - token budget 160: reward delta `+0.0540`, support delta `+0.1355`, current compressed count `0.8`;
  - token budget 120: reward delta `+0.0391`, support delta `+0.0683`, current compressed count `1.0`;
  - token budget 80: reward delta `+0.0341`, support delta `+0.0475`, current compressed count `0.6`.
- Live health after scoped API restart:
  - `http://127.0.0.1:18081/health` returned 200;
  - `https://bayleigh-juxtapositional-shirleen.ngrok-free.dev/health` returned 200.

## Update 2026-05-19: Longitudinal Memory/Factuality Consistency Layer

Implemented:

- Added `app.memory.consistency`:
  - generic user-steering extraction for corrections such as "pivot from X to Y", "instead of X use Y", "not X but Y";
  - persisted conversation-frame construction with active terms, avoided/retired terms, recent corrections, supported claims, contradicted claims, and unsupported claims;
  - prompt rendering for the active conversation frame;
  - frame-alignment and frame-drift scoring;
  - longitudinal consistency report for current claim support, prior evidence-supported memory, and frame drift;
  - conservative prior-claim conflict detection using high lexical overlap plus opposite negation polarity.
- Added `conversation_frame` persistence in `MemoryStore`.
- Added `supported_claim_evidence()` in `MemoryStore`, which turns prior entailed evidence-table claims into memory evidence candidates for later turns.
- Wired `ContextPolicy.plan()` to render the conversation frame into the model context.
- Wired `ContextPolicy.observe_turn()` to:
  - retrieve prior conversation frame;
  - retrieve prior entailed claims from evidence tables;
  - pass prior supported claims into `assess_claim_support()` so the existing BioNLI batch path can compare new claims against prior evidence-supported memory;
  - compute `longitudinal_consistency`;
  - pass the longitudinal report into `reward_report()`;
  - persist an updated conversation frame after each turn.
- Added reward fields:
  - `frame_alignment`;
  - `frame_drift_penalty`;
  - `prior_memory_conflict_penalty`;
  - `longitudinal_penalty`.
- Chat SSE now emits a `consistency_warning` for longitudinal frame drift or prior-memory conflicts.
- When memory debug is enabled, chat SSE emits `conversation_frame`.
- Streamlit memory-debug handling now captures `conversation_frame`.

Partially implemented:

- General off-topic drift is now represented as conversation-frame drift, not only mechanistic-vs-pharmacological drift. It is still lexical/heuristic and should later be augmented by LLM/frame classification and labeled calibration.
- Cross-turn BioNLI support is available through prior entailed claims being added as source evidence candidates; it is bounded by the existing claim-support batching and comparability gate.
- Prior-memory contradiction detection is conservative and currently catches strongest polarity conflicts, not all semantic contradictions.
- Existing sessions do not yet have conversation-frame docs until they receive new turns after this implementation.

Not implemented:

- Higher-confidence semantic drift detection. Current frame drift is lexical/heuristic only; later work should add a bounded semantic classifier with labeled calibration, explicit active-frame vs answer comparison, and metrics for false drift warnings.
- No full contradiction-resolution workflow or fork/thread flow.
- No human-labeled longitudinal consistency benchmark.
- No background migration that backfills conversation frames from old sessions.
- No external metrics export for frame drift/longitudinal penalties.

Blocked:

- No blocker for hosted no-GPU path.
- Higher-confidence semantic drift detection is blocked on either labeled calibration data or a provider-backed frame classifier.

Tests run:

- Focused hosted tests: `Ran 15 tests`, `OK`.
- Full hosted no-GPU API suite: `Ran 69 tests`, `OK`, `skipped=3`.
- Existing-memory smoke inside API container:
  - sampled 3 sessions with evidence tables;
  - existing sessions had no conversation-frame docs yet, as expected before new turns;
  - supported-claim evidence retrieval returned 0 for sampled old sessions because their saved evidence tables did not contain entailed claims suitable for replay.
- Restarted only `api` and `streamlit` services.
- Health checks:
  - `http://127.0.0.1:18081/health` returned 200;
  - `https://bayleigh-juxtapositional-shirleen.ngrok-free.dev/health` returned 200.

## Update 2026-05-19: Lung Cancer Synthetic Conversation Evaluation Lab

Implemented:

- Added standalone evaluation laboratory under `evals/lung_factuality_lab/`.
- Added Pydantic schemas for:
  - gold claims;
  - mechanism graphs;
  - scenarios;
  - injected traps;
  - conversation turns;
  - assistant answers;
  - extracted claims;
  - claim judgments;
  - decomposed turn scores;
  - conversation traces;
  - failure boards;
  - recommendations;
  - regression tests.
- Added JSON-compatible YAML config/data files:
  - `configs/default_eval.yaml`;
  - `configs/reward_weights.yaml`;
  - `configs/batch_runs.yaml`;
  - curated gold claims and mechanism graphs;
  - 8 initial lung-cancer/TME scenarios;
  - perturbation banks for wrong directions, unsupported claims, citation drift, and generic errors.
- Added deterministic pipeline modules:
  - scenario loader with PyYAML-or-JSON fallback;
  - synthetic conversation generator;
  - dummy assistant adapter with controllable failure modes;
  - HTTP chat adapter for future target-chatbot runs;
  - claim extractor;
  - evidence/gold-claim matcher;
  - mechanism graph matcher;
  - claim judge;
  - decomposed reward scorer;
  - trace writer;
  - diagnosis engine;
  - failure board builder;
  - recommendation engine;
  - regression planner;
  - Markdown report writer.
- Added CLI commands:
  - `python -m evals.lung_factuality_lab.src.run_single`;
  - `python -m evals.lung_factuality_lab.src.run_batch`;
  - `python -m evals.lung_factuality_lab.src.compare_runs`;
  - `python -m evals.lung_factuality_lab.src.regression_planner`.
- Per scenario run now writes:
  - `scenario.yaml`;
  - `generated_conversation.jsonl`;
  - `assistant_answers.jsonl`;
  - `extracted_claims.jsonl`;
  - `claim_judgments.jsonl`;
  - `turn_scores.jsonl`;
  - `conversation_trace.json`;
  - `failure_board.json`;
  - `simulation_report.md`;
  - `recommendations.json`;
  - `recommendations.md`;
  - `regression_plan.yaml`.

Partially implemented:

- The dummy and local deterministic evaluator are complete enough for the first lab iteration.
- The HTTP adapter exists, but the full target-chatbot/live SSE evaluation has not yet been validated as a standard batch.
- Biomedical normalization is intentionally lightweight and curated-fixture based; it is modular so later commits can replace extraction/judging without changing trace/report contracts.
- Scenario data starts with 8 seed scenarios, not a large benchmark.

Not implemented:

- Large labeled corpus of synthetic and real conversations.
- Provider-backed semantic drift classifier.
- Full source-paper retrieval inside the lab; current evidence layer is curated fixture data.
- Direct integration with OpenSearch documents or the live chat memory index.
- CI job for running the lab across commits.

Blocked:

- No hosted no-GPU blocker.
- High-fidelity target-chatbot evaluation requires choosing the stable endpoint/auth path and acceptable provider-cost budget.

Tests run:

- Local lab unit tests: `Ran 7 tests`, `OK`.
- Local CLI smoke:
  - `run_single` wrote all expected files for `expert_hgf_met_direction_001`;
  - `run_batch` wrote batch report/failure board/recommendations for all 8 seed scenarios;
  - `compare_runs` wrote comparison outputs;
  - `regression_planner` wrote a regression plan from a failure board.
- Authoritative repo validation through hosted API image:
  - `./scripts/compose-hosted.sh run --rm --no-deps -v "$PWD/evals:/app/evals:ro" api python -m unittest discover -s /app/evals/lung_factuality_lab/tests -p "test_*.py"` returned `Ran 7 tests`, `OK`;
  - hosted API-image batch smoke wrote `evals/lung_factuality_lab/runs/batch_smoke`.
- Running stack remained healthy; no GPU services were started.

## Update 2026-05-19: Seed Conversations And Evaluator Fixture Banks

Implemented:

- Added explicit separation between:
  - scenario purpose files;
  - seed conversation JSONL files;
  - user false-premise trap bank;
  - known wrong-answer replay bank.
- Added seed conversations under `evals/lung_factuality_lab/data/conversations/seed/`:
  - `expert_hgf_met_direction_001.jsonl`;
  - `expert_tam_cd8_immunosuppression_001.jsonl`;
  - `hypoxia_immune_escape_001.jsonl`;
  - `correction_scope_tme_only_001.jsonl`;
  - `cross_cancer_transfer_001.jsonl`;
  - `citation_drift_lung_vs_general_oncology_001.jsonl`.
- Added generated-conversation output location:
  - `evals/lung_factuality_lab/data/conversations/generated/.gitkeep`.
- Added `user_false_premise_bank.yaml` with reusable user-side traps and variant prompts.
- Added `assistant_wrong_answer_bank.yaml` with known bad assistant answers and expected evaluator/reward judgments.
- Added `conversation_loader.py`:
  - loads seed JSONL;
  - preserves turn, role, text, expected behavior, target claims, trap ids, must-mention, must-not-claim, and scope.
- Updated `ConversationTurn` schema to support seed JSONL fields while preserving old `user_message` compatibility.
- Updated `Scenario` schema and scenario data to support `conversation_file`.
- Updated `conversation_generator.py`:
  - loads seed conversations from scenario files;
  - produces deterministic variants for trap turns from false-premise banks;
  - preserves trap/scope/expected-behavior metadata in generated variants.
- Added `wrong_answer_replay` assistant adapter to test evaluator/reward behavior without depending on the live chatbot.
- Updated `run_single` and `run_batch`:
  - support seed conversations by default;
  - support `--variant-index`;
  - support `--assistant wrong_answer_replay`;
  - write generated conversation and full traces as before.

Partially implemented:

- Generated variants are deterministic and template-based for v1; no large combinatorial generator yet.
- Seed conversations cover the recommended core set, while the original 8-scenario batch remains available.
- Wrong-answer replay validates evaluator behavior for selected known failures, not a complete wrong-answer corpus.

Not implemented:

- Large automatically generated conversation corpus.
- Rich paraphrase generation through a provider model.
- Exhaustive wrong-answer bank across all gold claims and mechanisms.
- Live target-chatbot batch run using the seed conversations.

Blocked:

- No hosted no-GPU blocker.
- Larger generated variants and live target-chatbot batches still need an agreed provider-cost/latency budget.

Tests run:

- Local lab tests: `Ran 11 tests`, `OK`.
- Local CLI smoke:
  - `run_single --assistant wrong_answer_replay`;
  - `run_single --variant-index 2`;
  - `run_batch --variant-index 1`;
  - `compare_runs`;
  - `regression_planner`.
- Authoritative repo validation through hosted API image:
  - `Ran 11 tests`, `OK`;
  - `wrong_replay_smoke` wrote full run outputs;
  - `batch_seed_variant_smoke` wrote full batch outputs.
- Hosted stack status checked after validation; existing services remained up and GPU services remained off.

## Update 2026-05-25: Shape7m SSE Stability And Evaluator False-Positive Reduction

Implemented:

- Verified authoritative git and hosted stack status before changes; GPU-profile services remained off.
- Narrowed chat correction-only detection so normal task constraints such as "not clinical treatment advice" no longer trigger correction acknowledgement mode.
- Hardened API SSE streaming:
  - server-side generator exceptions now emit a structured `error` SSE event before stream end when possible;
  - server logs retain the exception without exposing secrets to the client.
- Hardened Streamlit chat UI:
  - structured `error` events show as warnings;
  - interrupted HTTP streams show a retry warning instead of a raw traceback;
  - partial assistant text is preserved when a stream is interrupted after tokens were received.
- Improved lab claim extraction/judging for rejected false premises and evaluator fixtures:
  - `activating` is recognized as an activation predicate;
  - rejected or quoted wrong phrases are matched against markdown-normalized text;
  - generic rejection/absence markers such as `do not support`, `without defining`, `unsupported claims about`, and `which contradict` prevent false factual-inversion penalties.
- Updated `evals/lung_factuality_lab/configs/reward_shape_registry.yaml`:
  - recorded Shape7l Sentinel C outcome;
  - recorded Shape7m HGF live/replay outcomes;
  - set current stage to `shape7m_hgf_evaluator_and_sse_fix_correction_scope_microcheck_pending`;
  - kept Stage 2 blocked until correction-scope/citation microchecks clear.
- Rebuilt and restarted only `api` and `streamlit` via `scripts/compose-hosted.sh`; `worker-cpu` stayed running and GPU services stayed off.

Partially implemented:

- Shape7m HGF replay cleared rejected false-premise factual-inversion false positives, but fresh live generation still needs broader guard validation.
- HGF remaining replay failures are now lower severity and mainly about missing required mechanism nodes or unsupported diagnostic wording, not missed traps.
- SSE hardening improves UI behavior on server-side stream failures, but it cannot prevent interruption if the API container is intentionally rebuilt during an active user chat.

Not implemented:

- Broader Stage 2 rerun after Shape7m.
- Full Sentinel C rerun after the latest Shape7m evaluator marker updates.
- Higher-confidence semantic drift model panel.
- Full OpenIE source-sentence NLI calibration and post-generation claim repair beyond the current lightweight guard.

Blocked:

- Do not start Stage 2 from Shape7l/Shape7m yet; Shape7l Sentinel C improved over Shape6 but still had too many failed turns and missed traps.
- Next required live microcheck is `correction_scope_tme_only_001__gen_004`, followed by citation drift if correction scope clears.

Tests run:

- `./scripts/compose-hosted.sh ps`: hosted stack up; `api`, `streamlit`, and `worker-cpu` running; GPU services not running.
- `curl -fsS http://127.0.0.1:18081/health`: `{"status":"ok"}`.
- `tests.test_chat_auto_context`: `Ran 7 tests`, `OK`.
- Focused API suite `tests.test_memory_web_search tests.test_memory_search_agent tests.test_chat_auto_context`: `Ran 46 tests`, `OK`.
- Lab claim-judging focused tests: `Ran 29 tests`, `OK`.
- Full lung factuality lab tests: `Ran 45 tests`, `OK`.
- Shape7m HGF live run: `evals/lung_factuality_lab/runs/shape7m_live_hgf_gen003_after_sse_evaluator_fix`.
- Shape7m HGF replay after absence-marker fix: `evals/lung_factuality_lab/runs/shape7m_replay_hgf_gen003_live2_absence_marker_fix`, failed turns reduced to 2, highest severity reduced to 3, missed traps 0.
