# 2026-05-14 Agent Handoff Context

This file is deterministic context for future development agents. Read it before planning or editing. Do not rediscover these details unless there is evidence they changed.

## Non-Omission Rule

The user explicitly requested that future agents must not silently omit details. When reporting status, use these labels:

- Implemented
- Partially implemented
- Not implemented
- Blocked
- Deferred by priority

If a feature is only scaffolded, say "partially implemented" or "scaffold only"; do not call it implemented.

## User Goal

The user wants the chatbot endpoint to use an inference-time memory/context-manager agent that behaves like an operating-system scheduler for finite model context:

- keep short/fast working context in the prompt;
- keep long/slower memory in OpenSearch/Elasticsearch;
- compress and index conversation context;
- avoid catastrophic forgetting due to context-window limits;
- use term/BM25-first retrieval, with semantic/vector search as fallback;
- use semantic triplets and source sentences to enrich context and judge consistency;
- use reward shaping without fine-tuning now, while logging traces for future RL/fine-tuning;
- track landmarks, reward state, state-action traces, and reusable policy patterns;
- optionally use NVIDIA hosted model for context-manager work when local GPU is constrained;
- redact private data before web search;
- warn model/user about possible inconsistencies;
- allow user corrections to be persisted;
- add future support for forks/threaded contradiction-resolution workflows, but forks are lower priority.

## Current Date And Environment

- Current date: 2026-05-19 for the latest continuation work; original handoff file began on 2026-05-14.
- User timezone: America/Mexico_City.
- Local artifact repo path: `/mnt/d/UserFolders/Documents/GitHub/ai-research-insights-doc-artifacts`.
- Important local source mirror: `source_snapshot/`.
- Remote implementation server: blue-demon GPU server.
- Remote host: `192.168.241.149`.
- Remote user: `iarroyof`.
- Remote implementation repo path: `/home/iarroyof/sabia/ai-research-insights`.
- Remote implementation app path: `/home/iarroyof/sabia/ai-research-insights/services/api/app`.
- Remote branch observed: `main`.
- Remote HEAD observed: `8d79f3d Added documentation without my mail`.
- Remote app and local source snapshot matched byte-for-byte before the memory implementation for `api_app` vs `services/api/app`, excluding `__pycache__`.

Do not print secrets. The blue-demon credentials file is located one directory above the local artifact repo as `../blue-demon.txt`. NVIDIA API token was integrated into the remote `.env`; do not print it.

## Docker And GPU Constraints

- The project runs with Docker Compose.
- The user said GPUs are currently in use.
- Do not start local GPU-heavy services unless explicitly permitted.
- Avoid starting:
  - `llm`
  - `worker-gpu`
  - `models-init`
  - `rebel-extractor`
  - any service that loads local models on GPU
- Safe checks used so far:
  - `docker compose config`
  - Python AST syntax checks
  - direct NVIDIA hosted HTTPS calls
  - no-deps API import test, when Docker CLI behaves
- Some `docker run` / `docker compose run --no-deps` smoke tests hung in Docker CLI before a visible container appeared. Stop only those smoke-test processes if needed; do not kill unrelated containers.

## Remote .env / NVIDIA State

The remote `.env` in `~/sabia/ai-research-insights/.env` has been updated with:

- `NVIDIA_API_KEY=<parsed nvapi token>`
- `CONTEXT_MANAGER_PROVIDER=nvidia`
- `NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1`
- `NVIDIA_MODEL=nvidia/llama-3.3-nemotron-super-49b-v1.5`

The remote `docker-compose.yml` was updated so the `api` service receives:

- `CONTEXT_MANAGER_PROVIDER`
- `CONTEXT_MANAGER_MODEL`
- `NVIDIA_API_KEY`
- `NVIDIA_BASE_URL`
- `NVIDIA_MODEL`
- `NVIDIA_REASONING_EFFORT`
- `NLI_PROVIDER`
- `HF_API_TOKEN`

`docker compose config` passed after this change.

## Hugging Face Token State

The local Hugging Face token file is:

```text
/mnt/d/UserFolders/Documents/hf_huggingface_token.txt
```

Do not print it. Parse only a token-like value, normally `hf_...`, and write it to the project environment as:

```text
HF_API_TOKEN=<token>
NLI_PROVIDER=hf_api
```

Target environment file on blue-demon:

```text
/home/iarroyof/sabia/ai-research-insights/.env
```

The API service already receives `HF_API_TOKEN` and `NLI_PROVIDER` through Docker Compose after the latest update.

HF token/model verification passed on blue-demon without local model loading:

- working endpoint:

```text
https://router.huggingface.co/hf-inference/models/pritamdeka/PubMedBERT-MNLI-MedNLI
```

- legacy endpoint failed with 404 and should not be used:

```text
https://api-inference.huggingface.co/models/pritamdeka/PubMedBERT-MNLI-MedNLI
```

- hosted pipeline accepted SEP-string input:

```text
hypothesis [SEP] premise
```

- sanity results:
  - entailment fixture dominant label: entailment `0.9987`;
  - contradiction fixture dominant label: contradiction `0.9978`.

NVIDIA hosted API was tested directly without local GPU use:

- Endpoint returned HTTP 200.
- With `max_tokens=64`, the Nemotron model returned empty final content and `finish_reason=length`, apparently spending budget on reasoning fields.
- With `max_tokens=512`, it returned `OK`.
- The client was hardened to extract `nvapi-...` from key files containing prose and to default NVIDIA policy calls to a larger budget.

## Files Changed In Latest Implementation

Local artifact mirror:

- `source_snapshot/api_app/config.py`
- `source_snapshot/default.yaml`
- `source_snapshot/docker-compose.yml`
- `source_snapshot/api_app/clients/llm.py`
- `source_snapshot/api_app/routers/chat.py`
- `source_snapshot/api_app/memory/__init__.py`
- `source_snapshot/api_app/memory/policy.py`
- `source_snapshot/api_app/memory/store.py`
- `source_snapshot/api_app/memory/rewards.py`
- `source_snapshot/api_app/memory/privacy.py`
- `source_snapshot/api_app/memory/web_search.py`
- `source_snapshot/api_app/memory/nli.py`

Remote implementation repo has corresponding changes:

- `config/default.yaml`
- `docker-compose.yml`
- `services/api/app/clients/llm.py`
- `services/api/app/config.py`
- `services/api/app/routers/chat.py`
- `services/api/app/memory/`

Observed remote status after latest changes:

- modified: `config/default.yaml`
- modified: `docker-compose.yml`
- modified: `services/api/app/clients/llm.py`
- modified: `services/api/app/config.py`
- modified: `services/api/app/routers/chat.py`
- untracked: `services/api/app/memory/`

No commit was made.

## Latest Seeded Evaluation Lab State - 2026-05-19

Implemented:

- `evals/lung_factuality_lab` now supports seed conversations, generated variants, dummy/wrong-answer replay/target adapters, claim extraction/judging, decomposed reward scoring, failure boards, recommendations, regression plans, batch runs, comparison runs, and scenario-linked output artifacts.
- Reward/evaluator shaping was run against the seeded conversations:
  - baseline dummy seeded batch average reward: `0.3514`;
  - shaped dummy seeded batch average reward: `0.4469`;
  - reward delta: `+0.0955` across 8 scenarios;
  - shaped dummy seeded batch summary: `total_turns=16`, `failed_turns=11`, `failure_count=14`, `missed_injected_traps=0`;
  - shaped wrong-answer replay batch average reward: `0.4194`, `failure_count=13`, `missed_injected_traps=0`.
- Fixes from seeded simulations:
  - resisted false-premise traps are no longer counted as missed failures;
  - TAM/CD8 and hypoxia/HIF mechanism nodes use aliases instead of brittle exact matching;
  - HGF/MET inversion checks no longer match `metabolic` as `MET`;
  - cross-cancer direct-proof claims are flagged as `cross_domain_transfer`;
  - batch failure items keep `scenario_id`, making recommendations and generated regression tests actionable.
- Local validation command:
  - `python -m unittest discover -s evals/lung_factuality_lab/tests -p 'test_*.py'`;
  - result: `Ran 15 tests`, `OK`.
- Hosted API-image validation command:
  - `./scripts/compose-hosted.sh run --rm --no-deps -v "$PWD/evals:/app/evals:ro" api python -m unittest discover -s /app/evals/lung_factuality_lab/tests -p "test_*.py"`;
  - result: `Ran 15 tests`, `OK`.
- Hosted seeded dummy smoke result:
  - average reward `0.4469`;
  - `total_turns=16`, `failed_turns=11`, `failure_count=14`, `missed_injected_traps=0`;
  - GPU-profile services were not started.

Partially implemented:

- The seeded lab evaluates dummy and replay adapters locally and can target the chatbot, but broad target-chatbot hosted simulation has not yet been run as a long evaluation suite.
- Higher-confidence semantic drift detection remains not implemented and should stay listed as pending.
- The reward shaping is deterministic and Q-like telemetry only; no model weights are trained.

## Large Generated Corpus Evaluation State - 2026-05-19

Implemented:

- Integrated `lung_factuality_large_corpus_v1.zip` from `C:\Users\nachi\Downloads\lung_factuality_large_corpus_v1.zip`.
- Corpus contents now live under `evals/lung_factuality_lab/`:
  - `data/conversations/generated/`;
  - `data/scenarios/generated_scenarios.yaml`;
  - `data/evidence/generated_gold_claims.yaml`;
  - `data/evidence/generated_mechanism_graphs.yaml`;
  - `data/perturbations/generated_user_false_premise_bank.yaml`;
  - `data/perturbations/generated_assistant_wrong_answer_bank.yaml`;
  - `data/verification/`;
  - `scripts/generate_large_corpus.py`.
- Added `configs/generated_batch_runs.yaml` to run all 120 generated scenarios.
- Lab loader now merges generated evidence/scenario/trap files and supports generated conversation metadata.
- Generated wrong-answer replay now matches base scenario plus variant index.
- Reward/evaluator shaping from the large run:
  - stricter forbidden-claim matching;
  - resisted-trap recognition for transfer-hypothesis/background/not-direct-proof language;
  - generated biomedical aliases for MDSC/Treg, HIF/PD-L1, ECM stiffness, CAF heterogeneity, and cross-cancer scope;
  - curated wrong variants judged before generic scope drift;
  - negated citation-scope language no longer counted as cross-domain transfer.

Benchmark results:

- Initial after-import dummy benchmark:
  - average reward `0.2235`;
  - missed injected traps `318`.
- Final shaped dummy benchmark:
  - average reward `0.88`;
  - `scenario_count=120`, `total_turns=840`, `failed_turns=120`, `failure_count=120`, `missed_injected_traps=0`;
  - reward delta `+0.6565`.
- Initial after-import generated wrong-answer replay benchmark:
  - average reward `0.1924`;
  - missed injected traps `340`.
- Final shaped generated wrong-answer replay benchmark:
  - average reward `0.74`;
  - `scenario_count=120`, `total_turns=840`, `failed_turns=240`, `failure_count=355`, `missed_injected_traps=0`;
  - reward delta `+0.5476`.

Hosted validation:

- Hosted API-image tests: `Ran 16 tests`, `OK`.
- Hosted dummy large smoke: average reward `0.88`, `missed_injected_traps=0`.
- Hosted generated wrong-answer replay large smoke: average reward `0.74`, `missed_injected_traps=0`.
- No hosted containers were restarted and GPU-profile services remained off.

Partially implemented:

- The large benchmark has been run with dummy and generated wrong-answer replay adapters. A full target-chatbot run against the live `/chat/` endpoint is still pending and should be treated as a separate, slower benchmark because it will call hosted LLM/search providers.
- The remaining dummy failures are turn-7 scope/diagnostic-answer gaps in the dummy adapter, not missed injected traps.

## Current Memory Implementation Summary

Implemented:

- Chat endpoint has a pre-generation memory/context policy hook.
- Chat endpoint has post-generation reward/trace storage.
- Session memory is stored in OpenSearch index `<prefix><tenant>_chat_memory`.
- Recent messages are used as working buffer.
- Session memory is searched with term/BM25-like OpenSearch query.
- Landmarks are stored for current focus, open question, latest reward state, and corrections.
- Triplet search uses existing `app.triplets.search.search_triplets`.
- DuckDuckGo Instant Answer search exists behind redaction and is off by default.
- Answer triplets are extracted after generation through existing extraction client when available.
- Reward report includes lexical relevance, context support, sentiment delta, triplet conflict penalty, NLI factuality, latency penalty, and token penalty.
- User correction endpoint exists: `POST /chat/memory/correction`.
- NVIDIA-compatible `LLMClient.chat_once()` exists for context-manager/reflection calls.
- Biomedical NLI hook exists with providers:
  - `hf_api` default after the latest update;
  - `llm`;
  - `http`.
  - `heuristic` fallback.

Partially implemented:

- Context manager is still deterministic scheduler, not learned policy.
- RL is trace logging and reward shaping only, not Q-learning, bandit learning, or fine-tuning.
- Memory hierarchy exists only as working buffer plus OpenSearch memory; no robust compression/eviction/promote/demote loop.
- NLI hook is implemented but a real biomedical NLI HTTP service is not deployed.
- Hugging Face Inference API support for `pritamdeka/PubMedBERT-MNLI-MedNLI` has been added as the preferred remote NLI route. It requires `HF_API_TOKEN` in the environment. Local model execution is fallback only.
- Triplet conflict detection is heuristic and should not be treated as truth.
- User warnings are SSE events, but UI workflow is not implemented.
- The idea-index is not implemented; only per-message `terms` are stored and boosted.
- Shared cross-session/cross-tenant policy memory is config placeholder only.
- Reflection is scaffolded behind `memory.use_llm_reflection`; not central to policy.
- Semantic/vector fallback was not newly implemented.

Not implemented:

- Full idea/concept frequency index.
- Aggregate concept tree with parent/child paths, co-occurrence, reward statistics, and transferability.
- Atomic claim extraction from full answer sentences.
- Premise/hypothesis comparability gate before NLI.
- Real PubMedBERT/MedNLI/BioNLI service deployment.
- Evidence table UI.
- Citation-backed answer UI improvements.
- Conversation forks/threading for contradiction resolution.
- Real action-value table/Q-table/contextual bandit.
- Human feedback UI beyond backend correction endpoint.
- Long-term memory compression summaries.
- Replay buffer export for future training/fine-tuning.

## Important Design Clarification From 2026-05-14

The user clarified that for factuality, the preferred primary check should be:

```text
premise = database/source PubMed sentence or small source sentence window
hypothesis = atomic answer claim or answer sentence
model = biomedical/clinical NLI
```

Triplets should not be the main truth judge. Triplets are too lossy for truth:

- can drop negation;
- can drop modality/speculation;
- can drop causal direction;
- can drop population/species/cell-line context;
- can flatten association vs causality;
- can obscure exact evidence language.

Triplets remain highly valuable for:

- retrieval expansion;
- finding candidate source sentences for NLI;
- memory compression;
- idea-index construction;
- concept/entity/relation graph navigation;
- contradiction candidate discovery;
- conversation orientation.

The final factuality reward should go back to original source sentences used as context.

## Current Research/User Feature Signals

Observed feature priorities from current web/research signals:

- Literature review tools focus on search, screening, data extraction, synthesis, quality assessment, and reporting.
- Transparency, traceability, and explainability are major trust challenges in AI-supported systematic review tools.
- Elicit-style workflows are valued because they find papers and produce research-backed summaries.
- SciLit-style workflows combine paper recommendation, highlight extraction, and citation sentence suggestion.
- User discussions repeatedly ask for:
  - referenced answers with page/source location;
  - extracting relevant information from many papers;
  - research boards / selected source sets;
  - citation graphs and semantic graphs;
  - chat over PDFs/knowledge base;
  - AI summaries that remain grounded in saved material;
  - dynamic RAG where selected source docs can be included/excluded by round.

Plan priority should therefore emphasize:

1. Source-grounded factuality and citation traceability.
2. Evidence table and extraction workflow.
3. Selected-source context control and dynamic RAG.
4. Idea/concept memory index for large knowledge bases.
5. Literature/citation/triplet graph navigation.
6. User feedback/correction loop.
7. Fork/thread workflows later.

## Key Source References Used For Planning

- Bolaños et al., "Artificial intelligence for literature reviews: opportunities and challenges", Artificial Intelligence Review, 2024. Relevant for SLR stages, extraction/screening, usability, transparency, and knowledge graphs.
- OECD page on Elicit as a language-model research tool. Relevant for literature-review workflow and research-backed answers.
- SciLit paper, "A Platform for Joint Scientific Literature Discovery, Summarization and Citation Generation". Relevant for paper recommendation, highlight extraction, and citation sentence suggestion.
- Public user discussions on Reddit about AI tools for research and knowledge management. Relevant as weak but useful product-signal evidence for "nice to have" features: cited answers, page numbers, PDF chat, dynamic RAG, semantic/citation graph, and knowledge-base memory.

## Commands Already Used Safely

Examples, do not print secrets:

```bash
sshpass -p '...' ssh -o StrictHostKeyChecking=no iarroyof@192.168.241.149 'cd sabia/ai-research-insights && git status --short && git diff --stat'
sshpass -p '...' ssh -o StrictHostKeyChecking=no iarroyof@192.168.241.149 'cd sabia/ai-research-insights && docker compose config >/tmp/ai_research_compose_config.yml && echo compose-config-ok'
python3 -m py_compile source_snapshot/api_app/config.py source_snapshot/api_app/clients/llm.py source_snapshot/api_app/routers/chat.py source_snapshot/api_app/memory/*.py
```

Because the local sandbox can fail in WSL mount paths, commands often required `sandbox_permissions=require_escalated`.

## Next Agent First Steps

1. Read this file.
2. Read `docs/2026-05-14-reimplementation-plan.md`.
3. Read `docs/2026-05-14-status-tracker.md`.
4. Inspect current diffs locally and remotely.
5. Do not start GPU services unless user explicitly allows.
6. Implement the next highest-priority phase: evidence-grounded NLI with premise/hypothesis comparability and source sentence provenance.
7. Enforce module-level test gates before integrating new code upward. Test NVIDIA hosted API paths first for LLM roles, Hugging Face API first for NLI, and local models only after explicit GPU approval.

## 2026-05-19 Factuality/Memory Replay Handoff

Implemented:

- Reward calibration now includes domain alignment, mechanistic-vs-pharmacological off-topic penalty, and best-supporting-item context support.
- Memory lifecycle selection is query-aware and has extractive compression fallback for over-budget memory items.
- The live context policy passes the user query into recent working-set selection.
- `tools/simulate_memory_quality.py` replays saved conversation memory docs and compares current lifecycle policy against a legacy recent-buffer baseline.

Partially implemented:

- Compression/eviction/promotion is active inside working-set selection but not yet a full background compaction system.
- Default-budget replay is neutral, while constrained-budget replay shows improvement.
- Simulation measures support/reward proxies over saved answers, not regenerated answer quality.

Not implemented:

- Labeled disease-specific factuality corpus.
- End-to-end hosted-LLM replay for answer-quality deltas.
- External metrics export for simulation/reward trends.

Blocked:

- Nothing blocked for hosted no-GPU path. Full answer-quality evaluation needs labeled data and provider-backed replay.

Tests run:

- Focused memory/search tests: `Ran 17 tests`, `OK`.
- Full hosted API tests: `Ran 64 tests`, `OK`, `skipped=3`.
- Saved replay default: reward delta `0.0000`, support delta `0.0000`.
- Saved replay constrained budgets:
  - 160 tokens: reward delta `+0.0540`;
  - 120 tokens: reward delta `+0.0391`;
  - 80 tokens: reward delta `+0.0341`.
- API was restarted only for the `api` service; local and ngrok health checks returned 200.

## 2026-05-19 Longitudinal Consistency Handoff

Implemented:

- `app.memory.consistency` now provides a deterministic conversation-frame layer:
  - active terms;
  - avoided/retired terms from user steering;
  - supported/contradicted/unsupported claim memory;
  - frame rendering for prompts;
  - frame drift and prior-memory conflict reporting.
- `MemoryStore` persists `conversation_frame` docs and can retrieve prior entailed claims from evidence tables as `memory_claim` evidence candidates.
- `ContextPolicy.plan()` injects the active conversation frame into context.
- `ContextPolicy.observe_turn()` uses prior entailed claims as additional evidence for claim support/BioNLI, computes longitudinal consistency, updates reward, persists the updated frame, and returns debug metadata.
- Chat SSE emits longitudinal `consistency_warning` events and `conversation_frame` debug events.
- Streamlit captures the new `conversation_frame` debug event.

Partially implemented:

- General off-topic drift is now handled as frame drift, but the detector is lexical and conservative.
- Cross-turn contradiction checks use current claim support, prior entailed claims, evidence tables, and a deterministic prior-claim polarity check; deeper semantic contradictions still need labeled calibration or provider-backed classification.
- Existing old sessions will only get conversation frames after a new turn or a future backfill job.

Not implemented:

- Higher-confidence semantic drift detection. Current frame drift is lexical/heuristic; planned follow-up is a calibrated semantic classifier over active frame, user corrections, question, answer, and evidence spans.
- Backfill/migration for old sessions.
- Full contradiction workflow with fork/thread resolution.
- Longitudinal answer-regeneration benchmark.
- External metrics export.

Blocked:

- No hosted no-GPU blocker.
- More advanced semantic drift needs labeled data or a bounded provider-backed classifier.

Tests run:

- Focused hosted tests: `Ran 15 tests`, `OK`.
- Full hosted no-GPU API suite: `Ran 69 tests`, `OK`, `skipped=3`.
- API-container smoke confirmed new store methods import and run against OpenSearch.
- Restarted scoped services: `api`, `streamlit`.
- Local and ngrok health checks returned 200.

## 2026-05-19 Lung Factuality Lab Handoff

Implemented:

- New package: `evals/lung_factuality_lab/`.
- The lab is a modular evaluation harness, not just a scoring script.
- It writes machine-readable traces and agent-readable reports for each run.
- It supports:
  - curated gold claims;
  - mechanism graphs;
  - scenario specs;
  - injected traps;
  - synthetic conversation turns;
  - dummy assistant failure modes;
  - optional HTTP target-chatbot adapter;
  - claim extraction;
  - claim judgment;
  - decomposed reward;
  - failure board;
  - recommendations;
  - regression plan;
  - run comparison.
- Initial batch has 8 seed scenarios covering:
  - basic TME factuality;
  - expert HGF/MET direction;
  - TAM/CD8 immunosuppression;
  - hypoxia/immune escape;
  - cross-cancer transfer;
  - correction/scope drift;
  - citation drift.

Partially implemented:

- The framework is intentionally fixture-driven and deterministic for v1.
- The HTTP adapter exists, but target-chatbot batch replay should be validated next against the live `/chat/` endpoint.
- Claim extraction and relation normalization are lightweight but isolated behind modules so they can be replaced across commits without invalidating output contracts.

Not implemented:

- Large labeled benchmark.
- CI across commits.
- Provider-backed semantic drift classifier.
- Live OpenSearch/source-paper retrieval inside the lab.

Blocked:

- No hosted no-GPU blocker.
- Live target-chatbot evaluation needs a stable endpoint/auth choice and provider-cost budget.

Tests run:

- Local lab unit tests: `Ran 7 tests`, `OK`.
- Local CLI smoke covered `run_single`, `run_batch`, `compare_runs`, and `regression_planner`.
- Authoritative repo validation used `scripts/compose-hosted.sh run` with the hosted API image and returned `Ran 7 tests`, `OK`.
- Authoritative hosted-image batch smoke wrote `evals/lung_factuality_lab/runs/batch_smoke`.

## 2026-05-19 Seed Conversation Handoff

Implemented:

- Seed conversations now live as JSONL data files under `evals/lung_factuality_lab/data/conversations/seed/`.
- Scenario files now support `conversation_file`, so scenario purpose is separated from user-turn content.
- User-side traps now live in `data/perturbations/user_false_premise_bank.yaml`.
- Known bad assistant answers now live in `data/perturbations/assistant_wrong_answer_bank.yaml`.
- `conversation_loader.py` loads seed JSONL.
- `conversation_generator.py` can create deterministic trap-turn variants while preserving metadata.
- `assistant_adapters.py` now includes `wrong_answer_replay` to validate the evaluator/reward system independently of the live chatbot.
- `run_single`/`run_batch` support `--variant-index` and wrong-answer replay.

Partially implemented:

- Variant generation is deterministic/template-based.
- Wrong-answer fixtures cover selected core failures only.

Not implemented:

- Large generated conversation corpus.
- Provider-generated paraphrase variants.
- Full live target-chatbot seed-conversation batch.

Blocked:

- No hosted no-GPU blocker.
- Larger variant generation and live replay need an agreed provider-cost budget.

Tests run:

- Local lab tests: `Ran 11 tests`, `OK`.
- Hosted API-image lab tests: `Ran 11 tests`, `OK`.
- Hosted API-image smoke runs:
  - `wrong_replay_smoke`;
  - `batch_seed_variant_smoke`.

## Mandatory Testing Policy

For every new module or testable submodule:

1. Write or adjust unit tests immediately.
2. Run mocked integration tests.
3. Run real-provider smoke tests where applicable.
4. Update `docs/2026-05-14-status-tracker.md`.
5. Do not let a higher-level feature depend on the module until these checks pass or the status is marked Blocked with reason.

Provider order:

- LLM main chat and context manager: NVIDIA hosted API first, local models later after GPU approval.
- Biomedical NLI: Hugging Face Inference API for `pritamdeka/PubMedBERT-MNLI-MedNLI` first, local model fallback later.
- Heuristic NLI: development fallback only, not a factuality authority.
