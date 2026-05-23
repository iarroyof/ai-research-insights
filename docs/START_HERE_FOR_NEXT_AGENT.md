# START HERE FOR NEXT AGENT

Read this file first. It is the concise entrypoint for continuing the AI Research Insights chatbot memory/factuality implementation.

## Immediate Read Order

1. `docs/START_HERE_FOR_NEXT_AGENT.md` - this file.
2. `docs/2026-05-14-agent-handoff-context.md` - deterministic environment, infra, secret-location, and current-state context.
3. `docs/2026-05-14-status-tracker.md` - exact implemented/partial/missing status.
4. `docs/2026-05-14-reimplementation-plan.md` - detailed implementation plan.

Do not start coding before reading those files.

If the task says to use the lung factuality lab and continue reward shaping in live conversations, also read:

1. `evals/lung_factuality_lab/README.md` - lab CLI, outputs, and live shaping ladder.
2. `evals/lung_factuality_lab/configs/reward_shape_registry.yaml` - accepted shape, rejected shapes, microfit gates, sentinel guards, protected holdout, and next allowed action.

For live reward shaping, diagnose one conversation at a time if needed, but do not promote a reward/evaluator shape from one conversation or replay-only metrics. Use the registered family microfit plus stratified guard ladder and record every attempted shape so a later agent does not revisit a weaker branch.

## Remote Implementation Repo

Main implementation lives on blue-demon:

```text
host: 192.168.241.149
user: iarroyof
repo: /home/iarroyof/sabia/ai-research-insights
api app: /home/iarroyof/sabia/ai-research-insights/services/api/app
```

Local artifact mirror:

```text
/mnt/d/UserFolders/Documents/GitHub/ai-research-insights-doc-artifacts
source mirror: source_snapshot/
```

The remote repo is a real Git checkout. The local directory is an artifact/report repo with a `source_snapshot/` mirror.

For implementation work, always treat this as authoritative:

```text
iarroyof@blue-demon:~/sabia/ai-research-insights/
```

Do not accidentally use the artifact directory as the project Git repo.

## Current Remote Git State

As of 2026-05-15 after the hosted/no-GPU infrastructure and zero-shot HF API work, blue-demon has uncommitted changes:

```text
 M .dockerignore
 M config/default.yaml
 M docker-compose.yml
 M services/api/app/clients/llm.py
 M services/api/app/config.py
 M services/api/app/routers/chat.py
 M services/api/app/services/zero_shot.py
 ?? docs/2026-05-14-agent-handoff-context.md
 ?? docs/2026-05-14-reimplementation-plan.md
 ?? docs/2026-05-14-status-tracker.md
 ?? docs/START_HERE_FOR_NEXT_AGENT.md
 ?? scripts/
 ?? services/api/Dockerfile.hosted
 ?? services/api/app/memory/
 ?? services/api/requirements.hosted.txt
 ?? services/api/tests/
```

No commit was made. Before large new work, strongly consider committing or otherwise saving this baseline after user approval.

## Hard Constraints

- The project runs with Docker Compose.
- GPUs are currently in use.
- Do not start local GPU/model-loading services unless the user explicitly allows.
- Avoid starting:
  - `llm`
  - `worker-gpu`
  - `models-init`
  - `rebel-extractor`
  - any service that loads local models on GPU.
- Safe validation:
  - Python syntax/AST checks.
  - Unit tests.
  - Mocked integration tests.
  - `docker compose config`.
  - Direct NVIDIA hosted API calls.
- Direct Hugging Face hosted API calls.
- Do not print secrets.

## Current Docker Infrastructure State

As of 2026-05-15:

- Docker daemon is active on blue-demon.
- Docker default runtime was changed from `nvidia` to `runc`; the `nvidia` runtime remains available for explicit GPU-profile services.
- The unrelated containers were checked after the latest project build/test run and are still running:
  - `set-attention`;
  - `webpunch-ai-feedback-insights-queue_worker-1`;
  - `webpunch-ai-feedback-insights-redis-1` healthy;
  - `webpunch-ai-feedback-insights-postgres-1`;
  - `webpunch-ai-feedback-insights-opensearch-1` healthy.
- `docker compose config` passes for the AI Research Insights repo.
- `scripts/compose-hosted.sh config --services` passes and excludes GPU/local model-serving services.
- `scripts/compose-local-gpu.sh config --services` passes and includes GPU services only through the explicit helper/profile.
- The project has been patched so hosted/no-GPU paths force `runc`:
  - non-GPU Compose services use `runtime: ${DOCKER_CPU_RUNTIME:-runc}`;
  - GPU-profile services use `runtime: ${DOCKER_GPU_RUNTIME:-nvidia}`;
  - `scripts/api-host-test.sh` passes `--runtime "${DOCKER_CPU_RUNTIME:-runc}"`;
  - Compose helper scripts export explicit runtime defaults.
- Hosted/no-GPU API build path is now conventional and small:
  - default API Dockerfile: `services/api/Dockerfile.hosted`;
  - hosted requirements: `services/api/requirements.hosted.txt`;
  - full local-ML/GPU API image remains `services/api/Dockerfile` and should not be used while GPUs are busy.
- Verified hosted image:
  - `ai-research-insights-api:hosted-verify` and `ai-research-insights-api:latest` both point to image ID `cabc9f2d83cb`;
  - image size: `464MB`.
- Verified commands:
  - hosted build: `docker build --progress=plain -t ai-research-insights-api:hosted-verify -f services/api/Dockerfile.hosted .`;
  - unit suite: `Ran 26 tests`, `OK`, `skipped=3`;
  - Compose one-shot API import: `compose-hosted-api-ok nvidia hf_api`;
  - isolated API health check: `api-health 200 {'status': 'ok'}`;
  - real HF zero-shot smoke: `zero-shot-hf-ok [('biomedical', 0.8951...), ('finance', 0.0184...)]`.
- Service isolation hardening:
  - default hosted project ports are configurable with `CADDY_HTTP_PORT`, `CADDY_HTTPS_PORT`, `API_HOST_PORT`, `OPENSEARCH_HOST_PORT`, `NEO4J_HTTP_PORT`, and `NEO4J_BOLT_PORT`;
  - `scripts/check-compose-isolation.sh` is a read-only preflight for host-port conflicts and foreign containers on `ai_research_insights_net`;
  - latest preflight result: `isolation-check-ok network=ai_research_insights_net ports=8080 8443 18081 19200 7474 7687`.
- `scripts/compose-hosted.sh` and `scripts/compose-local-gpu.sh` now set `COMPOSE_FILE=docker-compose.yml` by default so `docker-compose.override.yml` does not silently add extra host ports.
- Hosted Compose stack is currently running on blue-demon:
  - API: `http://127.0.0.1:18081`;
  - Caddy: `8080`/`8443`;
  - OpenSearch: `19200`;
  - Neo4j: `7474`/`7687`;
  - CPU worker is up and connected to Redis;
  - GPU-profile services are not running.
- `worker-cpu` now builds from `services/api/requirements.hosted.txt`; `worker-gpu` keeps `services/api/requirements.txt` only for explicit GPU/local mode.

Use `timeout` around Docker runtime tests. For non-GPU project containers, keep forcing `DOCKER_CPU_RUNTIME=runc` or `--runtime runc`. Do not restart Docker, restart containerd, prune Docker networks, stop unrelated containers, or reboot the host without explicit user approval.

## Secret Locations And Environment

Do not print tokens.

Blue-demon credentials:

```text
local file: ../blue-demon.txt
```

NVIDIA token:

```text
local file: /mnt/d/UserFolders/Documents/nvidia_toke.txt
remote env: /home/iarroyof/sabia/ai-research-insights/.env
env vars:
  NVIDIA_API_KEY
  CONTEXT_MANAGER_PROVIDER=nvidia
  NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1
  NVIDIA_MODEL=nvidia/llama-3.3-nemotron-super-49b-v1.5
```

Hugging Face token:

```text
local file: /mnt/d/UserFolders/Documents/hf_huggingface_token.txt
remote env: /home/iarroyof/sabia/ai-research-insights/.env
env vars:
  HF_API_TOKEN
  NLI_PROVIDER=hf_api
```

Compose already passes NVIDIA and HF variables to the API service.

## Verified Hosted Provider Behavior

### NVIDIA

Direct NVIDIA hosted API test succeeded without local GPU use.

Model:

```text
nvidia/llama-3.3-nemotron-super-49b-v1.5
```

Observed behavior:

- `max_tokens=64` produced empty final content with `finish_reason=length`, apparently because budget was spent in reasoning fields.
- `max_tokens=512` returned final content `OK`.
- Client was hardened to parse `nvapi-...` from prose-bearing token files and default NVIDIA policy calls to larger budget.

### Hugging Face BioNLI

Token and model verified without local GPU use.

Working endpoint:

```text
https://router.huggingface.co/hf-inference/models/pritamdeka/PubMedBERT-MNLI-MedNLI
```

Do not use the legacy endpoint:

```text
https://api-inference.huggingface.co/models/pritamdeka/PubMedBERT-MNLI-MedNLI
```

The legacy endpoint returned 404.

Hosted input format:

```text
hypothesis [SEP] premise
```

Sanity results:

```text
entailment fixture:
  entailment: 0.9987
  contradiction: 0.0013
  neutral: 0.0001

contradiction fixture:
  contradiction: 0.9978
  entailment: 0.0022
  neutral: 0.0
```

### Hugging Face Zero-Shot Classification

`app.services.zero_shot.score_labels()` now supports hosted HF API mode and defaults to it in hosted/no-GPU containers.

Runtime env:

```text
ZERO_SHOT_PROVIDER=hf_api
ZERO_SHOT_MODEL=facebook/bart-large-mnli
ZERO_SHOT_HF_API_BASE_URL=https://router.huggingface.co/hf-inference/models
ZERO_SHOT_HF_API_TIMEOUT_SEC=45
ZERO_SHOT_HF_API_MAX_RETRIES=2
ZERO_SHOT_HF_API_RETRY_BACKOFF_SEC=2.0
HF_API_TOKEN=<from .env>
```

Important behavior:

- The local `transformers` zero-shot path is lazy-loaded and only works in the full/local-ML image where `transformers` is installed.
- The hosted image intentionally omits `torch`, `transformers`, `sentence-transformers`, spaCy, SciSpaCy, and downloaded local model packages.
- HF router responses can be pipeline-style `{"labels": [...], "scores": [...]}` or element-style `[{"label": "...", "score": ...}]`; both shapes are now parsed.
- Retry/backoff is implemented for cold-start/rate-limit/server failures:
  - retryable status codes: `408`, `409`, `425`, `429`, `500`, `502`, `503`, `504`;
  - timeout/network errors retry up to `ZERO_SHOT_HF_API_MAX_RETRIES`;
  - `Retry-After` is honored when HF sends it.
- HF BioNLI has matching retry/backoff and in-process metrics.
- Diagnostics endpoint:

```http
GET /chat/memory/provider-metrics
Headers:
  X-Tenant-Id: default
  X-API-Key: <API_KEY>
```

It exposes counters only; prompts, responses, and secrets are not included.
- Real HF smoke passed from the hosted image:

```text
zero-shot-hf-ok [('biomedical', 0.8951265215873718), ('finance', 0.0184012558311224)]
```

## Mandatory Test Gate

No higher-level feature may depend on a newly implemented module until that module has:

1. Unit tests.
2. Mocked integration tests.
3. One real-provider smoke test where applicable.
4. Documented failure modes.
5. `docs/2026-05-14-status-tracker.md` updated.

Provider order:

```text
LLM/chat/context manager:
  NVIDIA hosted API first
  local models later, only after GPU approval

Biomedical NLI:
  Hugging Face API first
  local model fallback only
```

Heuristic NLI is development fallback only. Do not treat it as factuality authority.

## Current Implementation Summary

Implemented/partial code exists for:

- Chat endpoint pre-generation context policy hook.
- Chat endpoint post-generation reward/trace storage.
- Idea-index memory:
  - `services/api/app/memory/idea_index.py`;
  - stored as `doc_type=idea`;
  - used in `ContextPolicy.plan()`.
- Q-like action-value memory:
  - `services/api/app/memory/action_value.py`;
  - stored as `doc_type=action_value`;
  - updated in `ContextPolicy.observe_turn()`.
- Source-sentence factuality wiring in `ContextPolicy.observe_turn()`:
  - atomic answer claims;
  - evidence candidates from selected/pinned/source/triplet context;
  - claim support;
  - evidence table persistence;
  - optional chat debug SSE `evidence_table`.
- `services/api/app/memory/` package:
  - `policy.py`
  - `store.py`
  - `rewards.py`
  - `privacy.py`
  - `web_search.py`
  - `nli.py`
- OpenSearch-backed per-session memory.
- Basic landmarks.
- Triplet retrieval hook.
- DuckDuckGo redacted web search, off by default.
- User correction endpoint:

```http
POST /chat/memory/correction
```

- NVIDIA-compatible `LLMClient.chat_once()`.
- HF API provider in `memory/nli.py`.
- HF API provider in `services/api/app/services/zero_shot.py`.
- Hosted/no-GPU API image path using `services/api/Dockerfile.hosted`.

Do not mistake this for a complete RL/memory system. The scheduler is still deterministic. There is now a Q-like action-value table and a first idea-index, but they are not yet a learned policy or a full hierarchical concept tree. Compression/eviction/promotion and source-sentence factuality are partially wired, not production-complete.

## Current Highest-Priority Implementation

Start bottom-up. Do not begin with scheduler refactor.

### First Work Package

Implement the evidence-grounded factuality core:

1. `app.memory.claims`
   - split assistant answer into candidate sentences;
   - extract atomic factual claims;
   - preserve original answer sentence;
   - classify whether claim requires citation.

2. `app.memory.evidence`
   - define evidence candidate schema;
   - gather source/database sentences used in prompt;
   - gather pinned snippets;
   - gather triplet-linked source sentences;
   - preserve provenance: `paper_id`, `pmid`, `pmcid`, `title`, `section`, `sent_id`, `sentence_text`, `window_text`, `was_in_model_prompt`.

3. `app.memory.comparability`
   - gate premise/hypothesis pairs before NLI;
   - detect mismatches in disease, entity, relation, population/species/cell line, intervention/outcome, directionality, negation, temporality, and section context.

4. `app.memory.claim_support`
   - run HF NLI only for comparable source sentence vs answer claim pairs;
   - aggregate support status:
     - `entailed`
     - `contradicted`
     - `unsupported`
     - `not_comparable`
     - `not_checked`

5. Evidence table trace
   - store per-claim support/evidence table in OpenSearch;
   - add optional SSE payload for debugging/UI.

Only after these modules pass tests should they be wired into `ContextPolicy.observe_turn()`.

## Minimum Fixtures To Create

Create small tests for:

1. Entailment:

```text
premise: Aspirin inhibits platelet aggregation and is used to reduce thrombotic risk.
hypothesis: Aspirin inhibits platelet aggregation.
expected: entailment
```

2. Contradiction:

```text
premise: Aspirin inhibits platelet aggregation and is used to reduce thrombotic risk.
hypothesis: Aspirin does not inhibit platelet aggregation.
expected: contradiction
```

3. Neutral:

```text
premise: Aspirin inhibits platelet aggregation and is used to reduce thrombotic risk.
hypothesis: Metformin improves insulin sensitivity.
expected: neutral or not_comparable
```

4. Wrong evidence candidate:

```text
premise: Study reports platelet aggregation in cardiovascular disease.
hypothesis: A cancer biomarker predicts chemotherapy response.
expected: not_comparable before NLI
```

5. Unsupported answer claim:

```text
answer claim has no comparable source sentence.
expected: unsupported
```

## Correct Conceptual Design

Use:

```text
Triplets = retrieval expansion, graph structure, idea-index, candidate source selection, compact memory.
Source sentences = authoritative evidence.
Biomedical NLI = claim-vs-source factuality/reward.
```

Do not use triplets as the main truth judge. They are too lossy for negation, modality, population, causal direction, and uncertainty.

## Useful Commands

Check remote state:

```bash
sshpass -p '...' ssh -o StrictHostKeyChecking=no iarroyof@192.168.241.149 'cd sabia/ai-research-insights && git status --short && git diff --stat'
```

Non-GPU Compose validation:

```bash
sshpass -p '...' ssh -o StrictHostKeyChecking=no iarroyof@192.168.241.149 'cd sabia/ai-research-insights && docker compose config >/tmp/ai_research_compose_config.yml && echo compose-config-ok'
```

Python syntax check without bytecode:

```bash
sshpass -p '...' ssh -o StrictHostKeyChecking=no iarroyof@192.168.241.149 'cd sabia/ai-research-insights && python3 -c "import ast, pathlib; files=sorted(pathlib.Path(\"services/api/app/memory\").glob(\"*.py\")); [ast.parse(f.read_text(), filename=str(f)) for f in files]; print(\"syntax ok\", len(files))"'
```

## Open Questions

Ask the user before deciding:

1. Should factuality checking block final answers or only warn/lower reward?
2. Should evidence tables stream by default or only with debug flag?
3. Which disease-specific corpus should be used first for calibration?
4. Should a baseline commit be made before continuing implementation?

## Non-Omission Requirement

When reporting status, use:

- Implemented
- Partially implemented
- Not implemented
- Blocked
- Deferred

Do not silently omit incomplete pieces. Do not call scaffolding "implemented."
