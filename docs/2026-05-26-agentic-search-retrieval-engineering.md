# Agentic Search and Retrieval Engineering

This note describes the current hosted-mode chat retrieval system as a cooperating sequence of agents. It is intentionally domain-general: the logic must preserve user entities and requested relations for any biomedical question, not rely on topic-specific fixtures.

## Runtime Sequence

1. **Frame interpretation**
   - Owner: `SearchAgent`.
   - Input: current user message, session search notes, conversation frame, Q-like action-value hints.
   - Output: `search_frame`, query variants, state/action keys.
   - Constraints: remove task/meta terms, preserve biomedical anchors, canonicalize spelling variants, reuse prior frame for follow-up commands.

2. **Local multi-level retrieval**
   - Owner: `SearchAgent`.
   - Levels: title, sentence, paper.
   - Output: snippets with rank, BM25/retrieval score, source sentence IDs, query labels, disease/mechanism/evidence tags.
   - Guard: reject hits missing required entity anchors; do not let incidental high-BM25 terms pollute later queries.

3. **Evidence puzzle assembly**
   - Owner: `EvidenceAssembly`.
   - Output: covered nodes, missing nodes, relation evidence count, edge support status, clarification recommendation.
   - Contract: snippets are evidence pieces, not a completed causal chain.

4. **Session memory retrieval**
   - Owner: `MemoryAgent` through `ContextPolicy`.
   - Inputs: recent turns, episodic summaries, landmarks, ideas, triplets, conversation frame.
   - Output: working context prefix and native chat-history messages.
   - Contract: memory can steer framing, but unsupported prior claims must not become facts.

5. **External biomedical grounding**
   - Owner: `ContextPolicy`.
   - Sources: PubMed/PMC, PubTator, LitSense when available, DuckDuckGo only as a final sparse fallback.
   - Output: ranked external snippets with external query provenance and anchor coverage.
   - Deepening: discovered PubMed/PubTator PMCID results can trigger PMC full-text sentence extraction.
   - Guard: rank by full entity + process anchor coverage before partial matches.

6. **Answer policy**
   - Owner: `AnswerPolicyAgent`.
   - Output: answer mode and mode contract.
   - Modes include direct answer, novice rewrite, expert mechanism, phrase evaluation, correction acknowledgement, diagnostic trace answer, and clarification.
   - External grounding can override a local clarification hold only when it covers the missing puzzle nodes.

7. **Response generation**
   - Owner: hosted chat LLM.
   - Input: system policy, retrieval context, memory context, evidence assembly prompt, answer-mode contract.
   - Failure behavior: provider HTTP failures are converted into warnings plus an evidence-only fallback instead of breaking the SSE stream.

8. **Post-generation verification**
   - Owner: `PostGenerationVerifier`.
   - Output: accepted/repaired/blocked claims for guarded modes.
   - Contract: generated claims must not be more specific than supported puzzle edges.

9. **Memory observation and reward telemetry**
   - Owner: `ContextPolicy.observe_turn()`.
   - Output: claim support, evidence table, reward, consistency warnings, conversation frame updates, search notes, Q-like action-value telemetry.
   - Contract: telemetry informs later retrieval/search policy without training model weights.

## Domain Coupling Points

The retrieval logic is intended to be domain-general, but some vocabulary sets are intentionally domain-calibrated. In `services/api/app/memory/search_agent.py`, `BROAD_DOMAIN_ANCHORS` and `PROCESS_ANCHORS` currently contain biomedical/cancer-biology terms. They tell the anchor coverage gate which words are umbrella/process terms and therefore should not satisfy an entity-specific query by themselves.

This is a coupling point, not a bug. Before reusing the same architecture in another domain, expand or replace those sets with the domain's broad process vocabulary. Examples: climate workflows need terms like forcing, warming, emissions, adaptation, and mitigation; legal workflows need terms like liability, jurisdiction, claim, statute, and remedy; economics workflows need terms like inflation, demand, productivity, labor, and monetary policy. Entity/acronym aliases should remain separate from broad/process terms.

## User-Facing Diagnostics

Each chat response now includes `retrieval_pipeline` in citations. The sequence exposes the cooperating process outputs so a developer can see whether a failure came from query framing, local retrieval, evidence assembly, external grounding, answer policy, provider generation, or post-generation verification.

## Current Guardrails

- Do not use raw `docker compose`; use `scripts/compose-hosted.sh`.
- GPU-profile services remain off in hosted mode.
- BM25/local retrieval remains primary; external PubMed/PMC/PubTator/LitSense grounding is a fallback/deepening path.
- Vector search and learned RL weights remain deferred.
