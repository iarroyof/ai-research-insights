from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List

from app.clients.llm import LLMClient
from app.config import settings
from app.memory.action_value import action_key, state_key
from app.memory.claim_support import assess_claim_support, build_evidence_table, evidence_table_debug_payload
from app.memory.claims import extract_atomic_claims
from app.memory.consistency import longitudinal_consistency_report, render_conversation_frame
from app.memory.evidence import evidence_to_dicts, gather_evidence_candidates
from app.memory.nli import score_answer_triples
from app.memory.rewards import detect_triplet_conflicts, important_terms, reward_report, terms
from app.memory.store import MemoryStore
from app.memory.web_search import duckduckgo_search, litsense2_search, pubmed_pmc_search, pubtator3_search

try:
    from app.integrations.extraction_client import extract_triples
except Exception:
    extract_triples = None

try:
    from app.triplets.search import search_triplets
except Exception:
    search_triplets = None


@dataclass
class ContextPlan:
    turn_index: int
    context_prefix: str
    selected_context: List[Dict[str, Any]] = field(default_factory=list)
    retrieved_triplets: List[Dict[str, Any]] = field(default_factory=list)
    web_results: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)


def _render_recent(messages: list[dict]) -> str:
    if not messages:
        return ""
    lines = ["Recent conversation buffer:"]
    for m in messages:
        role = m.get("role", "message")
        text = (m.get("summary") or m.get("text") or "").strip()
        if text:
            lines.append(f"- {role}: {text[:900]}")
    return "\n".join(lines)


def _render_memory(hits: list[dict]) -> str:
    if not hits:
        return ""
    lines = ["Retrieved session memory:"]
    for h in hits:
        role = h.get("role", "memory")
        text = (h.get("summary") or h.get("text") or "").strip()
        score = h.get("_score")
        score_text = f" score={score:.3f}" if isinstance(score, (int, float)) else ""
        lines.append(f"- {role}{score_text}: {text[:700]}")
    return "\n".join(lines)


def _render_landmarks(items: list[dict]) -> str:
    if not items:
        return ""
    lines = ["Conversation landmarks:"]
    for item in items[:8]:
        name = item.get("name", "landmark")
        summary = (item.get("summary") or "").strip()
        if summary:
            lines.append(f"- {name}: {summary[:500]}")
    return "\n".join(lines)


def _render_summaries(items: list[dict]) -> str:
    if not items:
        return ""
    lines = ["Episodic session summaries:"]
    for item in items[:4]:
        summary = (item.get("summary") or item.get("text") or "").strip()
        if summary:
            lines.append(f"- {summary[:700]}")
    return "\n".join(lines)


def _render_triplets(triplets: list[dict]) -> str:
    if not triplets:
        return ""
    lines = ["Relevant semantic triplets:"]
    for t in triplets[: settings.memory.triplet_k]:
        subj = t.get("subject") or ""
        rel = t.get("relation") or t.get("predicate") or ""
        obj = t.get("object") or ""
        sent = t.get("sentence_text") or t.get("text") or ""
        if subj or obj:
            lines.append(f"- ({subj}; {rel}; {obj}) {sent[:280]}")
    return "\n".join(lines)


def _render_web(results: list[dict]) -> str:
    if not results:
        return ""
    lines = ["Privacy-filtered external grounding:"]
    for r in results[: settings.memory.web_k]:
        source = r.get("source") or "web"
        title = r.get("title") or "web result"
        snippet = r.get("snippet") or ""
        url = r.get("url") or ""
        lines.append(f"- {source} | {title}: {snippet[:350]} {url}".strip())
    return "\n".join(lines)


def _external_result_key(result: dict) -> str:
    return str(result.get("pmid") or result.get("pmcid") or result.get("url") or result.get("title") or "")


def _rank_external_results(query: str, results: list[dict]) -> list[dict]:
    query_terms = set(important_terms(query, 48))
    if not query_terms:
        return results

    def score(result: dict) -> tuple[float, float]:
        text_terms = set(important_terms(f"{result.get('title') or ''} {result.get('snippet') or ''}", 140))
        overlap = len(query_terms & text_terms) / max(1, len(query_terms))
        semantic = 0.1 if str(result.get("source") or "").startswith("litsense2") else 0.0
        provider_score = float(result.get("score", 0.0) or 0.0)
        return overlap + semantic, provider_score

    ranked = [dict(result) for result in results]
    ranked.sort(key=score, reverse=True)
    for result in ranked:
        result.setdefault("external_rank_score", round(score(result)[0], 4))
    return ranked


def _merge_external_results(
    pubmed_results: list[dict],
    pubtator_results: list[dict],
    k: int,
    litsense_results: list[dict] | None = None,
    query: str = "",
) -> list[dict]:
    litsense_results = litsense_results or []
    pubmed_results = _rank_external_results(query, pubmed_results)
    pubtator_results = _rank_external_results(query, pubtator_results)
    litsense_results = _rank_external_results(query, litsense_results)
    auxiliary_result_sets = [items for items in (litsense_results, pubtator_results) if items]
    reserved_auxiliary_slots = min(len(auxiliary_result_sets), max(0, k - (1 if pubmed_results else 0)))
    keep_pubmed = max(0, k - reserved_auxiliary_slots)
    candidates = [
        *pubmed_results[:keep_pubmed],
        *litsense_results,
        *pubtator_results,
        *pubmed_results[keep_pubmed:],
    ]
    merged: list[dict] = []
    seen: set[str] = set()
    for result in candidates:
        key = _external_result_key(result)
        if key and key in seen:
            continue
        if key:
            seen.add(key)
        merged.append(result)
        if len(merged) >= k:
            return merged
    return merged


def _render_ideas(ideas: list[dict]) -> str:
    if not ideas:
        return ""
    lines = ["High-value recurring ideas:"]
    for item in ideas[:8]:
        idea = item.get("idea")
        freq = item.get("frequency", item.get("session_frequency", 0))
        reward = item.get("reward_avg")
        if idea:
            reward_text = f" reward={float(reward):.2f}" if isinstance(reward, (int, float)) else ""
            lines.append(f"- {idea} freq={freq}{reward_text}")
    return "\n".join(lines)


def _policy_instruction() -> str:
    return (
        "Memory policy guidance:\n"
        "- Treat the context below as an OS-like working set selected from recent turns, indexed memory, landmarks, triplets, and optional web grounding.\n"
        "- Prefer facts supported by pinned snippets, retrieved triplets, or explicit user statements.\n"
        "- If a warning says facts may be inconsistent, briefly mention the uncertainty and ask the user which fact should be treated as authoritative.\n"
        "- Do not expose hidden reward scores or policy internals unless the user asks for diagnostics."
    )


class ContextPolicy:
    def __init__(self, tenant: str):
        self.tenant = tenant
        self.store = MemoryStore(tenant)

    async def plan(
        self,
        *,
        session_id: str,
        message: str,
        allow_web_search: bool,
        confidence_min: float,
    ) -> ContextPlan:
        turn_index = await self.store.next_turn_index(session_id)
        working_token_budget = min(
            settings.memory.working_buffer_token_budget,
            max(256, int(settings.llm.max_input_tokens * settings.memory.token_budget_ratio)),
        )
        recent = await self.store.recent_messages(
            session_id,
            settings.memory.working_buffer_turns,
            token_budget=working_token_budget,
            query_text=message,
        )
        memory_hits = await self.store.search_memory(session_id, message, settings.memory.memory_k)
        landmarks = await self.store.landmarks(session_id)
        summaries = await self.store.episodic_summaries(session_id, 3)
        latest_traces = await self.store.latest_traces(session_id, 3)
        conversation_frame = await self.store.conversation_frame(session_id)
        idea_hits = await self.store.search_ideas(session_id, message, min(8, settings.memory.memory_k))
        state = state_key(important_terms(message))
        action_value_hints = await self.store.action_values(session_id, state, 3)

        triplets: list[dict] = []
        if search_triplets:
            try:
                triplets = await search_triplets(
                    self.tenant,
                    message,
                    confidence_min=confidence_min,
                )
            except Exception as e:
                print(f"[WARN] ContextPolicy triplet search failed: {e}")

        web_payload = {"results": [], "query": "", "redacted": False}
        local_sparse = len(memory_hits) + len(triplets) < 3
        if allow_web_search and local_sparse:
            pubmed_payload = {"results": [], "query": "", "redacted": False}
            pubtator_payload = {"results": [], "query": "", "redacted": False}
            litsense_payload = {"results": [], "query": "", "redacted": False}
            try:
                pubmed_payload = await pubmed_pmc_search(message, settings.memory.web_k)
            except Exception as e:
                print(f"[WARN] ContextPolicy PubMed/PMC search failed: {e}")
            try:
                pubtator_payload = await pubtator3_search(message, settings.memory.web_k)
            except Exception as e:
                print(f"[WARN] ContextPolicy PubTator 3 search failed: {e}")
            try:
                litsense_payload = await litsense2_search(message, settings.memory.web_k)
            except Exception as e:
                print(f"[WARN] ContextPolicy LitSense 2.0 search failed: {e}")

            web_payload = (
                pubmed_payload
                if pubmed_payload.get("results")
                else litsense_payload
                if litsense_payload.get("results")
                else pubtator_payload
            )
            web_payload["results"] = _merge_external_results(
                pubmed_payload.get("results") or [],
                pubtator_payload.get("results") or [],
                settings.memory.web_k,
                litsense_payload.get("results") or [],
                message,
            )
            if not web_payload.get("results"):
                try:
                    web_payload = await duckduckgo_search(message, settings.memory.web_k)
                except Exception as e:
                    print(f"[WARN] ContextPolicy DuckDuckGo search failed: {e}")

        warnings: list[str] = []
        if any(w in set(terms(message)) for w in ("not", "never", "without", "contradict", "conflict")) and triplets:
            warnings.append(
                "The current question may negate or challenge facts found in retrieved triplets. Treat the answer as potentially inconsistent unless the evidence resolves it."
            )

        sections = [
            _policy_instruction(),
            render_conversation_frame(conversation_frame),
            _render_landmarks(landmarks),
            _render_summaries(summaries),
            _render_recent(recent),
            _render_ideas(idea_hits),
            _render_memory(memory_hits),
            _render_triplets(triplets),
            _render_web(web_payload.get("results") or []),
        ]
        if latest_traces:
            reflection_lines = [
                t.get("reflection")
                for t in latest_traces
                if isinstance(t.get("reflection"), str) and t.get("reflection")
            ]
            if reflection_lines:
                sections.append("Recent policy reflections:\n" + "\n".join(f"- {r[:500]}" for r in reflection_lines[:3]))

        prefix = "\n\n".join(s for s in sections if s)
        selected_context = []
        selected_context.extend({"source": "recent", **m} for m in recent)
        selected_context.extend({"source": "episodic_summary", "text": m.get("summary") or m.get("text") or "", **m} for m in summaries)
        selected_context.extend({"source": "idea", "text": str(m.get("idea") or ""), **m} for m in idea_hits)
        selected_context.extend({"source": "memory", **m} for m in memory_hits)
        selected_context.extend({"source": "triplet", "text": t.get("sentence_text") or t.get("text") or "", **t} for t in triplets)
        selected_context.extend({"source": "web", "text": r.get("snippet") or "", **r} for r in web_payload.get("results") or [])

        return ContextPlan(
            turn_index=turn_index,
            context_prefix=prefix,
            selected_context=selected_context,
            retrieved_triplets=triplets,
            web_results=web_payload.get("results") or [],
            warnings=warnings,
            meta={
                "turn_index": turn_index,
                "recent_count": len(recent),
                "recent_token_count": sum(int(m.get("token_count", 0) or 0) for m in recent),
                "working_token_budget": working_token_budget,
                "episodic_summary_count": len(summaries),
                "memory_hit_count": len(memory_hits),
                "idea_count": len(idea_hits),
                "action_value_hint_count": len(action_value_hints),
                "conversation_frame_terms": conversation_frame.get("active_terms", [])[:12] if conversation_frame else [],
                "conversation_frame_avoid_terms": conversation_frame.get("avoided_terms", [])[:12] if conversation_frame else [],
                "triplet_count": len(triplets),
                "web_result_count": len(web_payload.get("results") or []),
                "web_query_redacted": web_payload.get("redacted", False),
                "web_query": web_payload.get("query", ""),
            },
        )

    async def observe_turn(
        self,
        *,
        session_id: str,
        turn_index: int,
        question: str,
        answer: str,
        selected_context: List[Dict[str, Any]],
        retrieved_triplets: List[Dict[str, Any]],
        pinned_snippets: List[Dict[str, Any]] | None = None,
        source_sentences: List[Dict[str, Any]] | None = None,
        search_plan: Dict[str, Any] | None = None,
        started_at: float,
        token_budget: int,
    ) -> Dict[str, Any]:
        answer_triples: list[dict] = []
        if extract_triples:
            try:
                result = await extract_triples([answer[:1800]], timeout_sec=60, num_extractions=6)
                answer_triples = result.get("triples", []) or []
            except Exception as e:
                print(f"[WARN] ContextPolicy answer triplet extraction failed: {e}")

        conflicts = detect_triplet_conflicts(
            answer_triples,
            retrieved_triplets,
            threshold=settings.memory.contradiction_threshold,
        )
        nli_evidence = await score_answer_triples(answer_triples, retrieved_triplets)
        claims = extract_atomic_claims(answer)
        prior_frame = await self.store.conversation_frame(session_id)
        prior_supported_claims = await self.store.supported_claim_evidence(session_id, 24)
        source_items = list(source_sentences or [])
        source_items.extend(prior_supported_claims)
        evidence_candidates = gather_evidence_candidates(
            prompt_context=selected_context,
            pinned_snippets=pinned_snippets or [],
            source_sentences=source_items,
            triplet_results=retrieved_triplets,
        )
        claim_support = await assess_claim_support(
            claims,
            evidence_candidates,
            max_nli_pairs_per_claim=8,
        )
        answer_id = f"answer_{session_id}_{turn_index + 1}"
        evidence_table = build_evidence_table(
            answer_id=answer_id,
            session_id=session_id,
            turn_index=turn_index,
            claim_support=claim_support,
            tenant=self.tenant,
        )
        claim_support_dicts = [item.to_dict() for item in claim_support]
        longitudinal = longitudinal_consistency_report(
            question=question,
            answer=answer,
            claim_support=claim_support_dicts,
            prior_supported_claims=prior_supported_claims,
            frame=prior_frame,
        )
        reward = reward_report(
            question=question,
            answer=answer,
            selected_context=selected_context,
            conflicts=conflicts,
            nli_evidence=nli_evidence,
            claim_support=claim_support_dicts,
            longitudinal_consistency=longitudinal,
            search_plan=search_plan,
            elapsed_sec=max(0.0, time.monotonic() - started_at),
            token_budget=token_budget,
        )

        await self.store.add_message(
            session_id=session_id,
            role="user",
            text=question,
            turn_index=turn_index,
            importance=0.65,
        )
        await self.store.add_message(
            session_id=session_id,
            role="assistant",
            text=answer,
            turn_index=turn_index + 1,
            triples=answer_triples,
            importance=0.5 + 0.5 * reward["score"],
        )
        await self.store.add_episodic_summary(
            session_id=session_id,
            turn_index=turn_index + 1,
            messages=[
                {
                    "role": "user",
                    "text": question,
                    "turn_index": turn_index,
                    "importance": 0.65,
                },
                {
                    "role": "assistant",
                    "text": answer,
                    "turn_index": turn_index + 1,
                    "importance": 0.5 + 0.5 * reward["score"],
                    "claim_support": claim_support_dicts,
                    "evidence_supported": any(item.status == "entailed" for item in claim_support),
                },
            ],
            reward_score=float(reward["score"]),
        )
        updated_frame = await self.store.update_conversation_frame(
            session_id=session_id,
            question=question,
            answer=answer,
            claim_support=claim_support_dicts,
            turn_index=turn_index + 1,
        )
        await self.store.update_memory_lifecycle(session_id=session_id, current_turn_index=turn_index + 1)
        await self.store.update_landmarks(session_id, question, reward)
        await self.store.add_evidence_table(
            session_id=session_id,
            turn_index=turn_index,
            answer_id=answer_id,
            evidence_table=evidence_table,
        )

        reflection = ""
        if settings.memory.use_llm_reflection:
            reflection = await self._reflect(question, answer, reward, conflicts)
        state = state_key(important_terms(question))
        action = {
            "selected_context_count": len(selected_context),
            "retrieved_triplet_count": len(retrieved_triplets),
            "selected_idea_count": sum(1 for item in selected_context if item.get("source") == "idea"),
            "web_result_count": sum(1 for item in selected_context if item.get("source") == "web"),
            "evidence_candidate_count": len(evidence_candidates),
            "auto_context_result_count": int((search_plan or {}).get("result_count", 0) or 0),
            "search_query_count": len((search_plan or {}).get("variants", []) or []),
            "search_level_count": len((search_plan or {}).get("levels", []) or []),
            "search_used_llm": bool((search_plan or {}).get("used_llm", False)),
        }
        action = {**action, "action_key": action_key(action)}
        await self.store.update_idea_index(
            session_id=session_id,
            texts=[question, answer],
            turn_index=turn_index,
            reward_score=float(reward["score"]),
            shared=bool(settings.memory.shared_policy_enabled),
        )
        await self.store.update_action_value(
            session_id=session_id,
            state_key=state,
            action_key=action["action_key"],
            reward_score=float(reward["score"]),
            shared=bool(settings.memory.shared_policy_enabled),
        )
        if search_plan and search_plan.get("state_key") and search_plan.get("action_key"):
            await self.store.update_action_value(
                session_id=session_id,
                state_key=str(search_plan["state_key"]),
                action_key=str(search_plan["action_key"]),
                reward_score=float(reward["score"]),
                shared=bool(settings.memory.shared_policy_enabled),
            )
            note_parts = [
                str(search_plan.get("planner_note") or "").strip(),
                str(search_plan.get("note") or "").strip(),
            ]
            note = " ".join(part for part in note_parts if part).strip()
            if note:
                await self.store.add_search_policy_note(
                    session_id=session_id,
                    turn_index=turn_index,
                    note=note,
                    search_plan=search_plan,
                    reward_score=float(reward["score"]),
                )

        trace = {
            "state_terms": important_terms(question),
            "state_key": state,
            "action": action,
            "search_plan": search_plan or {},
            "reward": reward,
            "conflicts": conflicts,
            "nli_evidence": nli_evidence,
            "claim_support": claim_support_dicts,
            "longitudinal_consistency": longitudinal,
            "conversation_frame": {
                "summary": updated_frame.get("summary", ""),
                "active_terms": updated_frame.get("active_terms", [])[:16],
                "avoided_terms": updated_frame.get("avoided_terms", [])[:16],
                "supported_claim_count": len(updated_frame.get("supported_claims") or []),
                "contradicted_claim_count": len(updated_frame.get("contradicted_claims") or []),
            },
            "evidence_candidates": evidence_to_dicts(evidence_candidates),
            "evidence_table": evidence_table_debug_payload(evidence_table),
            "reflection": reflection,
            "answer_triple_count": len(answer_triples),
        }
        if settings.memory.reward_trace_enabled:
            await self.store.add_trace(session_id=session_id, turn_index=turn_index, trace=trace)
        return trace

    async def _reflect(self, question: str, answer: str, reward: dict, conflicts: list[dict]) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "You write one concise Reflexion-style memory note for a chatbot context policy. "
                    "Say what to retrieve or avoid next time. Do not include hidden chain-of-thought."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question: {question[:1000]}\n"
                    f"Answer: {answer[:1000]}\n"
                    f"Reward: {reward}\n"
                    f"Conflicts: {len(conflicts)}"
                ),
            },
        ]
        try:
            text = await LLMClient().chat_once(
                messages,
                provider=settings.llm.context_manager_provider,
                max_tokens=160,
            )
            return " ".join(text.split())[:700]
        except Exception as e:
            print(f"[WARN] ContextPolicy reflection failed: {e}")
            return ""
