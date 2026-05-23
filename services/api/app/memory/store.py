from __future__ import annotations

import asyncio
import hashlib
from datetime import datetime, timezone
from typing import Any, Dict, List

from app.config import settings
from app.memory.action_value import action_id, update_action_value
from app.memory.consistency import build_conversation_frame
from app.memory.idea_index import build_idea_updates, merge_idea_doc, rank_ideas
from app.memory.lifecycle import build_episodic_summary, classify_memory_state, select_working_set
from app.memory.rewards import important_terms
from app.search.os_client import os_client


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _stable_id(*parts: str) -> str:
    data = "\n".join(parts).encode("utf-8")
    return hashlib.sha256(data).hexdigest()[:32]


class MemoryStore:
    def __init__(self, tenant: str):
        self.tenant = tenant
        prefix = getattr(settings.opensearch, "index_prefix", "") or ""
        self.index = f"{prefix}{tenant}_chat_memory"

    async def _run(self, fn, *args, **kwargs):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: fn(*args, **kwargs))

    def _client(self):
        return os_client()

    async def ensure_index(self) -> None:
        def sync():
            client = self._client()
            if client.indices.exists(index=self.index):
                return
            client.indices.create(
                index=self.index,
                body={
                    "mappings": {
                        "properties": {
                            "doc_type": {"type": "keyword"},
                            "tenant": {"type": "keyword"},
                            "session_id": {"type": "keyword"},
                            "role": {"type": "keyword"},
                            "turn_index": {"type": "integer"},
                            "created_at": {"type": "date"},
                            "updated_at": {"type": "date"},
                            "terms": {"type": "keyword"},
                            "importance": {"type": "float"},
                            "text": {"type": "text"},
                            "summary": {"type": "text"},
                            "memory_state": {"type": "keyword"},
                            "memory_state_reason": {"type": "keyword"},
                            "token_count": {"type": "integer"},
                            "source_turn_start": {"type": "integer"},
                            "source_turn_end": {"type": "integer"},
                            "answer_id": {"type": "keyword"},
                            "evidence_table_id": {"type": "keyword"},
                            "claims.status": {"type": "keyword"},
                            "claims.claim_id": {"type": "keyword"},
                            "claims.best_evidence_id": {"type": "keyword"},
                            "idea_id": {"type": "keyword"},
                            "idea": {"type": "keyword"},
                            "normalized_idea": {"type": "keyword"},
                            "parent_idea": {"type": "keyword"},
                            "child_ideas": {"type": "keyword"},
                            "synonyms": {"type": "keyword"},
                            "concept_path": {"type": "keyword"},
                            "scope": {"type": "keyword"},
                            "frequency": {"type": "integer"},
                            "session_frequency": {"type": "integer"},
                            "reward_avg": {"type": "float"},
                            "cooccurring_ideas": {"type": "keyword"},
                            "action_id": {"type": "keyword"},
                            "state_key": {"type": "keyword"},
                            "action_key": {"type": "keyword"},
                            "q_value": {"type": "float"},
                            "visits": {"type": "integer"},
                            "search_state_key": {"type": "keyword"},
                            "search_action_key": {"type": "keyword"},
                            "search_strategy": {"type": "keyword"},
                            "search_queries": {"type": "keyword"},
                            "frame_id": {"type": "keyword"},
                            "active_terms": {"type": "keyword"},
                            "avoided_terms": {"type": "keyword"},
                            "supported_claims.claim": {"type": "text"},
                            "contradicted_claims.claim": {"type": "text"},
                        }
                    }
                },
            )

        try:
            await self._run(sync)
        except Exception as e:
            print(f"[WARN] MemoryStore.ensure_index failed: {e}")

    async def next_turn_index(self, session_id: str) -> int:
        def sync() -> int:
            body = {
                "size": 1,
                "query": {
                    "bool": {
                        "filter": [
                            {"term": {"doc_type": "message"}},
                            {"term": {"session_id": session_id}},
                        ]
                    }
                },
                "sort": [{"turn_index": {"order": "desc"}}],
            }
            res = self._client().search(index=self.index, body=body)
            hits = res.get("hits", {}).get("hits", [])
            if not hits:
                return 0
            return int(hits[0].get("_source", {}).get("turn_index", 0)) + 1

        try:
            return await self._run(sync)
        except Exception:
            return 0

    async def add_message(
        self,
        *,
        session_id: str,
        role: str,
        text: str,
        turn_index: int,
        summary: str | None = None,
        triples: list[dict] | None = None,
        importance: float = 0.5,
    ) -> None:
        await self.ensure_index()
        doc = {
            "doc_type": "message",
            "tenant": self.tenant,
            "session_id": session_id,
            "role": role,
            "turn_index": turn_index,
            "text": text,
            "summary": summary or text[:500],
            "triples": triples or [],
            "terms": important_terms(text),
            "importance": importance,
            "memory_state": "working",
            "created_at": _now(),
            "updated_at": _now(),
        }
        doc_id = _stable_id(self.tenant, session_id, str(turn_index), role, text[:80])
        try:
            await self._run(self._client().index, index=self.index, id=doc_id, body=doc, refresh=False)
        except Exception as e:
            print(f"[WARN] MemoryStore.add_message failed: {e}")

    async def recent_messages(self, session_id: str, limit: int, token_budget: int | None = None, query_text: str = "") -> List[Dict[str, Any]]:
        def sync() -> List[Dict[str, Any]]:
            body = {
                "size": max(limit, limit * 4),
                "query": {
                    "bool": {
                        "filter": [
                            {"term": {"doc_type": "message"}},
                            {"term": {"session_id": session_id}},
                        ]
                    }
                },
                "sort": [{"turn_index": {"order": "desc"}}],
            }
            res = self._client().search(index=self.index, body=body)
            hits = [h.get("_source", {}) for h in res.get("hits", {}).get("hits", [])]
            chronological = list(reversed(hits))
            if token_budget:
                selection = select_working_set(
                    chronological,
                    token_budget=token_budget,
                    working_turns=settings.memory.working_buffer_turns,
                    eviction_importance_threshold=settings.memory.eviction_importance_threshold,
                    query_text=query_text,
                )
                return selection.selected
            return chronological[-limit:]

        try:
            return await self._run(sync)
        except Exception:
            return []

    async def episodic_summaries(self, session_id: str, limit: int = 3) -> List[Dict[str, Any]]:
        def sync() -> List[Dict[str, Any]]:
            body = {
                "size": limit,
                "query": {
                    "bool": {
                        "filter": [
                            {"term": {"doc_type": "episodic_summary"}},
                            {"term": {"session_id": session_id}},
                        ]
                    }
                },
                "sort": [{"turn_index": {"order": "desc"}}],
            }
            res = self._client().search(index=self.index, body=body)
            return [h.get("_source", {}) for h in res.get("hits", {}).get("hits", [])]

        try:
            return await self._run(sync)
        except Exception:
            return []

    async def add_episodic_summary(
        self,
        *,
        session_id: str,
        turn_index: int,
        messages: List[Dict[str, Any]],
        reward_score: float = 0.0,
    ) -> None:
        await self.ensure_index()
        doc = build_episodic_summary(
            tenant=self.tenant,
            session_id=session_id,
            turn_index=turn_index,
            messages=messages,
            reward_score=reward_score,
        )
        now = _now()
        doc["created_at"] = now
        doc["updated_at"] = now
        doc_id = doc.get("summary_id") or _stable_id(self.tenant, session_id, "episodic", str(turn_index))
        try:
            await self._run(self._client().index, index=self.index, id=doc_id, body=doc, refresh=False)
        except Exception as e:
            print(f"[WARN] MemoryStore.add_episodic_summary failed: {e}")

    async def update_memory_lifecycle(self, *, session_id: str, current_turn_index: int) -> None:
        await self.ensure_index()

        def sync() -> None:
            body = {
                "size": settings.memory.lifecycle_update_k,
                "query": {
                    "bool": {
                        "filter": [
                            {"terms": {"doc_type": ["message", "episodic_summary", "landmark", "user_correction"]}},
                            {"term": {"session_id": session_id}},
                        ]
                    }
                },
                "sort": [{"turn_index": {"order": "desc"}}],
            }
            client = self._client()
            res = client.search(index=self.index, body=body)
            for hit in res.get("hits", {}).get("hits", []):
                src = hit.get("_source", {})
                state = classify_memory_state(
                    src,
                    current_turn_index=current_turn_index,
                    working_turns=settings.memory.working_buffer_turns,
                    eviction_importance_threshold=settings.memory.eviction_importance_threshold,
                )
                client.update(
                    index=self.index,
                    id=hit.get("_id"),
                    body={
                        "doc": {
                            "memory_state": state.memory_state,
                            "memory_state_reason": state.reason,
                            "token_count": state.token_count,
                            "updated_at": _now(),
                        }
                    },
                    refresh=False,
                )

        try:
            await self._run(sync)
        except Exception as e:
            print(f"[WARN] MemoryStore.update_memory_lifecycle failed: {e}")

    async def search_memory(self, session_id: str, query: str, k: int) -> List[Dict[str, Any]]:
        def sync() -> List[Dict[str, Any]]:
            body = {
                "size": k,
                "query": {
                    "bool": {
                        "must": [
                            {
                                "multi_match": {
                                    "query": query,
                                    "fields": ["text^2", "summary^2", "terms^3", "triples.subject", "triples.object"],
                                }
                            }
                        ],
                        "filter": [
                            {"term": {"doc_type": "message"}},
                            {"term": {"session_id": session_id}},
                        ],
                        "must_not": [{"term": {"memory_state": "evicted"}}],
                    }
                },
            }
            res = self._client().search(index=self.index, body=body)
            out: list[dict] = []
            for h in res.get("hits", {}).get("hits", []):
                src = h.get("_source", {})
                src["_score"] = h.get("_score", 0.0)
                out.append(src)
            return out

        try:
            return await self._run(sync)
        except Exception:
            return []

    async def landmarks(self, session_id: str) -> List[Dict[str, Any]]:
        def sync() -> List[Dict[str, Any]]:
            body = {
                "size": 20,
                "query": {
                    "bool": {
                        "filter": [
                            {"term": {"doc_type": "landmark"}},
                            {"term": {"session_id": session_id}},
                        ]
                    }
                },
                "sort": [{"updated_at": {"order": "desc"}}],
            }
            res = self._client().search(index=self.index, body=body)
            return [h.get("_source", {}) for h in res.get("hits", {}).get("hits", [])]

        try:
            return await self._run(sync)
        except Exception:
            return []

    async def update_landmarks(self, session_id: str, message: str, reward: dict | None = None) -> None:
        await self.ensure_index()
        focus_terms = important_terms(message, 12)
        docs = [
            {
                "name": "current_focus",
                "summary": ", ".join(focus_terms) if focus_terms else message[:180],
                "terms": focus_terms,
            }
        ]
        if "?" in message:
            docs.append({"name": "open_question", "summary": message[:500], "terms": focus_terms})
        if reward:
            docs.append(
                {
                    "name": "latest_reward_state",
                    "summary": f"score={reward.get('score')} support={reward.get('context_support')} conflicts={reward.get('triplet_conflict_penalty')}",
                    "terms": ["reward", "context", "consistency"],
                }
            )

        for item in docs:
            doc = {
                "doc_type": "landmark",
                "tenant": self.tenant,
                "session_id": session_id,
                "name": item["name"],
                "summary": item["summary"],
                "terms": item["terms"],
                "created_at": _now(),
                "updated_at": _now(),
            }
            doc_id = _stable_id(self.tenant, session_id, "landmark", item["name"])
            try:
                await self._run(self._client().index, index=self.index, id=doc_id, body=doc, refresh=False)
            except Exception as e:
                print(f"[WARN] MemoryStore.update_landmarks failed: {e}")

    async def add_trace(self, *, session_id: str, turn_index: int, trace: Dict[str, Any]) -> None:
        await self.ensure_index()
        doc = {
            "doc_type": "policy_trace",
            "tenant": self.tenant,
            "session_id": session_id,
            "turn_index": turn_index,
            **trace,
            "created_at": _now(),
            "updated_at": _now(),
        }
        doc_id = _stable_id(self.tenant, session_id, "trace", str(turn_index))
        try:
            await self._run(self._client().index, index=self.index, id=doc_id, body=doc, refresh=False)
        except Exception as e:
            print(f"[WARN] MemoryStore.add_trace failed: {e}")

    async def add_evidence_table(
        self,
        *,
        session_id: str,
        turn_index: int,
        answer_id: str,
        evidence_table: Dict[str, Any],
    ) -> None:
        await self.ensure_index()
        doc = {
            **evidence_table,
            "doc_type": "evidence_table",
            "tenant": self.tenant,
            "session_id": session_id,
            "turn_index": turn_index,
            "answer_id": answer_id,
            "updated_at": _now(),
        }
        doc_id = doc.get("evidence_table_id") or _stable_id(self.tenant, session_id, "evidence_table", str(turn_index), answer_id)
        doc["evidence_table_id"] = doc_id
        try:
            await self._run(self._client().index, index=self.index, id=doc_id, body=doc, refresh=False)
        except Exception as e:
            print(f"[WARN] MemoryStore.add_evidence_table failed: {e}")

    async def evidence_tables(self, session_id: str | None = None, limit: int = 10) -> List[Dict[str, Any]]:
        def sync() -> List[Dict[str, Any]]:
            filters: list[dict[str, Any]] = [{"term": {"doc_type": "evidence_table"}}]
            if session_id:
                filters.append({"term": {"session_id": session_id}})
            body = {
                "size": limit,
                "query": {"bool": {"filter": filters}},
                "sort": [{"turn_index": {"order": "desc"}}, {"updated_at": {"order": "desc"}}],
            }
            res = self._client().search(index=self.index, body=body)
            return [h.get("_source", {}) for h in res.get("hits", {}).get("hits", [])]

        try:
            return await self._run(sync)
        except Exception:
            return []

    async def conversation_frame(self, session_id: str) -> Dict[str, Any]:
        def sync() -> Dict[str, Any]:
            body = {
                "size": 1,
                "query": {
                    "bool": {
                        "filter": [
                            {"term": {"doc_type": "conversation_frame"}},
                            {"term": {"session_id": session_id}},
                        ]
                    }
                },
                "sort": [{"updated_at": {"order": "desc"}}],
            }
            res = self._client().search(index=self.index, body=body)
            hits = res.get("hits", {}).get("hits", [])
            return hits[0].get("_source", {}) if hits else {}

        try:
            return await self._run(sync)
        except Exception:
            return {}

    async def update_conversation_frame(
        self,
        *,
        session_id: str,
        question: str,
        answer: str,
        claim_support: List[Dict[str, Any]],
        turn_index: int,
    ) -> Dict[str, Any]:
        await self.ensure_index()
        existing = await self.conversation_frame(session_id)
        frame = build_conversation_frame(
            existing={**existing, "session_id": session_id},
            question=question,
            answer=answer,
            claim_support=claim_support,
            turn_index=turn_index,
        )
        frame.update(
            {
                "tenant": self.tenant,
                "session_id": session_id,
                "importance": 1.0,
                "memory_state": "promoted",
                "created_at": existing.get("created_at") or _now(),
                "updated_at": _now(),
            }
        )
        doc_id = frame.get("frame_id") or _stable_id(self.tenant, session_id, "conversation_frame")
        try:
            await self._run(self._client().index, index=self.index, id=doc_id, body=frame, refresh=False)
        except Exception as e:
            print(f"[WARN] MemoryStore.update_conversation_frame failed: {e}")
        return frame

    async def supported_claim_evidence(self, session_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        def sync() -> List[Dict[str, Any]]:
            body = {
                "size": max(5, limit),
                "query": {
                    "bool": {
                        "filter": [
                            {"term": {"doc_type": "evidence_table"}},
                            {"term": {"session_id": session_id}},
                        ]
                    }
                },
                "sort": [{"turn_index": {"order": "desc"}}, {"updated_at": {"order": "desc"}}],
            }
            res = self._client().search(index=self.index, body=body)
            out: list[dict[str, Any]] = []
            seen: set[str] = set()
            for hit in res.get("hits", {}).get("hits", []):
                src = hit.get("_source", {})
                for claim in src.get("claims") or []:
                    if not isinstance(claim, dict) or claim.get("status") != "entailed":
                        continue
                    text = str(claim.get("claim") or "").strip()
                    if not text:
                        continue
                    key = _stable_id(text)
                    if key in seen:
                        continue
                    seen.add(key)
                    out.append(
                        {
                            "source": "memory_claim",
                            "evidence_id": f"prior_claim_{key}",
                            "sentence_text": text,
                            "text": text,
                            "window_text": text,
                            "turn_index": src.get("turn_index"),
                            "answer_id": src.get("answer_id"),
                            "evidence_table_id": src.get("evidence_table_id"),
                            "retrieval_score": float(claim.get("best_entailment", 0.0) or 0.0),
                        }
                    )
                    if len(out) >= limit:
                        return out
            return out

        try:
            return await self._run(sync)
        except Exception:
            return []

    async def add_user_correction(
        self,
        *,
        session_id: str,
        correction: str,
        conflicting_claim: str | None = None,
        authoritative_fact: str | None = None,
        turn_index: int | None = None,
    ) -> None:
        await self.ensure_index()
        text = authoritative_fact or correction
        doc = {
            "doc_type": "user_correction",
            "tenant": self.tenant,
            "session_id": session_id,
            "turn_index": turn_index,
            "text": text,
            "summary": correction[:700],
            "conflicting_claim": conflicting_claim,
            "authoritative_fact": authoritative_fact,
            "terms": important_terms(text),
            "importance": 1.0,
            "created_at": _now(),
            "updated_at": _now(),
        }
        doc_id = _stable_id(self.tenant, session_id, "correction", str(turn_index), text[:160])
        try:
            await self._run(self._client().index, index=self.index, id=doc_id, body=doc, refresh=False)
        except Exception as e:
            print(f"[WARN] MemoryStore.add_user_correction failed: {e}")

        landmark = {
            "doc_type": "landmark",
            "tenant": self.tenant,
            "session_id": session_id,
            "name": "user_authoritative_correction",
            "summary": text[:700],
            "terms": important_terms(text),
            "created_at": _now(),
            "updated_at": _now(),
        }
        landmark_id = _stable_id(self.tenant, session_id, "landmark", "user_authoritative_correction")
        try:
            await self._run(self._client().index, index=self.index, id=landmark_id, body=landmark, refresh=False)
        except Exception as e:
            print(f"[WARN] MemoryStore.add_user_correction landmark failed: {e}")

    async def latest_traces(self, session_id: str, limit: int = 3) -> List[Dict[str, Any]]:
        def sync() -> List[Dict[str, Any]]:
            body = {
                "size": limit,
                "query": {
                    "bool": {
                        "filter": [
                            {"term": {"doc_type": "policy_trace"}},
                            {"term": {"session_id": session_id}},
                        ]
                    }
                },
                "sort": [{"turn_index": {"order": "desc"}}],
            }
            res = self._client().search(index=self.index, body=body)
            return [h.get("_source", {}) for h in res.get("hits", {}).get("hits", [])]

        try:
            return await self._run(sync)
        except Exception:
            return []

    async def add_search_policy_note(
        self,
        *,
        session_id: str,
        turn_index: int,
        note: str,
        search_plan: Dict[str, Any],
        reward_score: float,
    ) -> None:
        await self.ensure_index()
        queries = [
            str(item.get("query") or "")[:260]
            for item in search_plan.get("variants", [])
            if isinstance(item, dict) and item.get("query")
        ]
        doc = {
            "doc_type": "search_policy_note",
            "tenant": self.tenant,
            "session_id": session_id,
            "turn_index": turn_index,
            "text": note[:1000],
            "summary": note[:500],
            "note": note[:1000],
            "terms": important_terms(note),
            "importance": max(0.2, min(1.0, 0.35 + float(reward_score))),
            "search_plan": search_plan,
            "search_state_key": search_plan.get("state_key"),
            "search_action_key": search_plan.get("action_key"),
            "search_strategy": search_plan.get("strategy"),
            "search_queries": queries,
            "reward_score": float(reward_score),
            "created_at": _now(),
            "updated_at": _now(),
        }
        doc_id = _stable_id(self.tenant, session_id, "search_policy_note", str(turn_index), note[:160])
        try:
            await self._run(self._client().index, index=self.index, id=doc_id, body=doc, refresh=False)
        except Exception as e:
            print(f"[WARN] MemoryStore.add_search_policy_note failed: {e}")

    async def search_policy_notes(self, session_id: str, limit: int = 4) -> List[Dict[str, Any]]:
        def sync() -> List[Dict[str, Any]]:
            body = {
                "size": limit,
                "query": {
                    "bool": {
                        "filter": [
                            {"term": {"doc_type": "search_policy_note"}},
                            {"term": {"session_id": session_id}},
                        ]
                    }
                },
                "sort": [{"turn_index": {"order": "desc"}}, {"updated_at": {"order": "desc"}}],
            }
            res = self._client().search(index=self.index, body=body)
            return [h.get("_source", {}) for h in res.get("hits", {}).get("hits", [])]

        try:
            return await self._run(sync)
        except Exception:
            return []

    async def update_idea_index(
        self,
        *,
        session_id: str,
        texts: List[str],
        turn_index: int,
        reward_score: float = 0.0,
        shared: bool = False,
    ) -> None:
        await self.ensure_index()
        updates = build_idea_updates(
            tenant=self.tenant,
            session_id=session_id,
            texts=texts,
            turn_index=turn_index,
            reward_score=reward_score,
            shared=shared,
        )
        if not updates:
            return

        def sync() -> None:
            client = self._client()
            for update in updates:
                doc_id = update["idea_id"]
                existing: dict[str, Any] | None = None
                try:
                    res = client.get(index=self.index, id=doc_id)
                    existing = res.get("_source", {})
                except Exception:
                    existing = None
                doc = merge_idea_doc(existing, update)
                doc["updated_at"] = _now()
                doc.setdefault("created_at", _now())
                client.index(index=self.index, id=doc_id, body=doc, refresh=False)

        try:
            await self._run(sync)
        except Exception as e:
            print(f"[WARN] MemoryStore.update_idea_index failed: {e}")

    async def search_ideas(self, session_id: str, query: str, k: int) -> List[Dict[str, Any]]:
        def sync() -> List[Dict[str, Any]]:
            body = {
                "size": max(20, k * 4),
                "query": {
                    "bool": {
                        "must": [
                            {
                                "multi_match": {
                                    "query": query,
                                    "fields": [
                                        "idea^5",
                                        "normalized_idea^5",
                                        "synonyms^4",
                                        "concept_path^3",
                                        "child_ideas^3",
                                        "cooccurring_ideas^2",
                                        "terms^3",
                                    ],
                                }
                            }
                        ],
                        "filter": [
                            {"term": {"doc_type": "idea"}},
                            {"terms": {"scope": [session_id, "shared"]}},
                        ],
                    }
                },
                "sort": [{"importance": {"order": "desc"}}],
            }
            res = self._client().search(index=self.index, body=body)
            hits = []
            for h in res.get("hits", {}).get("hits", []):
                src = h.get("_source", {})
                src["_score"] = h.get("_score", 0.0)
                hits.append(src)
            return rank_ideas(hits, query, limit=k)

        try:
            return await self._run(sync)
        except Exception:
            return []

    async def debug_ideas(self, session_id: str | None = None, limit: int = 20) -> List[Dict[str, Any]]:
        def sync() -> List[Dict[str, Any]]:
            filters: list[dict[str, Any]] = [{"term": {"doc_type": "idea"}}]
            if session_id:
                filters.append({"terms": {"scope": [session_id, "shared"]}})
            body = {
                "size": limit,
                "query": {"bool": {"filter": filters}},
                "sort": [
                    {"importance": {"order": "desc"}},
                    {"frequency": {"order": "desc"}},
                    {"updated_at": {"order": "desc"}},
                ],
            }
            res = self._client().search(index=self.index, body=body)
            return [h.get("_source", {}) for h in res.get("hits", {}).get("hits", [])]

        try:
            return await self._run(sync)
        except Exception:
            return []

    async def update_action_value(
        self,
        *,
        session_id: str,
        state_key: str,
        action_key: str,
        reward_score: float,
        shared: bool = False,
    ) -> None:
        await self.ensure_index()
        scope = "shared" if shared else session_id
        doc_id = action_id(self.tenant, scope, state_key, action_key)

        def sync() -> None:
            client = self._client()
            existing: dict[str, Any] | None = None
            try:
                res = client.get(index=self.index, id=doc_id)
                existing = res.get("_source", {})
            except Exception:
                existing = None
            doc = update_action_value(
                existing,
                tenant=self.tenant,
                scope=scope,
                state=state_key,
                action=action_key,
                reward=reward_score,
            )
            doc["updated_at"] = _now()
            doc.setdefault("created_at", _now())
            client.index(index=self.index, id=doc_id, body=doc, refresh=False)

        try:
            await self._run(sync)
        except Exception as e:
            print(f"[WARN] MemoryStore.update_action_value failed: {e}")

    async def action_values(self, session_id: str, state_key: str, limit: int = 5) -> List[Dict[str, Any]]:
        def sync() -> List[Dict[str, Any]]:
            body = {
                "size": limit,
                "query": {
                    "bool": {
                        "filter": [
                            {"term": {"doc_type": "action_value"}},
                            {"term": {"state_key": state_key}},
                            {"terms": {"scope": [session_id, "shared"]}},
                        ]
                    }
                },
                "sort": [{"q_value": {"order": "desc"}}, {"visits": {"order": "desc"}}],
            }
            res = self._client().search(index=self.index, body=body)
            return [h.get("_source", {}) for h in res.get("hits", {}).get("hits", [])]

        try:
            return await self._run(sync)
        except Exception:
            return []

    async def debug_action_values(
        self,
        *,
        session_id: str | None = None,
        state_key: str | None = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        def sync() -> List[Dict[str, Any]]:
            filters: list[dict[str, Any]] = [{"term": {"doc_type": "action_value"}}]
            if session_id:
                filters.append({"terms": {"scope": [session_id, "shared"]}})
            if state_key:
                filters.append({"term": {"state_key": state_key}})
            body = {
                "size": limit,
                "query": {"bool": {"filter": filters}},
                "sort": [
                    {"q_value": {"order": "desc"}},
                    {"visits": {"order": "desc"}},
                    {"updated_at": {"order": "desc"}},
                ],
            }
            res = self._client().search(index=self.index, body=body)
            return [h.get("_source", {}) for h in res.get("hits", {}).get("hits", [])]

        try:
            return await self._run(sync)
        except Exception:
            return []
