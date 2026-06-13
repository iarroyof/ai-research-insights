"""vocabulary_store.py — Thompson-Sampling bandit vocabulary store.

Scope hierarchy: session -> run -> global_candidate.

Feature flag: VOCAB_STORE_ENABLED=true  (default: false — all methods no-op).
Redis key pattern: vocabstore:{scope}:{scope_id}:{term}
Redis DB: 1  (separate from application data in DB 0)
"""
from __future__ import annotations

import os
import random
import time
from typing import Any

# ---------------------------------------------------------------------------
# Redis client — lazy singleton, falls back to in-memory dict
# ---------------------------------------------------------------------------

_REDIS_HOST = os.environ.get("REDIS_HOST", "ai-research-insights-redis-1")
_REDIS_PORT = int(os.environ.get("REDIS_PORT", "6379"))
_REDIS_DB = int(os.environ.get("VOCAB_REDIS_DB", "1"))

_SESSION_TTL = 7_200      # 2 hours
_RUN_TTL = 172_800        # 48 hours

_redis_client: Any = None          # redis.Redis or FakeRedis or None
_in_memory_store: dict[str, dict] = {}  # fallback


def _get_redis() -> Any:
    """Return a connected redis.Redis client, or None if unavailable."""
    global _redis_client
    if _redis_client is not None:
        return _redis_client
    try:
        import redis  # type: ignore
        client = redis.Redis(
            host=_REDIS_HOST,
            port=_REDIS_PORT,
            db=_REDIS_DB,
            socket_connect_timeout=1,
            socket_timeout=1,
        )
        client.ping()
        _redis_client = client
    except Exception:
        _redis_client = None
    return _redis_client


def _reset_redis_client() -> None:
    """Force re-connection on next call (used in tests)."""
    global _redis_client
    _redis_client = None


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

_KEY_PREFIX = "vocabstore"


def _make_key(scope: str, scope_id: str, term: str) -> str:
    """Return the Redis hash key for a vocab entry."""
    return f"{_KEY_PREFIX}:{scope}:{scope_id}:{term}"


def _parse_key(key: str) -> tuple[str, str, str]:
    """Return (scope, scope_id, term) from a Redis key string."""
    parts = key.split(":", 3)
    if len(parts) == 4:
        return parts[1], parts[2], parts[3]
    return "", "", ""


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class VocabularyStore:
    """Per-scope Thompson-Sampling bandit vocabulary store."""

    # ------------------------------------------------------------------
    # Feature flag
    # ------------------------------------------------------------------

    @staticmethod
    def enabled() -> bool:
        """Return True only if VOCAB_STORE_ENABLED=true AND Redis reachable."""
        if os.environ.get("VOCAB_STORE_ENABLED", "false").lower() != "true":
            return False
        return _get_redis() is not None

    # ------------------------------------------------------------------
    # Internal read / write
    # ------------------------------------------------------------------

    def _read(self, key: str) -> dict:
        """Return {alpha, beta, last_seen, total_updates} for a key."""
        r = _get_redis()
        if r is not None:
            try:
                raw = r.hgetall(key)
                if raw:
                    return {
                        (k.decode() if isinstance(k, bytes) else k):
                        float(v.decode() if isinstance(v, bytes) else v)
                        for k, v in raw.items()
                    }
            except Exception:
                pass
        # In-memory fallback
        return dict(_in_memory_store.get(key, {}))

    def _write(self, key: str, data: dict) -> None:
        """Write fields to the Redis hash, or in-memory fallback."""
        r = _get_redis()
        if r is not None:
            try:
                r.hset(key, mapping={k: str(v) for k, v in data.items()})
                return
            except Exception:
                pass
        _in_memory_store[key] = dict(data)

    def _ttl(self, key: str, seconds: int) -> None:
        r = _get_redis()
        if r is not None:
            try:
                r.expire(key, seconds)
            except Exception:
                pass

    def _scan_prefix(self, pattern: str) -> list[str]:
        """Return all keys matching the given Redis glob pattern."""
        r = _get_redis()
        if r is not None:
            try:
                keys: list[str] = []
                for key in r.scan_iter(pattern, count=200):
                    keys.append(key.decode() if isinstance(key, bytes) else key)
                return keys
            except Exception:
                pass
        # In-memory fallback: simple prefix match (pattern uses * glob)
        prefix = pattern.rstrip("*")
        return [k for k in _in_memory_store if k.startswith(prefix)]

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def record_outcome(
        self,
        term: str,
        scope: str,
        scope_id: str,
        gap_delta: float,
        distractor_pen: float,
        base_reward: float,
    ) -> None:
        """Update Beta(alpha, beta) for this term given one retrieval outcome.

        utility = gap_delta * 0.6 - distractor_pen * 0.3 + (base_reward - 0.5) * 0.1
        utility > 0  -> alpha += utility (success credit)
        utility <= 0 -> beta  += abs(utility) (failure credit)
        """
        if not term:
            return
        key = _make_key(scope, scope_id, term.lower().strip())
        existing = self._read(key)
        alpha = float(existing.get("alpha", 1.0))
        beta = float(existing.get("beta", 1.0))
        total = int(float(existing.get("total_updates", 0)))

        utility = gap_delta * 0.6 - distractor_pen * 0.3 + (base_reward - 0.5) * 0.1
        if utility > 0:
            alpha += utility
        else:
            beta += abs(utility)

        self._write(key, {
            "alpha": round(alpha, 6),
            "beta": round(beta, 6),
            "last_seen": time.time(),
            "total_updates": total + 1,
        })

    def top_terms(
        self,
        scope: str,
        scope_id: str,
        limit: int = 30,
    ) -> list[tuple[str, float]]:
        """Return top terms by Thompson Sample (draw from Beta(alpha, beta)).

        Terms with fewer than 2 updates are excluded (min_updates guard).
        Returns list of (term, sample_value) sorted descending.
        """
        pattern = f"{_KEY_PREFIX}:{scope}:{scope_id}:*"
        keys = self._scan_prefix(pattern)
        results: list[tuple[str, float]] = []
        for key in keys:
            data = self._read(key)
            if not data:
                continue
            total = int(float(data.get("total_updates", 0)))
            if total < 2:
                continue
            alpha = float(data.get("alpha", 1.0))
            beta = float(data.get("beta", 1.0))
            sample = random.betavariate(max(alpha, 1e-6), max(beta, 1e-6))
            _, _, term = _parse_key(key)
            results.append((term, round(sample, 6)))
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]

    def session_top_terms(
        self,
        session_id: str,
        limit: int = 20,
    ) -> list[tuple[str, float]]:
        """Convenience: top_terms for session scope."""
        return self.top_terms("session", session_id, limit)

    def run_top_terms(
        self,
        run_id: str,
        limit: int = 20,
    ) -> list[tuple[str, float]]:
        """Convenience: top_terms for run scope."""
        return self.top_terms("run", run_id, limit)

    def promote_to_global_candidate(
        self,
        term: str,
        aliases: list[str],
        run_id: str,
    ) -> None:
        """Mark a run-scope term as a global candidate.

        Stores run_id and current alpha/beta under scope='global_candidate'.
        Accumulates run_ids promoting the term.
        """
        if not term:
            return
        term_clean = term.lower().strip()
        gc_key = _make_key("global_candidate", "shared", term_clean)

        # Read run-scope state for this term
        run_key = _make_key("run", run_id, term_clean)
        run_data = self._read(run_key)
        alpha = float(run_data.get("alpha", 1.0))
        beta = float(run_data.get("beta", 1.0))

        # Read existing global_candidate state
        r = _get_redis()
        if r is not None:
            try:
                raw = r.hgetall(gc_key)
                existing = {
                    (k.decode() if isinstance(k, bytes) else k):
                    (v.decode() if isinstance(v, bytes) else v)
                    for k, v in raw.items()
                }
            except Exception:
                existing = {}
        else:
            existing = dict(_in_memory_store.get(gc_key, {}))

        # Accumulate run ids
        run_ids_raw = existing.get("run_ids", "")
        existing_runs = set(str(run_ids_raw).split(",")) if run_ids_raw else set()
        existing_runs.discard("")
        existing_runs.add(run_id)

        # Accumulate aliases
        aliases_raw = existing.get("aliases", "")
        existing_aliases = set(str(aliases_raw).split(",")) if aliases_raw else set()
        existing_aliases.discard("")
        existing_aliases.update(a.lower().strip() for a in aliases if a)

        data_to_write: dict[str, Any] = {
            "alpha": round(alpha, 6),
            "beta": round(beta, 6),
            "run_ids": ",".join(sorted(existing_runs)),
            "run_count": str(len(existing_runs)),
            "aliases": ",".join(sorted(existing_aliases)),
            "last_seen": str(time.time()),
        }

        if r is not None:
            try:
                r.hset(gc_key, mapping=data_to_write)
                return
            except Exception:
                pass
        _in_memory_store[gc_key] = data_to_write  # type: ignore

    def global_candidates(
        self,
        min_run_count: int = 2,
        min_success_rate: float = 0.65,
    ) -> list[dict[str, Any]]:
        """Return global_candidate terms meeting min_run_count + min_success_rate.

        Each returned item has: term, alpha, beta, success_rate, run_count, run_ids, aliases.
        """
        pattern = f"{_KEY_PREFIX}:global_candidate:shared:*"
        keys = self._scan_prefix(pattern)
        results: list[dict[str, Any]] = []
        for key in keys:
            r = _get_redis()
            if r is not None:
                try:
                    raw = r.hgetall(key)
                    data: dict[str, Any] = {
                        (k.decode() if isinstance(k, bytes) else k):
                        (v.decode() if isinstance(v, bytes) else v)
                        for k, v in raw.items()
                    }
                except Exception:
                    data = {}
            else:
                data = dict(_in_memory_store.get(key, {}))

            if not data:
                continue
            run_count = int(float(data.get("run_count", 0)))
            alpha = float(data.get("alpha", 1.0))
            beta = float(data.get("beta", 1.0))
            success_rate = alpha / (alpha + beta) if (alpha + beta) > 0 else 0.0
            if run_count < min_run_count or success_rate < min_success_rate:
                continue
            _, _, term = _parse_key(key)
            run_ids_raw = data.get("run_ids", "")
            aliases_raw = data.get("aliases", "")
            results.append({
                "term": term,
                "alpha": alpha,
                "beta": beta,
                "success_rate": round(success_rate, 4),
                "run_count": run_count,
                "run_ids": [rid for rid in str(run_ids_raw).split(",") if rid],
                "aliases": [a for a in str(aliases_raw).split(",") if a],
            })
        return results

    def expire_session(self, session_id: str, ttl_seconds: int = _SESSION_TTL) -> None:
        """Set Redis TTL on all keys for this session."""
        pattern = f"{_KEY_PREFIX}:session:{session_id}:*"
        keys = self._scan_prefix(pattern)
        for key in keys:
            self._ttl(key, ttl_seconds)

    def expire_run(self, run_id: str, ttl_seconds: int = _RUN_TTL) -> None:
        """Set Redis TTL on all keys for this run (48h by default)."""
        pattern = f"{_KEY_PREFIX}:run:{run_id}:*"
        keys = self._scan_prefix(pattern)
        for key in keys:
            self._ttl(key, ttl_seconds)


# ---------------------------------------------------------------------------
# Module 4 — RunVocabManager
# ---------------------------------------------------------------------------


class RunVocabManager:
    """Identify high-utility run-scope terms and promote to global candidates."""

    def __init__(self, run_id: str) -> None:
        self.run_id = run_id

    def finalize(
        self,
        min_scenario_count: int = 2,
        min_success_rate: float = 0.65,
    ) -> list[dict[str, Any]]:
        """After all scenarios in a run are complete, promote high-utility terms.

        Returns a list of promoted candidate dicts for logging.
        """
        if not VocabularyStore.enabled():
            return []

        # Lazy import to avoid circular imports inside the API
        try:
            from app.memory.search_agent import ANCHOR_ALIASES as _AA, PUZZLE_NODE_STOP_TERMS as _STOP, _canonical_search_anchor  # type: ignore
            from app.memory.idea_index import normalize_idea, BIOMEDICAL_SYNONYMS  # type: ignore
        except ImportError:
            _AA = {}
            _STOP = set()  # type: ignore
            def _canonical_search_anchor(t: str) -> str: return t.lower().strip()  # type: ignore
            def normalize_idea(t: str) -> str: return t.lower().strip()  # type: ignore
            BIOMEDICAL_SYNONYMS = {}  # type: ignore

        vs = VocabularyStore()
        run_terms = vs.run_top_terms(self.run_id, limit=100)
        promoted: list[dict[str, Any]] = []

        for term, _sample in run_terms:
            # Re-read exact alpha/beta from store
            key = _make_key("run", self.run_id, term)
            r = _get_redis()
            if r is not None:
                try:
                    raw = r.hgetall(key)
                    data = {
                        (k.decode() if isinstance(k, bytes) else k):
                        float(v.decode() if isinstance(v, bytes) else v)
                        for k, v in raw.items()
                    }
                except Exception:
                    data = {}
            else:
                data = dict(_in_memory_store.get(key, {}))

            alpha = float(data.get("alpha", 1.0))
            beta = float(data.get("beta", 1.0))
            total = int(float(data.get("total_updates", 0)))
            success_rate = alpha / (alpha + beta) if (alpha + beta) > 0 else 0.0

            if total < min_scenario_count:
                continue
            if success_rate < min_success_rate:
                continue

            # Don't duplicate existing ANCHOR_ALIASES
            canonical = _canonical_search_anchor(term)
            if canonical in _AA:
                continue
            # Don't add stopwords
            if canonical in _STOP or term in _STOP:
                continue

            # Generate candidate aliases
            normalized = normalize_idea(canonical)
            aliases: list[str] = list(dict.fromkeys([a for a in [canonical, normalized, term] if a]))
            # Pull any synonym-group expansions
            for syn_alias, syn_canonical in BIOMEDICAL_SYNONYMS.items():
                if syn_canonical == canonical or syn_alias == canonical:
                    aliases.append(syn_alias)
                    aliases.append(syn_canonical)
            aliases = list(dict.fromkeys(a for a in aliases if a))

            vs.promote_to_global_candidate(term, aliases, self.run_id)
            promoted.append({
                "term": term,
                "success_rate": round(success_rate, 4),
                "alpha": round(alpha, 6),
                "beta": round(beta, 6),
                "aliases": aliases,
                "run_id": self.run_id,
            })

        return promoted

    def holdout_validation_required(self, candidates: list[dict[str, Any]]) -> bool:
        """Return True if any candidate could change effective ANCHOR_ALIASES coverage."""
        return bool(candidates)
