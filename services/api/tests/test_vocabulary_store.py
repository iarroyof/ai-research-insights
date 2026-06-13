"""tests/test_vocabulary_store.py — Unit tests for Module 1 (VocabularyStore).

Run inside the api container:
    cd /app && python3 -m pytest tests/test_vocabulary_store.py -v
"""
from __future__ import annotations

import os
import sys

# Ensure we can find the local module when run from /app inside the container
sys.path.insert(0, "/app")

import pytest
import fakeredis  # type: ignore

# -------------------------------------------------------------------------
# Helpers to wire up fakeredis for isolation
# -------------------------------------------------------------------------

import app.memory.vocabulary_store as vs_module  # noqa: E402


@pytest.fixture(autouse=True)
def reset_store():
    """Between tests: clear in-memory fallback, reset cached redis client."""
    vs_module._in_memory_store.clear()
    vs_module._reset_redis_client()
    yield
    vs_module._in_memory_store.clear()
    vs_module._reset_redis_client()


@pytest.fixture()
def fake_redis_env(monkeypatch):
    """Patch the module to use fakeredis and enable the feature flag."""
    server = fakeredis.FakeServer()
    fake = fakeredis.FakeRedis(server=server, db=1)

    monkeypatch.setattr(vs_module, "_redis_client", fake)
    monkeypatch.setenv("VOCAB_STORE_ENABLED", "true")
    yield fake
    fake.flushdb()


# -------------------------------------------------------------------------
# Test: enabled() returns False when flag is not set
# -------------------------------------------------------------------------

def test_enabled_false_when_not_set(monkeypatch):
    monkeypatch.delenv("VOCAB_STORE_ENABLED", raising=False)
    # Even if Redis is available, flag=false means disabled
    assert vs_module.VocabularyStore.enabled() is False


def test_enabled_false_when_explicitly_false(monkeypatch):
    monkeypatch.setenv("VOCAB_STORE_ENABLED", "false")
    assert vs_module.VocabularyStore.enabled() is False


def test_enabled_true_with_fakeredis(fake_redis_env, monkeypatch):
    monkeypatch.setenv("VOCAB_STORE_ENABLED", "true")
    assert vs_module.VocabularyStore.enabled() is True


# -------------------------------------------------------------------------
# Test: record_outcome updates alpha / beta correctly
# -------------------------------------------------------------------------

def test_record_outcome_positive_utility(fake_redis_env):
    """Positive utility (gap_delta dominates) -> alpha increases."""
    store = vs_module.VocabularyStore()
    store.record_outcome("fungi", "session", "sess1",
                         gap_delta=0.8, distractor_pen=0.1, base_reward=0.5)
    key = vs_module._make_key("session", "sess1", "fungi")
    data = store._read(key)
    alpha = float(data["alpha"])
    beta = float(data["beta"])
    # utility = 0.8*0.6 - 0.1*0.3 + 0.0 = 0.48 - 0.03 = 0.45
    assert alpha > 1.0, f"alpha should be > 1.0 (initial), got {alpha}"
    assert pytest.approx(alpha, abs=1e-4) == 1.0 + 0.45
    assert beta == pytest.approx(1.0, abs=1e-4)


def test_record_outcome_negative_utility(fake_redis_env):
    """Negative utility (distractor dominates) -> beta increases."""
    store = vs_module.VocabularyStore()
    store.record_outcome("fungi", "session", "sess1",
                         gap_delta=0.0, distractor_pen=0.7, base_reward=0.5)
    key = vs_module._make_key("session", "sess1", "fungi")
    data = store._read(key)
    alpha = float(data["alpha"])
    beta = float(data["beta"])
    # utility = 0.0 - 0.7*0.3 = -0.21  -> beta += 0.21
    assert alpha == pytest.approx(1.0, abs=1e-4)
    assert pytest.approx(beta, abs=1e-4) == 1.0 + 0.21


def test_record_outcome_multiple_updates_accumulate(fake_redis_env):
    """Multiple record_outcome calls accumulate alpha/beta over time."""
    store = vs_module.VocabularyStore()
    store.record_outcome("fungi", "session", "sess1", 0.8, 0.0, 0.5)
    store.record_outcome("fungi", "session", "sess1", 0.8, 0.0, 0.5)
    key = vs_module._make_key("session", "sess1", "fungi")
    data = store._read(key)
    total = int(float(data["total_updates"]))
    assert total == 2


# -------------------------------------------------------------------------
# Test: top_terms respects min_updates guard
# -------------------------------------------------------------------------

def test_top_terms_min_updates_guard(fake_redis_env):
    """Terms with < 2 updates must NOT appear in top_terms."""
    store = vs_module.VocabularyStore()
    # Only 1 update -> should NOT appear
    store.record_outcome("fungi", "session", "sess1", 0.9, 0.0, 0.5)
    result = store.session_top_terms("sess1", limit=20)
    assert len(result) == 0, f"Expected no terms (min_updates guard), got {result}"


def test_top_terms_appears_after_two_updates(fake_redis_env):
    """Term with >= 2 updates MUST appear in top_terms."""
    store = vs_module.VocabularyStore()
    store.record_outcome("fungi", "session", "sess1", 0.8, 0.0, 0.5)
    store.record_outcome("fungi", "session", "sess1", 0.8, 0.0, 0.5)
    result = store.session_top_terms("sess1", limit=20)
    terms = [t for t, _ in result]
    assert "fungi" in terms


def test_top_terms_returns_sorted_desc(fake_redis_env):
    """top_terms must be sorted by Thompson sample descending."""
    store = vs_module.VocabularyStore()
    # Give "fungi" very high alpha, "cancer" neutral
    for _ in range(4):
        store.record_outcome("fungi", "session", "sess1", 1.0, 0.0, 0.7)
    for _ in range(2):
        store.record_outcome("cancer", "session", "sess1", 0.0, 0.8, 0.3)
    result = store.session_top_terms("sess1", limit=10)
    scores = [s for _, s in result]
    assert scores == sorted(scores, reverse=True)


# -------------------------------------------------------------------------
# Test: expire_session sets TTL
# -------------------------------------------------------------------------

def test_expire_session_sets_ttl(fake_redis_env):
    """expire_session must set a positive TTL on all session keys."""
    store = vs_module.VocabularyStore()
    store.record_outcome("fungi", "session", "sess_ttl", 0.8, 0.0, 0.5)
    store.record_outcome("fungi", "session", "sess_ttl", 0.8, 0.0, 0.5)
    store.expire_session("sess_ttl", ttl_seconds=3600)
    key = vs_module._make_key("session", "sess_ttl", "fungi")
    ttl = fake_redis_env.ttl(key)
    assert ttl > 0, f"Expected positive TTL, got {ttl}"
    assert ttl <= 3600


# -------------------------------------------------------------------------
# Test: in-memory fallback when Redis is absent
# -------------------------------------------------------------------------

def test_in_memory_fallback(monkeypatch):
    """When Redis is unavailable the in-memory store is used transparently."""
    monkeypatch.setattr(vs_module, "_redis_client", None)
    # Also patch _get_redis to always return None (simulate no Redis)
    monkeypatch.setattr(vs_module, "_get_redis", lambda: None)
    monkeypatch.setenv("VOCAB_STORE_ENABLED", "true")

    store = vs_module.VocabularyStore()
    # enabled() returns False when Redis is None — so call record_outcome directly
    store.record_outcome("fungi", "session", "fallback_sess", 0.8, 0.0, 0.5)
    store.record_outcome("fungi", "session", "fallback_sess", 0.8, 0.0, 0.5)

    key = vs_module._make_key("session", "fallback_sess", "fungi")
    assert key in vs_module._in_memory_store
    data = vs_module._in_memory_store[key]
    assert float(data["alpha"]) > 1.0


# -------------------------------------------------------------------------
# Test: promote_to_global_candidate and global_candidates
# -------------------------------------------------------------------------

def test_promote_and_retrieve_global_candidates(fake_redis_env):
    """Promoted terms should appear in global_candidates() when criteria met."""
    store = vs_module.VocabularyStore()
    # Simulate two runs with positive outcomes -> high alpha
    for run_id in ["run_a", "run_b"]:
        for _ in range(3):
            store.record_outcome("mycoterm", "run", run_id, 0.9, 0.0, 0.7)
        store.promote_to_global_candidate("mycoterm", ["mycoterm", "mycobacterium"], run_id)

    candidates = store.global_candidates(min_run_count=2, min_success_rate=0.65)
    terms = [c["term"] for c in candidates]
    assert "mycoterm" in terms
    cand = next(c for c in candidates if c["term"] == "mycoterm")
    assert cand["run_count"] == 2
    assert cand["success_rate"] >= 0.65


def test_global_candidates_filtered_by_run_count(fake_redis_env):
    """Terms from only 1 run should NOT appear when min_run_count=2."""
    store = vs_module.VocabularyStore()
    for _ in range(3):
        store.record_outcome("rare_term", "run", "run_only", 0.9, 0.0, 0.7)
    store.promote_to_global_candidate("rare_term", ["rare_term"], "run_only")

    candidates = store.global_candidates(min_run_count=2, min_success_rate=0.65)
    terms = [c["term"] for c in candidates]
    assert "rare_term" not in terms
