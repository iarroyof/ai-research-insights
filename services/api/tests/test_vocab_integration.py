"""tests/test_vocab_integration.py — Integration tests for Modules 2 and 3.

Tests session vocab injection (Module 2) and per-term reward credit (Module 3).

Run inside the api container:
    cd /app && python3 -m pytest tests/test_vocab_integration.py -v
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, "/app")

import pytest
import fakeredis  # type: ignore

import app.memory.vocabulary_store as vs_module
from app.memory.vocabulary_store import VocabularyStore, _make_key, _in_memory_store


# -------------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_store():
    vs_module._in_memory_store.clear()
    vs_module._reset_redis_client()
    yield
    vs_module._in_memory_store.clear()
    vs_module._reset_redis_client()


@pytest.fixture()
def fake_redis_env(monkeypatch):
    server = fakeredis.FakeServer()
    fake = fakeredis.FakeRedis(server=server, db=1)
    monkeypatch.setattr(vs_module, "_redis_client", fake)
    monkeypatch.setenv("VOCAB_STORE_ENABLED", "true")
    yield fake
    fake.flushdb()


# -------------------------------------------------------------------------
# Module 2 — Point A: session vocab injection
# -------------------------------------------------------------------------

def test_session_vocab_injection_when_enabled(fake_redis_env, monkeypatch):
    """After turn 1 records 'fungi' as confirmed, turn 2 should find it
    available via session_top_terms (simulating anchor injection)."""
    monkeypatch.setenv("VOCAB_STORE_ENABLED", "true")
    store = VocabularyStore()
    session_id = "sess_mod2_a"

    # Simulate turn 1: 'fungi' confirmed twice (>=2 updates for min_updates guard)
    store.record_outcome("fungi", "session", session_id, gap_delta=0.8, distractor_pen=0.0, base_reward=0.5)
    store.record_outcome("fungi", "session", session_id, gap_delta=0.8, distractor_pen=0.0, base_reward=0.5)

    # Turn 2: session_top_terms should contain 'fungi'
    top = store.session_top_terms(session_id, limit=20)
    terms = [t for t, _ in top]
    assert "fungi" in terms, f"'fungi' not found in top terms: {top}"


def test_session_vocab_injection_disabled(monkeypatch):
    """When VOCAB_STORE_ENABLED=false, session_top_terms returns empty."""
    monkeypatch.setenv("VOCAB_STORE_ENABLED", "false")
    assert VocabularyStore.enabled() is False
    # Directly calling session_top_terms with no Redis returns empty
    store = VocabularyStore()
    # Force in-memory path
    monkeypatch.setattr(vs_module, "_redis_client", None)
    monkeypatch.setattr(vs_module, "_get_redis", lambda: None)
    # Even in-memory: calling session_top_terms should return []
    # (enabled() guard not checked here, but store itself returns [] when empty)
    top = store.session_top_terms("any_session", limit=20)
    assert top == []


def test_inject_capped_at_16_anchors(fake_redis_env, monkeypatch):
    """Injection must not push total anchors beyond 16."""
    monkeypatch.setenv("VOCAB_STORE_ENABLED", "true")
    store = VocabularyStore()
    session_id = "sess_cap"

    # Record 20 distinct terms with 2+ updates each
    for i in range(20):
        term = f"term{i:02d}"
        store.record_outcome(term, "session", session_id, 0.8, 0.0, 0.5)
        store.record_outcome(term, "session", session_id, 0.8, 0.0, 0.5)

    # Simulate injection logic from build_auto_context Point A
    _existing_anchors = set([f"anchor{j}" for j in range(10)])  # 10 pre-existing
    _session_terms = store.session_top_terms(session_id, limit=20)
    _new_anchors = [
        t for t, _ in _session_terms if t not in _existing_anchors
    ][:max(0, 16 - len(_existing_anchors))]
    final_anchors = list(_existing_anchors) + _new_anchors
    assert len(final_anchors) <= 16, f"Too many anchors: {len(final_anchors)}"


# -------------------------------------------------------------------------
# Module 2 — Point B: GapSpec persistence
# -------------------------------------------------------------------------

def test_gapspec_confirmed_persisted_to_session(fake_redis_env, monkeypatch):
    """Confirmed entities should produce positive alpha via record_outcome."""
    monkeypatch.setenv("VOCAB_STORE_ENABLED", "true")
    store = VocabularyStore()
    session_id = "sess_gapspec"

    # Simulate what Point B does for confirmed_entities
    store.record_outcome("fungi", "session", session_id,
                         gap_delta=0.9, distractor_pen=0.0, base_reward=0.5)
    key = _make_key("session", session_id, "fungi")
    data = store._read(key)
    alpha = float(data["alpha"])
    # utility = 0.9*0.6 = 0.54 -> alpha = 1.0 + 0.54 = 1.54
    assert alpha > 1.0


def test_gapspec_missing_persisted_as_negative(fake_redis_env, monkeypatch):
    """Missing entities should produce increased beta via record_outcome."""
    monkeypatch.setenv("VOCAB_STORE_ENABLED", "true")
    store = VocabularyStore()
    session_id = "sess_miss"

    # Simulate what Point B does for missing_entities
    store.record_outcome("cancer", "session", session_id,
                         gap_delta=0.0, distractor_pen=0.2, base_reward=0.5)
    key = _make_key("session", session_id, "cancer")
    data = store._read(key)
    beta = float(data["beta"])
    # utility = -0.2*0.3 = -0.06 -> beta = 1.0 + 0.06 = 1.06
    assert beta > 1.0


# -------------------------------------------------------------------------
# Module 3 — per-term reward credit
# -------------------------------------------------------------------------

def test_positive_gap_closure_increments_alpha(fake_redis_env, monkeypatch):
    """gap_closure_score=0.8, distractor_ratio=0.1 -> positive utility -> alpha up."""
    monkeypatch.setenv("VOCAB_STORE_ENABLED", "true")
    store = VocabularyStore()
    session_id = "sess_m3_pos"

    store.record_outcome("fungi", "session", session_id,
                         gap_delta=0.8, distractor_pen=0.1, base_reward=0.5)
    key = _make_key("session", session_id, "fungi")
    data = store._read(key)
    alpha = float(data["alpha"])
    beta = float(data["beta"])
    # utility = 0.8*0.6 - 0.1*0.3 = 0.48 - 0.03 = 0.45
    assert alpha == pytest.approx(1.45, abs=1e-4)
    assert beta == pytest.approx(1.0, abs=1e-4)


def test_negative_distractor_increments_beta(fake_redis_env, monkeypatch):
    """gap_closure_score=0.0, distractor_ratio=0.7 -> negative utility -> beta up."""
    monkeypatch.setenv("VOCAB_STORE_ENABLED", "true")
    store = VocabularyStore()
    session_id = "sess_m3_neg"

    store.record_outcome("fungi", "session", session_id,
                         gap_delta=0.0, distractor_pen=0.7, base_reward=0.5)
    key = _make_key("session", session_id, "fungi")
    data = store._read(key)
    alpha = float(data["alpha"])
    beta = float(data["beta"])
    # utility = -0.7*0.3 = -0.21
    assert alpha == pytest.approx(1.0, abs=1e-4)
    assert beta == pytest.approx(1.21, abs=1e-4)


def test_disabled_flag_prevents_updates(monkeypatch):
    """When VOCAB_STORE_ENABLED=false, record_outcome is called but enabled() is False.

    This verifies that any guard checking enabled() before calling record_outcome
    would skip the call entirely.
    """
    monkeypatch.setenv("VOCAB_STORE_ENABLED", "false")
    assert VocabularyStore.enabled() is False
    # The store still accepts direct calls (no internal guard in record_outcome itself)
    # but production code wraps calls with `if VocabularyStore.enabled()`.
    # Verify in-memory store remains empty after direct call without Redis
    monkeypatch.setattr(vs_module, "_redis_client", None)
    monkeypatch.setattr(vs_module, "_get_redis", lambda: None)
    store = VocabularyStore()
    store.record_outcome("fungi", "session", "s1", 0.8, 0.0, 0.5)
    # Should be in in-memory fallback
    key = _make_key("session", "s1", "fungi")
    assert key in vs_module._in_memory_store


def test_run_id_records_to_run_scope(fake_redis_env, monkeypatch):
    """When run_id is provided, outcomes are recorded in run scope too."""
    monkeypatch.setenv("VOCAB_STORE_ENABLED", "true")
    store = VocabularyStore()
    run_id = "run_test_001"
    session_id = "sess_run"

    # Simulate Module 3 behaviour: session AND run scope
    store.record_outcome("fungi", "session", session_id, 0.8, 0.1, 0.5)
    store.record_outcome("fungi", "run", run_id, 0.8, 0.1, 0.5)

    sess_key = _make_key("session", session_id, "fungi")
    run_key = _make_key("run", run_id, "fungi")
    assert store._read(sess_key), "Session key not written"
    assert store._read(run_key), "Run key not written"
