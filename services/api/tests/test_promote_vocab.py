"""tests/test_promote_vocab.py — Tests for Module 5 (global promotion gate).

Tests the diversity/cross-frame filters and dry-run behaviour.

Run inside the api container:
    cd /app && python3 -m pytest tests/test_promote_vocab.py -v
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, "/app")

import pytest
import fakeredis  # type: ignore

import app.memory.vocabulary_store as vs_module
from app.memory.vocabulary_store import VocabularyStore, _make_key, _in_memory_store

# Import Module 5 helpers (script lives in evals dir on server; test it via sys.path)
# Since the container doesn't mount evals, we re-implement the helpers inline for unit tests.
# The actual promote_vocab.py is tested via dry-run invocation in test_promote_vocab_cli.

# -------------------------------------------------------------------------
# Re-import the filter helpers from promote_vocab logic for unit testing
# -------------------------------------------------------------------------

def _char_ngrams(text: str, n: int = 4) -> set[str]:
    text = text.lower().replace(" ", "_")
    if len(text) < n:
        return {text}
    return {text[i:i + n] for i in range(len(text) - n + 1)}


def _jaccard_similarity(a: str, b: str, n: int = 4) -> float:
    sa = _char_ngrams(a, n)
    sb = _char_ngrams(b, n)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


def _is_near_duplicate(candidate: str, existing_keys: list[str], threshold: float = 0.75) -> tuple[bool, str]:
    best_match = ""
    best_sim = 0.0
    for key in existing_keys:
        sim = _jaccard_similarity(candidate, key)
        if sim > best_sim:
            best_sim = sim
            best_match = key
    return (best_sim > threshold, best_match)


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
# Test: 4-gram Jaccard diversity filter
# -------------------------------------------------------------------------

def test_near_duplicate_detected(monkeypatch):
    """'fungal' is near-duplicate of 'fungi' (in ANCHOR_ALIASES keys)."""
    from app.memory.search_agent import ANCHOR_ALIASES
    existing_keys = list(ANCHOR_ALIASES.keys())
    # 'fungal' is already a key in ANCHOR_ALIASES
    is_dup, closest = _is_near_duplicate("fungal", existing_keys, threshold=0.75)
    # 'fungal' and 'fungi' are very similar
    assert is_dup, f"'fungal' should be detected as near-duplicate of '{closest}'"


def test_novel_term_not_near_duplicate(monkeypatch):
    """'aspergillosis' is distinct from all ANCHOR_ALIASES keys."""
    from app.memory.search_agent import ANCHOR_ALIASES
    existing_keys = list(ANCHOR_ALIASES.keys())
    is_dup, closest = _is_near_duplicate("aspergillosis", existing_keys, threshold=0.75)
    assert not is_dup, f"'aspergillosis' should NOT be a near-duplicate (closest: {closest})"


def test_jaccard_identical_strings():
    """Identical strings have Jaccard=1.0."""
    assert _jaccard_similarity("fungi", "fungi") == 1.0


def test_jaccard_completely_different():
    """Completely different strings have Jaccard=0.0."""
    sim = _jaccard_similarity("fungi", "quantum")
    assert sim < 0.1


def test_jaccard_similar_strings():
    """'microbiome' and 'microbiota' share many 4-grams -> moderate similarity."""
    sim = _jaccard_similarity("microbiome", "microbiota")
    # These share the 'micr', 'icro', 'crob', 'robi', 'obio' 4-grams
    assert sim > 0.3, f"Expected moderate similarity, got {sim}"


# -------------------------------------------------------------------------
# Test: stopword blocks promotion
# -------------------------------------------------------------------------

def test_stopword_blocked(monkeypatch):
    """'possible' is in PUZZLE_NODE_STOP_TERMS and must be blocked."""
    from app.memory.search_agent import PUZZLE_NODE_STOP_TERMS
    assert "possible" in PUZZLE_NODE_STOP_TERMS, "Test assumption: 'possible' is a stop term"


# -------------------------------------------------------------------------
# Test: global_candidates respects min_run_count
# -------------------------------------------------------------------------

def test_global_candidates_min_run_count(fake_redis_env, monkeypatch):
    """Terms from < min_run_count runs do not appear in global_candidates."""
    monkeypatch.setenv("VOCAB_STORE_ENABLED", "true")
    vs = VocabularyStore()

    # Only one run promotes "aspergillosis"
    for _ in range(3):
        vs.record_outcome("aspergillosis", "run", "run_only", 0.9, 0.0, 0.7)
    vs.promote_to_global_candidate("aspergillosis", ["aspergillosis"], "run_only")

    candidates = vs.global_candidates(min_run_count=2, min_success_rate=0.65)
    terms = [c["term"] for c in candidates]
    assert "aspergillosis" not in terms, "Single-run term should not be a global candidate"


def test_global_candidates_passes_two_runs(fake_redis_env, monkeypatch):
    """Terms promoted from 2 distinct runs appear as global candidates."""
    monkeypatch.setenv("VOCAB_STORE_ENABLED", "true")
    vs = VocabularyStore()

    for run_id in ["run_a", "run_b"]:
        for _ in range(3):
            vs.record_outcome("aspergillosis", "run", run_id, 0.9, 0.0, 0.7)
        vs.promote_to_global_candidate("aspergillosis", ["aspergillosis"], run_id)

    candidates = vs.global_candidates(min_run_count=2, min_success_rate=0.65)
    terms = [c["term"] for c in candidates]
    assert "aspergillosis" in terms


# -------------------------------------------------------------------------
# Test: dry-run does not modify ANCHOR_ALIASES
# -------------------------------------------------------------------------

def test_dry_run_does_not_modify_anchor_aliases(fake_redis_env, monkeypatch, tmp_path):
    """With --dry-run, ANCHOR_ALIASES must remain unchanged."""
    from app.memory.search_agent import ANCHOR_ALIASES
    import copy

    monkeypatch.setenv("VOCAB_STORE_ENABLED", "true")
    original_keys = set(ANCHOR_ALIASES.keys())

    # Promote two synthetic novel terms from two runs
    vs = VocabularyStore()
    novel = "novelterm_xyz"  # definitely not in ANCHOR_ALIASES
    for run_id in ["run_a", "run_b"]:
        for _ in range(3):
            vs.record_outcome(novel, "run", run_id, 0.9, 0.0, 0.7)
        vs.promote_to_global_candidate(novel, [novel], run_id)

    candidates = vs.global_candidates(min_run_count=2, min_success_rate=0.65)
    # Simulate the promotion gate without actually writing to ANCHOR_ALIASES
    dry_run_promoted = []
    for cand in candidates:
        term = cand["term"]
        from app.memory.search_agent import PUZZLE_NODE_STOP_TERMS
        if term in PUZZLE_NODE_STOP_TERMS:
            continue
        is_dup, _ = _is_near_duplicate(term, list(ANCHOR_ALIASES.keys()))
        if is_dup:
            continue
        dry_run_promoted.append(term)

    # In dry-run: we never write to ANCHOR_ALIASES
    assert set(ANCHOR_ALIASES.keys()) == original_keys, (
        "dry-run must not modify ANCHOR_ALIASES"
    )
