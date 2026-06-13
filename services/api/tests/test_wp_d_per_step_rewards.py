"""Unit tests for WP-D: per-step reward functions (rewards.py)."""
from __future__ import annotations

import pytest
from app.memory.rewards import (
    distractor_ratio,
    gap_closure_score,
    query_novelty,
    reward_report,
)


# ----- query_novelty ---------------------------------------------------------

def test_query_novelty_first_query():
    assert query_novelty({"egfr", "nsclc"}, []) == 1.0


def test_query_novelty_partial():
    prev = [{"egfr", "nsclc"}]
    current = {"egfr", "alk"}
    result = query_novelty(current, prev)
    # alk is novel (1 out of 2 terms)
    assert abs(result - 0.5) < 1e-9


def test_query_novelty_exact_repeat():
    terms = {"egfr", "nsclc"}
    assert query_novelty(terms, [terms]) == 0.0


def test_query_novelty_empty_current():
    assert query_novelty(set(), [{"egfr"}]) == 0.0


# ----- gap_closure_score -----------------------------------------------------

class _GS:
    """Minimal duck-typed GapSpec for testing."""
    def __init__(self, confirmed, missing):
        self.confirmed_entities = set(confirmed)
        self.missing_entities = set(missing)


def test_gap_closure_half():
    before = _GS(confirmed=set(), missing={"EGFR", "ALK"})
    after  = _GS(confirmed={"EGFR"}, missing={"ALK"})
    assert abs(gap_closure_score(before, after) - 0.5) < 1e-9


def test_gap_closure_nothing_missing():
    before = _GS(confirmed={"EGFR"}, missing=set())
    after  = _GS(confirmed={"EGFR", "ALK"}, missing=set())
    assert gap_closure_score(before, after) == 0.0


def test_gap_closure_all_confirmed():
    before = _GS(confirmed=set(), missing={"EGFR", "ALK"})
    after  = _GS(confirmed={"EGFR", "ALK"}, missing=set())
    assert gap_closure_score(before, after) == 1.0


# ----- distractor_ratio -------------------------------------------------------

def test_distractor_ratio_one_of_three():
    snippets = [
        {"text": "EGFR mutation drives resistance"},  # has anchor
        {"text": "ALK fusion detected"},               # has gap entity
        {"text": "unrelated topic about weather"},      # distractor
    ]
    anchors = {"egfr"}
    gs = _GS(confirmed=set(), missing={"alk"})
    ratio = distractor_ratio(snippets, anchors, gap_spec=gs)
    assert abs(ratio - 1/3) < 1e-9


def test_distractor_ratio_empty():
    assert distractor_ratio([], {"egfr"}, gap_spec=None) == 0.0


def test_distractor_ratio_all_distractors():
    snippets = [{"text": "weather forecast"}, {"text": "stock market"}]
    ratio = distractor_ratio(snippets, {"egfr"}, gap_spec=None)
    assert ratio == 1.0


# ----- reward_report step_rewards integration --------------------------------

def _dummy_reward_report_kwargs():
    return dict(
        question="What is the role of EGFR?",
        answer="EGFR drives downstream signalling.",
        selected_context=[{"text": "EGFR promotes cell growth", "role": "search"}],
        conflicts=[],
        elapsed_sec=1.0,
        token_budget=500,
    )


def test_reward_report_with_step_rewards():
    step_rewards = [
        {"query_novelty": 1.0, "gap_closure_score": 0.8, "distractor_ratio": 0.1}
    ]
    result = reward_report(**_dummy_reward_report_kwargs(), step_rewards=step_rewards)
    assert "avg_gap_closure_score" in result
    assert "avg_distractor_ratio" in result
    assert "avg_query_novelty" in result
    assert 0.0 <= result["score"] <= 1.0


def test_reward_report_no_step_rewards():
    result = reward_report(**_dummy_reward_report_kwargs())
    # keys should still exist, default to 0.0
    assert result.get("avg_gap_closure_score", 0.0) == 0.0


def test_reward_report_gap_closure_raises_score():
    # High gap_closure + low distractor should push score up vs 0 step rewards
    base = reward_report(**_dummy_reward_report_kwargs())
    with_gaps = reward_report(
        **_dummy_reward_report_kwargs(),
        step_rewards=[{"query_novelty": 0.9, "gap_closure_score": 1.0, "distractor_ratio": 0.0}],
    )
    assert with_gaps["score"] >= base["score"]
