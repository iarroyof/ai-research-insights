"""Unit tests for WP-B (GapSpec) and WP-C (_snippet_utility) in search_agent.py."""
from __future__ import annotations

import pytest
from app.memory.search_agent import GapSpec, _snippet_utility


# ----- GapSpec ---------------------------------------------------------------

def test_gap_spec_coverage_ratio():
    gs = GapSpec()
    gs.confirmed_entities = {"EGFR", "NSCLC"}
    gs.missing_entities = {"ALK"}
    gs.update_coverage()
    assert abs(gs.coverage_ratio - 2/3) < 1e-9


def test_gap_spec_coverage_empty():
    gs = GapSpec()
    gs.update_coverage()
    assert gs.coverage_ratio == 0.0


def test_gap_spec_to_dict_keys():
    gs = GapSpec()
    gs.confirmed_entities = {"X"}
    gs.missing_entities = {"Y"}
    gs.update_coverage()
    d = gs.to_dict()
    assert "confirmed_entities" in d
    assert "missing_entities" in d
    assert "coverage_ratio" in d
    assert d["confirmed_entities"] == ["X"]
    assert d["missing_entities"] == ["Y"]


# ----- _snippet_utility -------------------------------------------------------

def test_snippet_utility_ordering():
    snippets = [
        {"text": "EGFR drives signalling in NSCLC patients", "retrieval_score": 8.0},
        {"text": "random text about dogs", "retrieval_score": 1.0},
        {"text": "EGFR mutation detected", "retrieval_score": 5.0},
    ]
    anchors = {"egfr", "nsclc"}
    scores = [_snippet_utility(s, anchors) for s in snippets]
    # First snippet (high score + both anchors) should rank highest
    assert scores[0] > scores[1]
    assert scores[2] > scores[1]


def test_snippet_utility_bm25_clamp():
    snippet = {"text": "irrelevant", "retrieval_score": 100.0}
    score = _snippet_utility(snippet, set())
    # bm25_norm should clamp at 1.0, so score max = 0.35
    assert score <= 0.36


def test_snippet_utility_gap_closing():
    gs = GapSpec()
    gs.missing_entities = {"alk", "ros1"}
    snippet_relevant = {"text": "ALK fusion and ROS1 rearrangement", "retrieval_score": 0.0}
    snippet_irrelevant = {"text": "weather report", "retrieval_score": 0.0}
    s_rel = _snippet_utility(snippet_relevant, set(), gap_spec=gs)
    s_irr = _snippet_utility(snippet_irrelevant, set(), gap_spec=gs)
    assert s_rel > s_irr
