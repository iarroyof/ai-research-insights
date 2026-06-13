"""Unit tests for WP-A: _extract_pubtator_entities (web_search.py)."""
from __future__ import annotations

import pytest
from app.memory.web_search import _extract_pubtator_entities


def test_extract_gene_and_disease():
    text = "TP53 @GENE_7157 mutation in @DISEASE_D002289 patients"
    result = _extract_pubtator_entities(text)
    assert result.get("GENE") == ["7157"]
    assert result.get("DISEASE") == ["D002289"]


def test_extract_empty_string():
    assert _extract_pubtator_entities("") == {}
    assert _extract_pubtator_entities(None) == {}


def test_extract_chemical_and_gene():
    text = "@CHEMICAL_D000086 synergy @GENE_1956"
    result = _extract_pubtator_entities(text)
    assert result.get("CHEMICAL") == ["D000086"]
    assert result.get("GENE") == ["1956"]


def test_extract_no_duplicates():
    text = "@GENE_7157 interacts with @GENE_7157 again"
    result = _extract_pubtator_entities(text)
    assert result.get("GENE") == ["7157"]  # deduplicated
