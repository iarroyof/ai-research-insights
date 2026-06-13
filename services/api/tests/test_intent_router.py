# services/api/tests/test_intent_router.py
"""Tests for the tier-1 zero-shot intent router (P-7).

Covers the plumbing: NIM-primary parsing, MNLI fallback, hypothesis->label
mapping, and the both-fail path. The semantic quality (typo/paraphrase
robustness) is a model property validated by the real-provider smoke test, not
asserted here.
"""
import unittest
from unittest import mock

from app.memory import intent_router as ir
from app.prompts.agent_prompts import ROUTER_INTENT_HYPOTHESES as H


class FakeLLM:
    """Stand-in for LLMClient: returns canned text or raises."""

    def __init__(self, text=None, exc=None):
        self._text = text
        self._exc = exc

    async def chat_once(self, messages, *, agent=None, max_tokens=None, **kw):
        if self._exc:
            raise self._exc
        return self._text


class ParseNimTests(unittest.TestCase):
    def test_json_label_and_confidence(self):
        r = ir._parse_nim('{"intent": "prior_context", "confidence": 0.91}')
        self.assertEqual(r["intent"], "prior_context")
        self.assertAlmostEqual(r["confidence"], 0.91)
        self.assertEqual(r["source"], "nim")

    def test_json_missing_confidence_defaults(self):
        r = ir._parse_nim('{"intent": "new_query"}')
        self.assertEqual(r["intent"], "new_query")
        self.assertEqual(r["confidence"], 0.85)

    def test_confidence_clamped(self):
        r = ir._parse_nim('{"intent": "new_query", "confidence": 9}')
        self.assertEqual(r["confidence"], 1.0)

    def test_label_token_fallback(self):
        r = ir._parse_nim("The intent is augment_prior here.")
        self.assertEqual(r["intent"], "augment_prior")
        self.assertEqual(r["confidence"], 0.7)

    def test_invalid_returns_none(self):
        self.assertIsNone(ir._parse_nim("no idea"))
        self.assertIsNone(ir._parse_nim(""))

    def test_unknown_label_in_json_returns_none(self):
        self.assertIsNone(ir._parse_nim('{"intent": "banana"}'))


class PremiseTests(unittest.TestCase):
    def test_includes_recent_turn_and_message(self):
        notes = [{"recent_turns": ["A: prev answer", "U: prev question"]}]
        p = ir._premise("yes", notes)
        self.assertIn("U: prev question", p)
        self.assertIn("User: yes", p)

    def test_no_notes(self):
        self.assertEqual(ir._premise("yes", None), "User: yes")


ROUTER_LABELS = ("prior_context", "new_query", "augment_prior")


def _mnli_scores(top_label, top=0.9):
    """Build a score_labels-shaped return mapping hypotheses to probabilities."""
    out = {H[k]: 0.15 for k in ROUTER_LABELS}
    out[H[top_label]] = top
    return [out]


class ClassifyTests(unittest.IsolatedAsyncioTestCase):
    async def test_nim_primary_used(self):
        with mock.patch.object(
            ir, "LLMClient",
            lambda: FakeLLM(text='{"intent":"prior_context","confidence":0.8}'),
        ):
            r = await ir.classify_intent_zeroshot("yes", None)
        self.assertEqual(r["intent"], "prior_context")
        self.assertEqual(r["source"], "nim")

    async def test_mnli_fallback_when_nim_fails(self):
        with mock.patch.object(
            ir, "LLMClient", lambda: FakeLLM(exc=RuntimeError("nim down"))
        ), mock.patch.object(
            ir.zero_shot, "score_labels",
            lambda texts, labels: _mnli_scores("prior_context", 0.92),
        ):
            r = await ir.classify_intent_zeroshot("those", None)
        self.assertEqual(r["intent"], "prior_context")
        self.assertEqual(r["source"], "mnli")
        self.assertAlmostEqual(r["confidence"], 0.92)

    async def test_nim_invalid_falls_through_to_mnli(self):
        with mock.patch.object(
            ir, "LLMClient", lambda: FakeLLM(text="garbage with no label")
        ), mock.patch.object(
            ir.zero_shot, "score_labels",
            lambda texts, labels: _mnli_scores("new_query", 0.77),
        ):
            r = await ir.classify_intent_zeroshot("what regulates EGFR", None)
        self.assertEqual(r["intent"], "new_query")
        self.assertEqual(r["source"], "mnli")

    async def test_both_backends_fail_returns_none(self):
        def boom(*a, **k):
            raise RuntimeError("hf down")

        with mock.patch.object(
            ir, "LLMClient", lambda: FakeLLM(exc=RuntimeError("nim down"))
        ), mock.patch.object(ir.zero_shot, "score_labels", boom):
            r = await ir.classify_intent_zeroshot("yes", None)
        self.assertIsNone(r)


if __name__ == "__main__":
    unittest.main()
