# services/api/tests/test_prompt_cache_ordering.py
"""P-5: cache-friendly prompt ordering invariant.

Every dynamic system-prompt factory must put its STATIC base constant first and
the per-turn dynamic part last. That way NIM/transformer KV-cache prefix matching
hits for the constant leading tokens even as the dynamic suffix changes each turn.
These tests lock the invariant so a future edit can't silently reintroduce a
dynamic-first ordering (which would defeat prefix caching — DEVELOPMENT_STATUS P-5).
"""
import unittest

from app.prompts import agent_prompts as ap


class StaticPrefixOrderingTests(unittest.TestCase):
    def test_answer_static_prefix_first(self):
        for mode in ("direct_answer", "novice_rewrite", "expert_mechanism",
                     "phrase_evaluation", "diagnostic_trace_answer",
                     "correction_acknowledgement", "clarification"):
            self.assertTrue(
                ap.answer_system_prompt(mode).startswith(ap._ANSWER_BASE_POLICY),
                f"answer mode {mode!r} not static-prefix-first",
            )

    def test_answer_modes_share_static_prefix(self):
        # Different modes must share the full static base as a common prefix.
        a = ap.answer_system_prompt("direct_answer")
        b = ap.answer_system_prompt("expert_mechanism")
        self.assertTrue(a.startswith(ap._ANSWER_BASE_POLICY))
        self.assertTrue(b.startswith(ap._ANSWER_BASE_POLICY))

    def test_frame_static_prefix_first(self):
        for intent in ("new_query", "augment_prior"):
            self.assertTrue(ap.frame_system_prompt(intent).startswith(ap._FRAME_BASE))

    def test_intent_resolution_static_prefix_first(self):
        for args in ((True, True, ["egfr"]), (False, False, []), (True, False, [])):
            self.assertTrue(
                ap.intent_resolution_system_prompt(*args).startswith(ap._INTENT_RESOLUTION_BASE)
            )

    def test_ner_static_prefix_first(self):
        self.assertTrue(ap.ner_grounding_system_prompt(True).startswith(ap._NER_BASE))
        self.assertTrue(ap.ner_grounding_system_prompt(False, 3).startswith(ap._NER_BASE))

    def test_reflection_static_prefix_first(self):
        for polarity in ("positive", "negative", "mixed"):
            self.assertTrue(
                ap.reflection_system_prompt(polarity).startswith(ap._REFLECTION_BASE)
            )


if __name__ == "__main__":
    unittest.main()
