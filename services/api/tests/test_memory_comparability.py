import unittest

from app.memory.comparability import compare_premise_hypothesis


ASPIRIN_PREMISE = "Aspirin inhibits platelet aggregation and is used to reduce thrombotic risk."


class ComparabilityTests(unittest.TestCase):
    def test_entailment_fixture_is_comparable(self):
        result = compare_premise_hypothesis(ASPIRIN_PREMISE, "Aspirin inhibits platelet aggregation.")
        self.assertIs(result.comparable, True)
        self.assertEqual(result.blocking_mismatch, "none")

    def test_contradiction_fixture_remains_comparable_despite_negation(self):
        result = compare_premise_hypothesis(ASPIRIN_PREMISE, "Aspirin does not inhibit platelet aggregation.")
        self.assertIs(result.comparable, True)
        self.assertIn("negation_mismatch", result.reasons)

    def test_neutral_fixture_is_not_comparable(self):
        result = compare_premise_hypothesis(ASPIRIN_PREMISE, "Metformin improves insulin sensitivity.")
        self.assertIs(result.comparable, False)
        self.assertIn(result.blocking_mismatch, {"entity", "disease", "relation"})

    def test_wrong_evidence_blocks_before_nli(self):
        result = compare_premise_hypothesis(
            "Study reports platelet aggregation in cardiovascular disease.",
            "A cancer biomarker predicts chemotherapy response.",
        )
        self.assertIs(result.comparable, False)
        self.assertIn(result.blocking_mismatch, {"disease", "entity"})
