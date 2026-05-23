import os
import unittest


@unittest.skipUnless(os.environ.get("RUN_HF_SMOKE") == "1" and os.environ.get("HF_API_TOKEN"), "HF smoke test requires RUN_HF_SMOKE=1 and HF_API_TOKEN")
class HfNliSmokeTests(unittest.IsolatedAsyncioTestCase):
    async def test_hf_api_smoke_fixtures(self):
        from app.memory.nli import classify_nli

        premise = "Aspirin inhibits platelet aggregation and is used to reduce thrombotic risk."

        entailment = await classify_nli(premise, "Aspirin inhibits platelet aggregation.")
        assert entailment["label"] == "entailment"
        assert entailment["entailment"] > entailment["contradiction"]

        contradiction = await classify_nli(premise, "Aspirin does not inhibit platelet aggregation.")
        assert contradiction["label"] == "contradiction"
        assert contradiction["contradiction"] > contradiction["entailment"]
