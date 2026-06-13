import os
import unittest
from unittest.mock import patch


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


class NliPanelUnitTests(unittest.IsolatedAsyncioTestCase):
    async def test_panel_aggregates_successes_and_preserves_member_outputs(self):
        from app.memory import nli

        async def fake_model(pairs, model):
            if model == "model-a":
                return [{"label": "entailment", "entailment": 0.8, "contradiction": 0.1, "neutral": 0.1, "provider": "hf_api", "model": model}]
            return [{"label": "neutral", "entailment": 0.4, "contradiction": 0.2, "neutral": 0.4, "provider": "hf_api", "model": model}]

        with patch.object(nli.settings.memory, "nli_panel_models", "model-a,model-b"), patch.object(
            nli.settings.memory, "nli_model", "model-a"
        ), patch.object(nli.settings.memory, "nli_panel_min_successes", 1), patch.object(
            nli, "_hf_api_nli_batch_for_model", side_effect=fake_model
        ):
            result = await nli._hf_api_nli_panel_batch([("Aspirin inhibits platelets.", "Aspirin inhibits platelets.")])

        self.assertEqual(result[0]["provider"], "nli_panel")
        self.assertEqual(result[0]["panel_success_count"], 2)
        self.assertEqual(result[0]["panel_size"], 2)
        self.assertEqual(len(result[0]["panel"]), 2)
