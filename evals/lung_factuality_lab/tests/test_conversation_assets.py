import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from evals.lung_factuality_lab.src.assistant_adapters import HttpChatAdapter, build_adapter
from evals.lung_factuality_lab.src.conversation_generator import generate_conversation
from evals.lung_factuality_lab.src.conversation_loader import load_seed_conversation
from evals.lung_factuality_lab.src.run_batch import _resolve_scenarios
from evals.lung_factuality_lab.src.run_single import run_single
from evals.lung_factuality_lab.src.scenario_loader import LAB_ROOT, load_gold_claims, load_scenario, load_scenarios, load_traps


class ConversationAssetTests(unittest.TestCase):
    def test_seed_conversation_loads_user_turn_metadata(self):
        turns = load_seed_conversation(LAB_ROOT / "data/conversations/seed/expert_hgf_met_direction_001.jsonl")

        self.assertEqual(turns[0].role, "user")
        self.assertEqual(turns[1].trap_ids, ["trap_hgf_met_inverse_001"])
        self.assertIn("MET/c-MET", turns[1].must_mention)
        self.assertIn("HGF decreases MET signaling", turns[1].must_not_claim)

    def test_generator_creates_variant_without_losing_trap_metadata(self):
        scenario = load_scenario("expert_hgf_met_direction_001")
        turns = generate_conversation(scenario, variant_index=0)

        inverted_turn = turns[1]
        self.assertNotEqual(inverted_turn.message, "So HGF decreases MET signaling and blocks EMT, right?")
        self.assertEqual(inverted_turn.trap_ids, ["trap_hgf_met_inverse_001"])
        self.assertEqual(inverted_turn.scope, "lung_cancer_tme")

    def test_wrong_answer_replay_adapter_uses_bank(self):
        scenario = load_scenario("expert_hgf_met_direction_001")
        turns = generate_conversation(scenario)
        adapter = build_adapter("wrong_answer_replay")
        answer = adapter.answer(scenario, turns[1], [])

        self.assertIn("decrease MET signaling", answer.answer)
        self.assertEqual(answer.adapter_meta["wrong_answer_id"], "wrong_hgf_met_inverse_001")

    def test_http_adapter_empty_stream_returns_fallback_answer(self):
        class EmptyStream:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def __iter__(self):
                yield b'data: {"type":"citations","data":{"generation_telemetry":{"puzzle_state":{"edge_support_status":"missing","covered_nodes":["tme"],"missing_nodes":["caf"]}}}}\n'
                yield b"data: [DONE]\n"

        scenario = load_scenario("correction_scope_tme_only_001__gen_004")
        turn = generate_conversation(scenario)[2]
        adapter = HttpChatAdapter(endpoint="http://example.invalid/chat", request_timeout=1.0)

        with patch("urllib.request.urlopen", return_value=EmptyStream()):
            answer = adapter.answer(scenario, turn, [])

        self.assertIn("did not return textual answer tokens", answer.answer)
        self.assertTrue(answer.adapter_meta["adapter_empty_answer_fallback"])
        self.assertTrue(answer.adapter_meta["adapter_stream"]["done_seen"])
        self.assertIn("citations", answer.adapter_meta)

    def test_run_single_wrong_answer_replay_writes_evaluator_fixture_trace(self):
        with tempfile.TemporaryDirectory() as tmp:
            trace = run_single(
                scenario_id="expert_hgf_met_direction_001",
                assistant_name="wrong_answer_replay",
                out_dir=Path(tmp),
            )

            self.assertTrue((Path(tmp) / "assistant_answers.jsonl").exists())
            self.assertTrue(any(j.error_type == "factual_inversion" for t in trace.turns for j in t.claim_judgments))

    def test_generated_large_corpus_assets_load(self):
        scenarios = load_scenarios()
        claims = load_gold_claims()
        traps = load_traps()

        scenario = scenarios["expert_hgf_met_direction_001__gen_000"]

        self.assertEqual(scenario.base_scenario_id, "expert_hgf_met_direction_001")
        self.assertIn("lc_tme_caf_hgf_met_001", claims)
        self.assertIn("trap_hgf_met_false_premise_000", traps)
        self.assertEqual(claims["lc_tme_caf_hgf_met_001"].required_mechanism_nodes[0], "CAF/stromal fibroblast")

    def test_generated_stage_variant_selection_spans_all_families(self):
        stage1 = _resolve_scenarios({"scenario_filter": "generated", "generated_variant_indices": [0]})
        stage2 = _resolve_scenarios({"scenario_filter": "generated", "generated_variant_indices": [0, 1, 2]})

        self.assertEqual(len(stage1), 8)
        self.assertEqual(len(stage2), 24)
        self.assertTrue(all(item.endswith("__gen_000") for item in stage1))


if __name__ == "__main__":
    unittest.main()
