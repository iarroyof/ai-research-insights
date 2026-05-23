import unittest

from evals.lung_factuality_lab.src.claim_extractor import extract_claims
from evals.lung_factuality_lab.src.claim_judge import judge_claims
from evals.lung_factuality_lab.src.reward_scorer import score_turn
from evals.lung_factuality_lab.src.scenario_loader import load_gold_claims, load_mechanism_graphs, load_reward_config, load_scenario


class RewardPenaltyTests(unittest.TestCase):
    def test_inversion_penalized_more_than_vague_answer(self):
        scenario = load_scenario("expert_hgf_met_direction_001")
        config = load_reward_config()
        inverted = judge_claims(
            extract_claims("HGF decreases MET signaling and blocks EMT.", turn=4),
            gold_claims=load_gold_claims(),
            mechanism_graphs=load_mechanism_graphs(),
            target_gold_claims=scenario.target_gold_claims,
            target_mechanism_graphs=[],
            traps=scenario.injected_traps,
        )
        vague = judge_claims(
            extract_claims("The tumor microenvironment affects cancer progression.", turn=1),
            gold_claims=load_gold_claims(),
            mechanism_graphs=load_mechanism_graphs(),
            target_gold_claims=scenario.target_gold_claims,
            target_mechanism_graphs=[],
            traps=[],
        )

        inverted_score = score_turn(turn=4, judgments=inverted, traps=scenario.injected_traps, reward_config=config)
        vague_score = score_turn(turn=1, judgments=vague, traps=[], reward_config=config)

        self.assertLess(inverted_score.turn_reward, vague_score.turn_reward)


    def test_empty_answer_is_penalized_below_vague_answer(self):
        config = load_reward_config()
        empty = [
            __import__("evals.lung_factuality_lab.src.schemas", fromlist=["ClaimJudgment"]).ClaimJudgment(
                claim_id="empty_answer_1",
                label="unsupported",
                reason="Assistant returned an empty answer for this turn.",
                confidence=1.0,
                error_type="empty_answer",
                severity=4,
            )
        ]
        vague = judge_claims(
            extract_claims("The tumor microenvironment affects cancer progression.", turn=1),
            gold_claims=load_gold_claims(),
            mechanism_graphs=load_mechanism_graphs(),
            target_gold_claims=["lc_tme_caf_hgf_met_001"],
            target_mechanism_graphs=[],
            traps=[],
        )

        empty_score = score_turn(turn=1, judgments=empty, traps=[], reward_config=config)
        vague_score = score_turn(turn=1, judgments=vague, traps=[], reward_config=config)

        self.assertLess(empty_score.turn_reward, vague_score.turn_reward)

    def test_repeated_same_penalty_is_capped_by_type(self):
        ClaimJudgment = __import__("evals.lung_factuality_lab.src.schemas", fromlist=["ClaimJudgment"]).ClaimJudgment
        config = load_reward_config()
        judgments = [
            ClaimJudgment(claim_id=f"j{i}", label="out_of_scope", reason="scope", confidence=1.0, error_type="scope_drift", severity=3)
            for i in range(3)
        ]

        score = score_turn(turn=1, judgments=judgments, traps=[], reward_config=config)

        self.assertEqual(len(score.penalties_applied), 1)
        self.assertEqual(score.penalties_applied[0]["type"], "scope_drift")
        self.assertEqual(score.penalties_applied[0]["count"], 3)

    def test_live_evidence_assembly_telemetry_penalizes_unsupported_bridge(self):
        ClaimJudgment = __import__("evals.lung_factuality_lab.src.schemas", fromlist=["ClaimJudgment"]).ClaimJudgment
        config = load_reward_config()
        supported = [
            ClaimJudgment(claim_id="partial", label="partially_supported", reason="partial", confidence=0.9)
        ]
        unsupported = [
            ClaimJudgment(
                claim_id="bridge",
                label="unsupported",
                reason="unsupported bridge",
                confidence=0.9,
                error_type="unsupported_plausible_mechanism",
                severity=3,
            )
        ]
        partial_score = score_turn(
            turn=1,
            judgments=supported,
            traps=[],
            reward_config=config,
            search_telemetry={"evidence_puzzle": {"edge_support_status": "partial"}, "assembly_quality": 0.7},
        )
        bridge_score = score_turn(
            turn=1,
            judgments=unsupported,
            traps=[],
            reward_config=config,
            search_telemetry={"evidence_puzzle": {"edge_support_status": "missing"}, "assembly_quality": 0.2},
        )

        self.assertGreater(partial_score.turn_reward, bridge_score.turn_reward)
        self.assertIn("evidence_bridge_safety", bridge_score.component_scores)


if __name__ == "__main__":
    unittest.main()
