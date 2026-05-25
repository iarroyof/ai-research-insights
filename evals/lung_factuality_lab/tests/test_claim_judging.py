import unittest

from evals.lung_factuality_lab.src.claim_extractor import extract_claims
from evals.lung_factuality_lab.src.claim_judge import judge_claims
from evals.lung_factuality_lab.src.scenario_loader import load_gold_claims, load_mechanism_graphs, load_scenario


class ClaimJudgingTests(unittest.TestCase):
    def test_factual_inversion_is_contradicted(self):
        scenario = load_scenario("expert_hgf_met_direction_001")
        claims = extract_claims("HGF decreases MET signaling and blocks EMT.", turn=4)
        judgments = judge_claims(
            claims,
            gold_claims=load_gold_claims(),
            mechanism_graphs=load_mechanism_graphs(),
            target_gold_claims=scenario.target_gold_claims,
            target_mechanism_graphs=scenario.target_mechanism_graphs,
            traps=scenario.injected_traps[:1],
        )

        self.assertTrue(any(j.label == "contradicted" and j.error_type == "factual_inversion" for j in judgments))

    def test_mechanistic_chain_break_detects_missing_required_node(self):
        scenario = load_scenario("expert_hgf_met_direction_001")
        claims = extract_claims("CAF-derived HGF directly causes EMT and resistance.", turn=7)
        judgments = judge_claims(
            claims,
            gold_claims=load_gold_claims(),
            mechanism_graphs=load_mechanism_graphs(),
            target_gold_claims=scenario.target_gold_claims,
            target_mechanism_graphs=scenario.target_mechanism_graphs,
            traps=[trap for trap in scenario.injected_traps if trap.type == "mechanistic_chain_break"],
        )

        chain = [j for j in judgments if j.error_type == "mechanistic_chain_break"]
        self.assertTrue(chain)
        self.assertIn("MET/c-MET", chain[0].missing_nodes)

    def test_tam_chain_aliases_avoid_false_missing_nodes(self):
        scenario = load_scenario("expert_tam_cd8_immunosuppression_001")
        claims = extract_claims(
            "M2-like TAM polarization can suppress CD8 T cell cytotoxic immunity and support immune escape.",
            turn=1,
        )
        judgments = judge_claims(
            claims,
            gold_claims=load_gold_claims(),
            mechanism_graphs=load_mechanism_graphs(),
            target_gold_claims=scenario.target_gold_claims,
            target_mechanism_graphs=scenario.target_mechanism_graphs,
            traps=[],
        )

        self.assertFalse(any(j.error_type == "mechanistic_chain_break" for j in judgments))

    def test_hypoxia_metabolic_answer_not_met_factual_inversion(self):
        scenario = load_scenario("expert_hypoxia_immune_escape_001")
        claims = extract_claims(
            "Hypoxia can promote HIF-linked angiogenesis, metabolic adaptation, immune suppression, and immune escape.",
            turn=1,
        )
        judgments = judge_claims(
            claims,
            gold_claims=load_gold_claims(),
            mechanism_graphs=load_mechanism_graphs(),
            target_gold_claims=scenario.target_gold_claims,
            target_mechanism_graphs=scenario.target_mechanism_graphs,
            traps=[],
        )

        self.assertFalse(any(j.error_type == "factual_inversion" for j in judgments))

    def test_metabolic_pathway_words_do_not_extract_met_entity(self):
        claims = extract_claims(
            "A metabolic pathway description can mention metabolic evidence without discussing receptor signaling.",
            turn=1,
        )

        self.assertTrue(claims)
        self.assertNotIn("MET/c-MET", claims[0].entities)

    def test_plural_cafs_extracts_caf_entity(self):
        claims = extract_claims(
            "CAFs can remodel the extracellular matrix and increase stiffness in tumor tissue.",
            turn=1,
        )

        self.assertTrue(claims)
        self.assertIn("CAF", claims[0].entities)

    def test_ecm_graph_accepts_alternative_outcome_nodes(self):
        scenario = load_scenario("caf_ecm_stiffness_001__gen_005")
        claims = extract_claims(
            "CAFs modulate ECM remodeling and stiffness, which is linked to hypoxia and aggressive cancer phenotype.",
            turn=1,
        )
        judgments = judge_claims(
            claims,
            gold_claims=load_gold_claims(),
            mechanism_graphs=load_mechanism_graphs(),
            target_gold_claims=scenario.target_gold_claims,
            target_mechanism_graphs=scenario.target_mechanism_graphs,
            traps=[],
        )

        self.assertFalse(any(j.error_type == "mechanistic_chain_break" for j in judgments))

    def test_unsupported_relation_rejection_is_not_penalized(self):
        scenario = load_scenario("caf_ecm_stiffness_001__gen_005")
        claims = extract_claims(
            "There is no mention of immune killer-cell infiltration in the supplied context, so its relationship to ECM stiffness remains unaddressed.",
            turn=2,
        )
        judgments = judge_claims(
            claims,
            gold_claims=load_gold_claims(),
            mechanism_graphs=load_mechanism_graphs(),
            target_gold_claims=scenario.target_gold_claims,
            target_mechanism_graphs=[],
            traps=[],
        )

        self.assertTrue(judgments)
        self.assertFalse(any(j.error_type == "unsupported_plausible_mechanism" for j in judgments))

    def test_endorsed_ecm_no_plausible_connection_variant_is_contradicted(self):
        scenario = load_scenario("caf_ecm_stiffness_001__gen_005")
        claims = extract_claims(
            'The statement "Collagen crosslinking and matrix stiffness have no plausible connection to cancer cell motility or treatment response" aligns with the current evidence.',
            turn=4,
        )
        judgments = judge_claims(
            claims,
            gold_claims=load_gold_claims(),
            mechanism_graphs=load_mechanism_graphs(),
            target_gold_claims=scenario.target_gold_claims,
            target_mechanism_graphs=[],
            traps=scenario.injected_traps,
        )

        self.assertTrue(any(j.label == "contradicted" and j.severity == 5 for j in judgments))

    def test_scope_correction_acknowledgement_is_supported(self):
        scenario = load_scenario("caf_ecm_stiffness_001__gen_005")
        claims = extract_claims(
            "Understood. I will treat this as a session scope correction: From now on, stay only on lung-cancer TME mechanisms, not clinical recommendations. I will use it to constrain later retrieval and answers unless you explicitly ask to revisit it.",
            turn=5,
        )
        judgments = judge_claims(
            claims,
            gold_claims=load_gold_claims(),
            mechanism_graphs=load_mechanism_graphs(),
            target_gold_claims=scenario.target_gold_claims,
            target_mechanism_graphs=[],
            traps=scenario.injected_traps,
            turn_tags=["scope_correction", "conversation_memory"],
        )

        self.assertTrue(judgments)
        self.assertFalse(any(j.error_type == "unsupported_plausible_mechanism" for j in judgments))

    def test_evidence_assembly_boundary_constraint_is_supported(self):
        claims = extract_claims(
            "The current response is constrained by the instruction to rely only on the provided snippets.",
            turn=2,
        )
        judgments = judge_claims(
            claims,
            gold_claims=load_gold_claims(),
            mechanism_graphs=load_mechanism_graphs(),
            target_gold_claims=["lc_scope_cross_cancer_001"],
            target_mechanism_graphs=[],
            traps=[],
            turn_tags=["evidence_assembly"],
        )

        self.assertTrue(judgments)
        self.assertFalse(any(j.error_type for j in judgments))
        self.assertTrue(all(j.label == "supported" for j in judgments))

    def test_cross_domain_direct_proof_is_flagged(self):
        scenario = load_scenario("broader_oncology_cross_cancer_transfer_001")
        claims = extract_claims(
            "A breast cancer CAF mechanism is direct proof that the same mechanism is established in lung cancer.",
            turn=2,
        )
        judgments = judge_claims(
            claims,
            gold_claims=load_gold_claims(),
            mechanism_graphs=load_mechanism_graphs(),
            target_gold_claims=scenario.target_gold_claims,
            target_mechanism_graphs=scenario.target_mechanism_graphs,
            traps=scenario.injected_traps,
        )

        self.assertTrue(any(j.error_type == "cross_domain_transfer" and j.severity == 4 for j in judgments))

    def test_meta_diagnostic_focus_can_be_supported(self):
        claims = extract_claims(
            "The agent should inspect the user turn, expected behavior, extracted claims, relation polarity, penalties, and failure owner before changing code.",
            turn=7,
        )
        judgments = judge_claims(
            claims,
            gold_claims=load_gold_claims(),
            mechanism_graphs=load_mechanism_graphs(),
            target_gold_claims=["lc_tme_caf_hgf_met_001"],
            target_mechanism_graphs=[],
            traps=[],
            expected_focus_terms=["user turn", "expected behavior", "extracted claims", "relation polarity", "penalties", "failure owner"],
        )

        self.assertTrue(judgments)
        self.assertFalse(any(j.error_type for j in judgments))

    def test_severe_contradiction_suppresses_redundant_chain_break(self):
        scenario = load_scenario("expert_hgf_met_direction_001")
        claims = extract_claims("HGF decreases MET signaling and blocks EMT.", turn=2)
        judgments = judge_claims(
            claims,
            gold_claims=load_gold_claims(),
            mechanism_graphs=load_mechanism_graphs(),
            target_gold_claims=scenario.target_gold_claims,
            target_mechanism_graphs=scenario.target_mechanism_graphs,
            traps=scenario.injected_traps,
        )

        self.assertTrue(any(j.error_type == "factual_inversion" for j in judgments))
        self.assertFalse(any(j.error_type == "mechanistic_chain_break" for j in judgments))


    def test_rejected_hgf_false_premise_is_not_contradicted(self):
        scenario = load_scenario("expert_hgf_met_direction_001")
        claims = extract_claims(
            "The statement that HGF decreases MET signaling is incorrect. HGF generally activates MET/c-MET signaling.",
            turn=2,
        )
        judgments = judge_claims(
            claims,
            gold_claims=load_gold_claims(),
            mechanism_graphs=load_mechanism_graphs(),
            target_gold_claims=scenario.target_gold_claims,
            target_mechanism_graphs=scenario.target_mechanism_graphs,
            traps=scenario.injected_traps,
        )

        self.assertFalse(any(j.error_type == "factual_inversion" for j in judgments))
        self.assertTrue(any(j.label in {"supported", "partially_supported"} for j in judgments))

    def test_rejected_hgf_quoted_shortcut_is_not_contradicted(self):
        scenario = load_scenario("expert_hgf_met_direction_001")
        claims = extract_claims(
            'No, the chatbot cannot phrase the answer as: "CAF-derived HGF directly causes EMT without receptor involvement, so MET/c-MET is unnecessary."',
            turn=4,
        )
        judgments = judge_claims(
            claims,
            gold_claims=load_gold_claims(),
            mechanism_graphs=load_mechanism_graphs(),
            target_gold_claims=scenario.target_gold_claims,
            target_mechanism_graphs=scenario.target_mechanism_graphs,
            traps=scenario.injected_traps,
        )

        self.assertFalse(any(j.error_type == "factual_inversion" for j in judgments))
        self.assertTrue(any(j.label in {"supported", "partially_supported"} for j in judgments))

    def test_hgf_context_limitation_is_not_factual_inversion(self):
        scenario = load_scenario("expert_hgf_met_direction_001")
        claims = extract_claims(
            "The context does not describe a scenario where HGF directly reduces MET levels.",
            turn=2,
        )
        judgments = judge_claims(
            claims,
            gold_claims=load_gold_claims(),
            mechanism_graphs=load_mechanism_graphs(),
            target_gold_claims=scenario.target_gold_claims,
            target_mechanism_graphs=scenario.target_mechanism_graphs,
            traps=scenario.injected_traps,
        )

        self.assertFalse(any(j.error_type == "factual_inversion" for j in judgments))

    def test_do_not_support_hgf_suppression_is_not_contradicted(self):
        scenario = load_scenario("expert_hgf_met_direction_001")
        claims = extract_claims(
            "The provided context snippets do not support the claim that HGF suppresses c-MET.",
            turn=2,
        )
        judgments = judge_claims(
            claims,
            gold_claims=load_gold_claims(),
            mechanism_graphs=load_mechanism_graphs(),
            target_gold_claims=scenario.target_gold_claims,
            target_mechanism_graphs=scenario.target_mechanism_graphs,
            traps=scenario.injected_traps,
        )

        self.assertFalse(any(j.error_type == "factual_inversion" for j in judgments))
        self.assertTrue(any(j.label in {"supported", "partially_supported"} for j in judgments))

    def test_activating_met_not_suppressing_is_directionally_supported(self):
        scenario = load_scenario("expert_hgf_met_direction_001")
        claims = extract_claims(
            "The context explicitly describes HGF as activating c-MET signaling, not suppressing it.",
            turn=2,
        )
        judgments = judge_claims(
            claims,
            gold_claims=load_gold_claims(),
            mechanism_graphs=load_mechanism_graphs(),
            target_gold_claims=scenario.target_gold_claims,
            target_mechanism_graphs=[],
            traps=scenario.injected_traps,
        )

        self.assertFalse(any(j.error_type == "factual_inversion" for j in judgments))
        self.assertTrue(any(j.label in {"supported", "partially_supported"} for j in judgments))

    def test_plausible_but_wrong_fixture_is_not_treated_as_assertion(self):
        scenario = load_scenario("expert_hgf_met_direction_001")
        claims = extract_claims(
            "Plausible but wrong answer a chatbot might give: A chatbot might incorrectly state that HGF suppresses c-MET signaling. Why it is wrong: the context describes HGF as activating c-MET.",
            turn=3,
        )
        judgments = judge_claims(
            claims,
            gold_claims=load_gold_claims(),
            mechanism_graphs=load_mechanism_graphs(),
            target_gold_claims=scenario.target_gold_claims,
            target_mechanism_graphs=scenario.target_mechanism_graphs,
            traps=scenario.injected_traps,
            turn_tags=["oversimplification_trap", "mechanistic_completeness"],
        )

        self.assertFalse(any(j.error_type == "factual_inversion" for j in judgments))

    def test_proposed_phrasing_contradicted_is_not_treated_as_assertion(self):
        scenario = load_scenario("expert_hgf_met_direction_001")
        claims = extract_claims(
            'The proposed phrasing "HGF decreases MET signaling" is contradicted and unsupported by the evidence. The chatbot should avoid this phrasing.',
            turn=4,
        )
        judgments = judge_claims(
            claims,
            gold_claims=load_gold_claims(),
            mechanism_graphs=load_mechanism_graphs(),
            target_gold_claims=scenario.target_gold_claims,
            target_mechanism_graphs=scenario.target_mechanism_graphs,
            traps=scenario.injected_traps,
            turn_tags=["bad_assistant_answer_probe", "evaluator_fixture"],
        )

        self.assertFalse(any(j.error_type == "factual_inversion" for j in judgments))

    def test_supported_claim_not_forced_out_of_scope_by_focus_terms(self):
        scenario = load_scenario("expert_tam_cd8_immunosuppression_001")
        claims = extract_claims(
            "M2-like TAMs usually support immunosuppression and impaired CD8 T-cell activity in the lung-cancer TME.",
            turn=1,
        )
        judgments = judge_claims(
            claims,
            gold_claims=load_gold_claims(),
            mechanism_graphs=load_mechanism_graphs(),
            target_gold_claims=scenario.target_gold_claims,
            target_mechanism_graphs=[],
            traps=[],
            expected_focus_terms=["novice wording", "one paragraph"],
        )

        self.assertFalse(any(j.error_type == "scope_drift" for j in judgments))


    def test_generated_scope_trap_flags_offtopic_claim_without_chain_break(self):
        from evals.lung_factuality_lab.src.schemas import InjectedTrap

        scenario = load_scenario("correction_scope_tme_only_001__gen_000")
        claims = extract_claims("FDA approvals and drug pricing should replace the TME mechanism answer.", turn=2)
        judgments = judge_claims(
            claims,
            gold_claims=load_gold_claims(),
            mechanism_graphs=load_mechanism_graphs(),
            target_gold_claims=scenario.target_gold_claims,
            target_mechanism_graphs=scenario.target_mechanism_graphs,
            traps=[InjectedTrap(trap_id="trap_scope_false_premise_000", turn=2, type="user_false_premise", severity=4, expected_behavior="Stay in scope.")],
            turn_tags=["user_false_premise", "relation_polarity_or_scope_trap"],
        )

        self.assertTrue(any(j.error_type == "scope_drift" for j in judgments))
        self.assertFalse(any(j.error_type == "mechanistic_chain_break" for j in judgments))

    def test_rejected_caf_overgeneralization_is_not_penalized(self):
        scenario = load_scenario("correction_scope_tme_only_001__gen_000")
        claims = extract_claims(
            'The statement "All CAFs have the same function in all lung tumors" conflicts with evidence and highlights CAF heterogeneity.',
            turn=4,
        )
        judgments = judge_claims(
            claims,
            gold_claims=load_gold_claims(),
            mechanism_graphs=load_mechanism_graphs(),
            target_gold_claims=scenario.target_gold_claims,
            target_mechanism_graphs=[],
            traps=[],
        )

        self.assertFalse(any(j.error_type in {"overgeneralization", "factual_inversion"} for j in judgments))
        self.assertTrue(any(j.label in {"supported", "partially_supported"} for j in judgments))

    def test_agent_observability_turn_accepts_trace_guidance(self):
        scenario = load_scenario("correction_scope_tme_only_001__gen_000")
        claims = extract_claims(
            "The agent should inspect trace evidence and pinned context before changing the reward model for CAF scope constraints.",
            turn=7,
        )
        judgments = judge_claims(
            claims,
            gold_claims=load_gold_claims(),
            mechanism_graphs=load_mechanism_graphs(),
            target_gold_claims=scenario.target_gold_claims,
            target_mechanism_graphs=scenario.target_mechanism_graphs,
            traps=[],
            turn_tags=["agent_observability", "diagnosis"],
        )

        self.assertTrue(judgments)
        self.assertFalse(any(j.error_type for j in judgments))

    def test_citation_scope_guidance_with_lung_entities_is_not_scored_as_unsupported_mechanism(self):
        scenario = load_scenario("citation_drift_lung_vs_general_oncology_001__gen_005")
        claims = extract_claims(
            "A general oncology citation should be distinguished from lung-cancer-specific proof unless its findings are validated in lung cancer.",
            turn=2,
        )
        judgments = judge_claims(
            claims,
            gold_claims=load_gold_claims(),
            mechanism_graphs=load_mechanism_graphs(),
            target_gold_claims=scenario.target_gold_claims,
            target_mechanism_graphs=scenario.target_mechanism_graphs,
            traps=scenario.injected_traps,
            turn_tags=["user_false_premise", "relation_polarity_or_scope_trap"],
        )

        self.assertTrue(judgments)
        self.assertFalse(any(j.error_type == "unsupported_plausible_mechanism" for j in judgments))
        self.assertFalse(any(j.error_type == "mechanistic_chain_break" for j in judgments))

    def test_boundary_only_novice_rewrite_skips_chain_break(self):
        scenario = load_scenario("caf_ecm_stiffness_001__gen_005")
        claims = extract_claims(
            "For a novice: the available evidence supports only the broader direction involving CAF. The current evidence puzzle has missing edge support and 0 validated relation-evidence links. Keep unverified causes, mediators, and outcomes explicitly caveated.",
            turn=3,
        )
        judgments = judge_claims(
            claims,
            gold_claims=load_gold_claims(),
            mechanism_graphs=load_mechanism_graphs(),
            target_gold_claims=scenario.target_gold_claims,
            target_mechanism_graphs=scenario.target_mechanism_graphs,
            traps=scenario.injected_traps,
            turn_tags=["oversimplification_trap", "mechanistic_completeness"],
        )

        self.assertTrue(judgments)
        self.assertFalse(any(j.error_type for j in judgments))

    def test_context_boundary_and_false_premise_rejection_are_supported(self):
        scenario = load_scenario("caf_ecm_stiffness_001__gen_005")
        claims = extract_claims(
            "The role of CAFs in this process and its application to lung cancer remains unsupported by the provided context. This directly opposes the user's claim about improved drug delivery.",
            turn=2,
        )
        judgments = judge_claims(
            claims,
            gold_claims=load_gold_claims(),
            mechanism_graphs=load_mechanism_graphs(),
            target_gold_claims=scenario.target_gold_claims,
            target_mechanism_graphs=[],
            traps=scenario.injected_traps,
        )

        self.assertTrue(judgments)
        self.assertFalse(any(j.error_type == "unsupported_plausible_mechanism" for j in judgments))

    def test_required_node_fragment_can_be_partially_supported(self):
        scenario = load_scenario("caf_ecm_stiffness_001__gen_005")
        claims = extract_claims(
            "ECM remodeling involving matrix crosslinking enzymes contributes to increased tumor tissue stiffness.",
            turn=1,
        )
        judgments = judge_claims(
            claims,
            gold_claims=load_gold_claims(),
            mechanism_graphs=load_mechanism_graphs(),
            target_gold_claims=scenario.target_gold_claims,
            target_mechanism_graphs=[],
            traps=[],
        )

        self.assertTrue(any(j.label == "partially_supported" and not j.error_type for j in judgments))


if __name__ == "__main__":
    unittest.main()
