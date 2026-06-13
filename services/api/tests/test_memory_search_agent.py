import unittest

from app.config import settings
from app.memory.search_agent import (
    _fallback_queries_from_text,
    _feedback_term_report,
    _feedback_terms_from_results,
    _hit_anchor_coverage,
    _is_off_topic_hit,
    build_auto_context,
    deterministic_query_variants,
    _domain_search_frame,
    _evidence_assembly,
    plan_auto_context,
    search_state_key,
)


class FakeStore:
    async def search_policy_notes(self, **kwargs):
        return [{"note": "Broaden short biomedical questions with synonyms and alternate disease names."}]

    async def action_values(self, **kwargs):
        return []

    async def conversation_frame(self, session_id):
        return {}


class FakeFrameStore(FakeStore):
    async def search_policy_notes(self, **kwargs):
        return []

    async def conversation_frame(self, session_id):
        return {
            "active_terms": [
                "fungi",
                "tumorigenesis",
                "how",
                "happens",
                "supplied",
                "context",
            ]
        }


async def fake_search(tenant, query, filters, k):
    base = [
        {
            "paper_id": "paper-1",
            "sent_id": "s1",
            "text": "PD-L1 expression is associated with checkpoint inhibitor response in NSCLC.",
            "score": 3.0,
        },
        {
            "paper_id": "paper-1",
            "sent_id": "s1",
            "text": "PD-L1 expression is associated with checkpoint inhibitor response in NSCLC.",
            "score": 2.5,
        },
        {
            "paper_id": "paper-2",
            "sent_id": "s2",
            "text": "Non-small cell lung cancer trials often report immune checkpoint outcomes.",
            "score": 2.0,
        },
    ]
    return base[:k]


class FakeMultilevelSearch:
    def __init__(self):
        self.calls = []

    async def __call__(self, tenant, level, query, filters, k):
        self.calls.append({"level": level, "query": query, "filters": filters, "k": k})
        if level == "title":
            return [
                {
                    "paper_id": "paper-title",
                    "sent_id": "title",
                    "title": "Nivolumab and PD-L1 biomarkers in NSCLC",
                    "text": "Nivolumab and PD-L1 biomarkers in NSCLC",
                    "score": 5.0,
                    "search_level": "title",
                }
            ][:k]
        if level == "paper":
            return [
                {
                    "paper_id": "paper-paper",
                    "sent_id": "abstract",
                    "title": "Immune checkpoint treatment",
                    "text": "This paper reviews nivolumab, pembrolizumab, PD-L1, and NSCLC response biomarkers.",
                    "score": 4.0,
                    "search_level": "paper",
                }
            ][:k]
        return [
            {
                "paper_id": "paper-sentence",
                "sent_id": "s1",
                "title": "PD-L1 response evidence",
                "text": "PD-L1 expression is associated with checkpoint inhibitor response in NSCLC.",
                "score": 3.0,
                "search_level": "sentence",
            },
            {
                "paper_id": "paper-sentence",
                "sent_id": "s1",
                "title": "PD-L1 response evidence",
                "text": "PD-L1 expression is associated with checkpoint inhibitor response in NSCLC.",
                "score": 2.5,
                "search_level": "sentence",
            },
        ][:k]


class FakeNoisyMultilevelSearch:
    def __init__(self):
        self.calls = []

    async def __call__(self, tenant, level, query, filters, k):
        self.calls.append({"level": level, "query": query, "filters": filters, "k": k})
        if level == "title":
            return [
                {
                    "paper_id": "paper-survey",
                    "sent_id": "title",
                    "title": "Administered questionnaire formats and radio button free text questions",
                    "text": "Four different questionnaire formats used radio buttons and free text.",
                    "score": 1.0,
                    "search_level": "title",
                }
            ][:k]
        if level == "paper":
            return [
                {
                    "paper_id": "paper-metabolism",
                    "sent_id": "abstract",
                    "title": "Dietary metabolism and local tissue acidity",
                    "text": "Dietary metabolic exposures and lactate can affect local tissue environments.",
                    "score": 3.0,
                    "search_level": "paper",
                }
            ][:k]
        return [
            {
                "paper_id": "paper-lactate",
                "sent_id": "s1",
                "title": "Lactate and acidic tissue environments",
                "text": "Lactate accumulation can lower extracellular pH in metabolically active tissue.",
                "score": 4.0,
                "search_level": "sentence",
            }
        ][:k]


class FakeFungiMultilevelSearch:
    def __init__(self):
        self.calls = []

    async def __call__(self, tenant, level, query, filters, k):
        self.calls.append({"level": level, "query": query, "filters": filters, "k": k})
        if level == "title":
            return [
                {
                    "paper_id": "paper-nfkb",
                    "sent_id": "nfkb",
                    "title": "NF-kB and inflammation in tumorigenesis",
                    "text": "NF-kB regulates immune response and inflammation and supports a major role in tumorigenesis.",
                    "score": 5.0,
                    "search_level": "title",
                },
                {
                    "paper_id": "paper-fungi",
                    "sent_id": "fungi",
                    "title": "Fungi and other microbes in tumorigenesis",
                    "text": "Other microbes also play essential roles in tumorigenesis, including fungi.",
                    "score": 4.0,
                    "search_level": "title",
                },
            ][:k]
        return []


class MemorySearchAgentTests(unittest.IsolatedAsyncioTestCase):
    def test_state_key_buckets_do_not_embed_query_terms(self):
        key = search_state_key("Does PD-L1 predict response in non-small cell lung cancer?")

        self.assertIn("search:v1", key)
        self.assertIn("intent:question", key)
        self.assertNotIn("pd", key.lower())
        self.assertNotIn("lung", key.lower())
        self.assertNotIn("cancer", key.lower())

    def test_deterministic_variants_include_normalized_biomedical_terms(self):
        variants = deterministic_query_variants(
            "Does PD-L1 predict response in non-small cell lung cancer?",
            strategy="wide",
            max_variants=5,
        )
        joined = " ".join(item.query.lower() for item in variants)

        self.assertGreaterEqual(len(variants), 3)
        self.assertIn("pdl1", joined)
        self.assertIn("nsclc", joined)

    def test_user_critical_terms_precede_broad_domain_bridge_for_acronym_queries(self):
        frame = _domain_search_frame(
            "Explain CAF-associated ECM remodeling and matrix stiffness mechanistically."
        )
        variants = deterministic_query_variants(
            "Explain CAF-associated ECM remodeling and matrix stiffness mechanistically.",
            strategy="medium",
            max_variants=4,
            search_frame=frame,
        )
        labels = [item.label for item in variants]
        joined = " ".join(item.query.lower() for item in variants)

        self.assertIn("important_terms", labels)
        self.assertIn("biomedical_synonyms", labels)
        self.assertLess(labels.index("important_terms"), labels.index("domain_bridge"))
        self.assertIn("cancer associated fibroblast", joined)
        self.assertIn("extracellular matrix", joined)

    def test_domain_bridge_maps_functional_synergy_to_mechanistic_tme_terms(self):
        frame = _domain_search_frame("What is the functional synergy that defines aggressive lung carcinoma?")
        variants = deterministic_query_variants(
            "What is the functional synergy that defines aggressive lung carcinoma?",
            strategy="medium",
            max_variants=4,
            search_frame=frame,
        )
        joined = " ".join(item.query.lower() for item in variants)

        self.assertEqual(frame["frame"], "mechanism_or_pathway")
        self.assertIn("functional synergy", joined)
        self.assertIn("aggressive lung", joined)
        self.assertIn("combination index", frame["avoid_terms"])

    def test_meta_task_words_do_not_dominate_biomedical_search_frame(self):
        query = (
            "For a multi-turn evaluation, give the careful biomedical framing for TME-only "
            "scope control across multi-turn conversation. Use cautious mechanistic language, "
            "not clinical treatment advice."
        )
        frame = _domain_search_frame(query)
        preferred = " ".join(frame["preferred_queries"]).lower()
        variants = deterministic_query_variants(query, search_frame=frame, max_variants=4)
        variant_text = " ".join(item.query.lower() for item in variants)

        self.assertIn("tme", preferred)
        self.assertIn("tumor microenvironment", preferred)
        self.assertNotIn("multi turn", preferred)
        self.assertNotIn("evaluation", preferred)
        self.assertNotIn("scope control", preferred)
        self.assertNotIn("treatment", preferred)
        self.assertNotIn("multi-turn evaluation", variant_text)
        self.assertNotIn("scope control", variant_text)

    def test_meta_task_hit_feedback_is_rejected_when_biomedical_anchor_missing(self):
        query = (
            "For a multi-turn evaluation, give the careful biomedical framing for TME-only "
            "scope control across multi-turn conversation."
        )
        report = _feedback_term_report(
            [
                {
                    "title": "Scope of the problem",
                    "text": "In turn, the scope of the problem presented is substantial.",
                }
            ],
            anchor_queries=[query, "tme tumor microenvironment"],
        )

        self.assertEqual(report["accepted_terms"], [])
        self.assertGreaterEqual(report["rejected_result_count"], 1)

    def test_broad_umbrella_anchor_does_not_bootstrap_incidental_mechanism_feedback(self):
        report = _feedback_term_report(
            [
                {
                    "title": "Growing evidence suggests TREM-1 involvement in oncogenesis through TME inflammation",
                    "text": "Growing evidence suggests TREM-1 involvement through cancer-associated inflammation and the tumor microenvironment.",
                }
            ],
            anchor_queries=["TME tumor microenvironment"],
        )

        accepted = " ".join(report["accepted_terms"]).lower()
        self.assertNotIn("trem", accepted)
        self.assertNotIn("involvement", accepted)

    def test_specific_entity_anchor_filters_hit_missing_entity(self):
        frame = _domain_search_frame("what fungi are described as playing essential roles in tumorigenesis and how it happens")

        nfkb = _hit_anchor_coverage(
            {
                "title": "NF-kB and inflammation in tumorigenesis",
                "text": "NF-kB regulates immune response and inflammation and supports tumorigenesis.",
            },
            frame,
        )
        fungi = _hit_anchor_coverage(
            {
                "title": "Fungi and other microbes in tumorigenesis",
                "text": "Other microbes also play essential roles in tumorigenesis, including fungi.",
            },
            frame,
        )

        self.assertFalse(nfkb["passes"])
        self.assertIn("fungi", nfkb["missing_anchors"])
        self.assertTrue(fungi["passes"])

    def test_current_query_prevents_stale_tme_note_from_forcing_fungal_analogy_frame(self):
        frame = _domain_search_frame(
            "What experimental approaches treat cancer as a fungal infection?",
            notes=[{"note": "Use TME tumor microenvironment growth bridge terms for the prior lung cancer turn."}],
        )

        self.assertEqual(frame["frame"], "cross_domain_or_analogy")
        self.assertIn("cancer", " ".join(frame["preferred_queries"]).lower())

    def test_feedback_terms_filter_caption_noise_and_keep_domain_terms(self):
        terms = _feedback_terms_from_results(
            [
                {
                    "title": "The figure shows that immune cells in TME regulate tumor growth.",
                    "text": "CAF macrophage hypoxia angiogenesis cytokine crosstalk promotes invasion.",
                }
            ],
            limit=8,
        )

        self.assertNotIn("figure", terms)
        self.assertNotIn("show", terms)
        self.assertIn("hypoxia", terms)

    def test_feedback_terms_filter_instrument_noise_inside_partly_relevant_hit(self):
        terms = _feedback_terms_from_results(
            [
                {
                    "title": "Food habits questionnaire with free text question formats",
                    "text": "Four text questions and one test guarantee response collection.",
                }
            ],
            limit=8,
        )

        self.assertNotIn("questionnaire", terms)
        self.assertNotIn("ques", terms)
        self.assertNotIn("test", terms)
        self.assertNotIn("four", terms)

    def test_feedback_terms_filter_dialogue_and_correspondence_metadata(self):
        terms = _feedback_terms_from_results(
            [
                {
                    "title": "Physician 9 noted treatment risk and benefit between patients",
                    "text": "Correspondence Jane dos Santos jlsantos uesc brthi patient talk.",
                }
            ],
            limit=8,
        )

        self.assertNotIn("physician", terms)
        self.assertNotIn("noted", terms)
        self.assertNotIn("correspondence", terms)
        self.assertNotIn("jane", terms)
        self.assertNotIn("jlsanto", terms)

    def test_cross_domain_analogy_rejects_case_treatment_feedback_without_target_domain(self):
        report = _feedback_term_report(
            [
                {
                    "title": "Pulmonary cryptococcosis diagnosed after treatment",
                    "text": "The patient received fluconazole antifungal therapy after diagnosis.",
                }
            ],
            anchor_queries=["cancer therapy inspired by antifungal strategy"],
            limit=8,
        )

        self.assertEqual(report["accepted_terms"], [])
        self.assertGreater(report["rejected_result_count"], 0)

    def test_off_topic_synergy_hit_is_skipped_for_mechanistic_frame(self):
        frame = _domain_search_frame("functional synergy aggressive lung carcinoma")
        self.assertTrue(
            _is_off_topic_hit(
                {
                    "title": "CI value less than 1 defines synergy and antagonism",
                    "text": "Combination index and dose response define drug synergy.",
                },
                frame,
            )
        )
        self.assertFalse(
            _is_off_topic_hit(
                {
                    "title": "TME crosstalk in lung cancer",
                    "text": "Hypoxia and CAF macrophage crosstalk promote NSCLC invasion.",
                },
                frame,
            )
        )

    def test_llm_query_fallback_accepts_bulleted_provider_text(self):
        queries = _fallback_queries_from_text(
            "- PD-L1 checkpoint inhibitor response\n"
            "- programmed death ligand 1 immunotherapy NSCLC\n"
            "Note: broaden with disease synonyms.",
            limit=4,
        )

        self.assertEqual(queries[:2], [
            "PD-L1 checkpoint inhibitor response",
            "programmed death ligand 1 immunotherapy NSCLC",
        ])

    async def test_build_auto_context_runs_multilevel_search_and_uses_feedback_terms(self):
        old_llm_refine = settings.memory.auto_context_llm_refine
        old_k = settings.memory.auto_context_k
        old_variants = settings.memory.auto_context_query_variants
        settings.memory.auto_context_llm_refine = False
        settings.memory.auto_context_k = 5
        settings.memory.auto_context_query_variants = 4
        fake_multilevel = FakeMultilevelSearch()
        try:
            result = await build_auto_context(
                tenant="default",
                session_id="s1",
                message="Does PD-L1 predict response in non-small cell lung cancer?",
                store=FakeStore(),
                selected_context_count=0,
                confidence_min=0.5,
                multilevel_search_fn=fake_multilevel,
            )
        finally:
            settings.memory.auto_context_llm_refine = old_llm_refine
            settings.memory.auto_context_k = old_k
            settings.memory.auto_context_query_variants = old_variants

        snippets = result["snippets"]
        plan = result["plan"]
        self.assertGreaterEqual(len(snippets), 3)
        self.assertTrue(all(item.get("auto_context") for item in snippets))
        self.assertEqual(snippets[0]["source"], "auto_context")
        self.assertEqual(plan["levels"], ["title", "paper", "sentence"])
        self.assertEqual([report["level"] for report in plan["level_reports"]], ["title", "paper", "sentence"])
        self.assertIn("nivolumab", " ".join(fake_multilevel.calls[-1]["query"].lower() for _ in [0]))
        self.assertIn("search:v1", plan["state_key"])
        self.assertIn("search:v1", plan["action_key"])
        self.assertEqual(plan["result_count"], len(snippets))
        self.assertTrue(plan["note"])
        self.assertIn("evidence_assembly", plan)
        self.assertEqual(plan["evidence_assembly"]["information_need"], "question")
        self.assertGreaterEqual(len(plan["candidate_frames"]), 2)
        self.assertIn("frame_result_counts", plan["evidence_assembly"])
        self.assertGreaterEqual(len(plan["retrieval_records"]), 1)
        first_record = plan["retrieval_records"][0]
        self.assertEqual(first_record["rank"], 1)
        self.assertEqual(first_record["bm25_score"], 5.0)
        self.assertEqual(first_record["source_sentence_id"], "title")
        self.assertIn("mechanism_tags", first_record)
        self.assertEqual(snippets[0]["bm25_score"], 5.0)
        self.assertEqual(snippets[0]["retrieval_rank"], 1)

    async def test_build_auto_context_keeps_sentence_search_fn_compatibility(self):
        old_llm_refine = settings.memory.auto_context_llm_refine
        old_k = settings.memory.auto_context_k
        old_variants = settings.memory.auto_context_query_variants
        settings.memory.auto_context_llm_refine = False
        settings.memory.auto_context_k = 2
        settings.memory.auto_context_query_variants = 1
        try:
            result = await build_auto_context(
                tenant="default",
                session_id="s1",
                message="Does PD-L1 predict response in non-small cell lung cancer?",
                store=FakeStore(),
                selected_context_count=0,
                confidence_min=0.5,
                search_fn=fake_search,
            )
        finally:
            settings.memory.auto_context_llm_refine = old_llm_refine
            settings.memory.auto_context_k = old_k
            settings.memory.auto_context_query_variants = old_variants

        self.assertGreaterEqual(len(result["snippets"]), 1)

    async def test_followup_query_reuses_prior_supported_search_frame(self):
        plan = await plan_auto_context(
            message="Start by explaining the conceptual analogy and develop the latest candidate frameworks you suggested.",
            selected_context_count=0,
            notes=[
                {
                    "note": "The prior search found evidence for antifungal-inspired cancer therapy.",
                    "search_plan": {
                        "variants": [
                            {
                                "label": "llm_refined",
                                "query": "cancer therapy antifungal strategy immune evidence",
                                "source": "llm",
                            }
                        ]
                    },
                }
            ],
            action_value_hints=[],
            max_variants=4,
            allow_llm_refine=False,
        )

        joined = " ".join(item.query.lower() for item in plan.variants)
        self.assertIn("antifungal", joined)
        self.assertEqual(plan.variants[0].label, "prior_frame")
        self.assertTrue(any(item.label == "prior_frame" for item in plan.variants))

    async def test_search_more_followup_reuses_prior_biomedical_frame(self):
        plan = await plan_auto_context(
            message="search more on all your available data sources",
            selected_context_count=0,
            notes=[
                {
                    "note": "The prior search was about fungi and tumorigenesis mechanisms.",
                    "search_plan": {
                        "variants": [
                            {
                                "label": "llm_refined",
                                "query": "fungi tumorigenesis mechanism mycobiome cancer",
                                "source": "llm",
                            }
                        ]
                    },
                }
            ],
            action_value_hints=[],
            max_variants=4,
            allow_llm_refine=False,
        )

        joined = " ".join(item.query.lower() for item in plan.variants)
        self.assertEqual(plan.variants[0].label, "prior_frame")
        self.assertIn("fungi", joined)
        self.assertIn("tumorigenesis", joined)
        self.assertNotIn("supplementary material", joined)

    async def test_unrelated_new_question_does_not_reuse_prior_frame(self):
        plan = await plan_auto_context(
            message="what are the body pH and metabolic impairment roles in cancer development?",
            selected_context_count=0,
            notes=[
                {
                    "note": "The prior search was about fungi and tumorigenesis mechanisms.",
                    "search_plan": {
                        "variants": [
                            {
                                "label": "llm_refined",
                                "query": "fungi tumorigenesis mechanism mycobiome cancer",
                                "source": "llm",
                            }
                        ]
                    },
                }
            ],
            action_value_hints=[],
            max_variants=4,
            allow_llm_refine=False,
        )

        joined = " ".join(item.query.lower() for item in plan.variants)
        self.assertFalse(any(item.label == "prior_frame" for item in plan.variants))
        self.assertIn("metabolic", joined)
        self.assertNotIn("fungi", joined)

    async def test_build_auto_context_search_more_uses_conversation_frame_when_notes_missing(self):
        old_llm_refine = settings.memory.auto_context_llm_refine
        old_k = settings.memory.auto_context_k
        old_variants = settings.memory.auto_context_query_variants
        settings.memory.auto_context_llm_refine = False
        settings.memory.auto_context_k = 4
        settings.memory.auto_context_query_variants = 4
        fake_search = FakeFungiMultilevelSearch()
        try:
            result = await build_auto_context(
                tenant="default",
                session_id="s-frame",
                message="search more on all your available data sources",
                store=FakeFrameStore(),
                selected_context_count=0,
                confidence_min=0.5,
                multilevel_search_fn=fake_search,
            )
        finally:
            settings.memory.auto_context_llm_refine = old_llm_refine
            settings.memory.auto_context_k = old_k
            settings.memory.auto_context_query_variants = old_variants

        queries = " ".join(call["query"].lower() for call in fake_search.calls)
        self.assertIn("fungi", queries)
        self.assertIn("tumorigenesis", queries)
        self.assertNotIn("supplementary", queries)
        self.assertTrue(result["snippets"])

    async def test_style_rewrite_followup_reuses_prior_biomedical_frame(self):
        plan = await plan_auto_context(
            message="Give me a one-paragraph version for a novice user, but keep the biomedical direction correct.",
            selected_context_count=0,
            notes=[
                {
                    "note": "The prior query was about CAF-associated ECM remodeling and matrix stiffness.",
                    "search_plan": {
                        "variants": [
                            {
                                "label": "biomedical_synonyms",
                                "query": "caf cancer associated fibroblast ecm extracellular matrix stiffness remodeling",
                                "source": "deterministic",
                            }
                        ]
                    },
                }
            ],
            action_value_hints=[],
            max_variants=4,
            allow_llm_refine=False,
        )

        self.assertEqual(plan.variants[0].label, "prior_frame")
        self.assertIn("cancer associated fibroblast", plan.variants[0].query.lower())

    async def test_concise_caveat_followup_reuses_prior_biomedical_frame(self):
        plan = await plan_auto_context(
            message="If I ask for a concise answer, what essential caveat must not disappear?",
            selected_context_count=0,
            notes=[
                {
                    "note": "The prior query was about CAF-associated ECM remodeling and stiffness.",
                    "search_plan": {
                        "variants": [
                            {
                                "label": "llm_refined",
                                "query": "CAF ECM remodeling stiffness lung cancer progression",
                                "source": "llm",
                            }
                        ]
                    },
                }
            ],
            action_value_hints=[],
            max_variants=4,
            allow_llm_refine=False,
        )

        self.assertEqual(plan.variants[0].label, "prior_frame")
        joined = " ".join(item.query.lower() for item in plan.variants)
        self.assertIn("caf", joined)
        self.assertIn("ecm", joined)
        self.assertNotIn("ask concise caveat", joined)

    async def test_followup_evidence_puzzle_uses_prior_frame_nodes(self):
        plan = await plan_auto_context(
            message="Use what is actually supported and separate direct evidence from hypotheses.",
            selected_context_count=0,
            notes=[
                {
                    "note": "The prior query was about pH, food habits, and tissue growth.",
                    "search_plan": {
                        "variants": [
                            {
                                "label": "llm_refined",
                                "query": "metabolic pathway body pH food habits tissue growth",
                                "source": "llm",
                            }
                        ]
                    },
                }
            ],
            action_value_hints=[],
            max_variants=4,
            allow_llm_refine=False,
        )

        assembly = _evidence_assembly(
            message="Use what is actually supported and separate direct evidence from hypotheses.",
            plan=plan,
            snippets=[],
            level_reports=[],
        )

        nodes = " ".join(assembly["evidence_puzzle"]["candidate_nodes"]).lower()
        self.assertIn("food", nodes)
        self.assertIn("ph", nodes)
        self.assertTrue(assembly["clarification_recommended"])

    async def test_phrase_check_followup_reuses_prior_evidence_frame(self):
        plan = await plan_auto_context(
            message='Can the chatbot phrase the answer as: "Collagen crosslinking and matrix stiffness have no plausible connection to cancer cell motility or treatment response."?',
            selected_context_count=0,
            notes=[
                {
                    "note": "The prior query was about CAF-associated ECM remodeling and stiffness.",
                    "search_plan": {
                        "variants": [
                            {
                                "label": "llm_refined",
                                "query": "CAF ECM remodeling stiffness lung cancer progression",
                                "strategy": "medium",
                            }
                        ]
                    },
                }
            ],
            action_value_hints=[],
            max_variants=4,
            allow_llm_refine=False,
        )

        self.assertEqual(plan.variants[0].label, "prior_frame")
        self.assertIn("ecm remodeling", plan.variants[0].query.lower())

        assembly = _evidence_assembly(
            message='Can the chatbot phrase the answer as: "Collagen crosslinking and matrix stiffness have no plausible connection to cancer cell motility or treatment response."?',
            plan=plan,
            snippets=[],
            level_reports=[],
        )
        self.assertFalse(assembly["clarification_recommended"])
        self.assertIn("judge the proposed wording first", assembly["prompt_context"].lower())

    async def test_rewrite_and_reward_followups_do_not_hold_for_clarification(self):
        for message in (
            "Give me a one-paragraph version for a novice user, but keep the biomedical direction correct.",
            "Now answer again in two sentences after my correction. What should the reward model check?",
        ):
            plan = await plan_auto_context(
                message=message,
                selected_context_count=0,
                notes=[
                    {
                        "note": "Prior supported frame.",
                        "search_plan": {
                            "variants": [
                                {
                                    "label": "llm_refined",
                                    "query": "CAF ECM remodeling stiffness lung cancer progression",
                                }
                            ]
                        },
                    }
                ],
                action_value_hints=[],
                max_variants=4,
                allow_llm_refine=False,
            )
            assembly = _evidence_assembly(
                message=message,
                plan=plan,
                snippets=[],
                level_reports=[],
            )

            self.assertFalse(assembly["clarification_recommended"])

    async def test_evidence_assembly_prompt_warns_absence_is_not_exclusion_evidence(self):
        plan = await plan_auto_context(
            message='Can I say this relation has no plausible connection?',
            selected_context_count=0,
            notes=[],
            action_value_hints=[],
            max_variants=2,
            allow_llm_refine=False,
        )
        assembly = _evidence_assembly(
            message='Can I say this relation has no plausible connection?',
            plan=plan,
            snippets=[],
            level_reports=[],
        )

        self.assertIn("absence of a relation", assembly["prompt_context"].lower())
        self.assertIn("no plausible connection", assembly["prompt_context"].lower())

    async def test_unanchored_early_hit_feedback_does_not_poison_later_queries(self):
        old_llm_refine = settings.memory.auto_context_llm_refine
        old_k = settings.memory.auto_context_k
        old_variants = settings.memory.auto_context_query_variants
        settings.memory.auto_context_llm_refine = False
        settings.memory.auto_context_k = 4
        settings.memory.auto_context_query_variants = 1
        noisy_search = FakeNoisyMultilevelSearch()
        try:
            result = await build_auto_context(
                tenant="default",
                session_id="s-noisy",
                message="Develop a metabolic pathway relating body pH, food habits, and tissue growth.",
                store=FakeStore(),
                selected_context_count=0,
                confidence_min=0.5,
                multilevel_search_fn=noisy_search,
            )
        finally:
            settings.memory.auto_context_llm_refine = old_llm_refine
            settings.memory.auto_context_k = old_k
            settings.memory.auto_context_query_variants = old_variants

        later_queries = " ".join(call["query"].lower() for call in noisy_search.calls if call["level"] != "title")
        first_report = result["plan"]["level_reports"][0]
        assembly = result["plan"]["evidence_assembly"]
        self.assertNotIn("questionnaire", later_queries)
        self.assertNotIn("radio", later_queries)
        self.assertGreater(first_report["rejected_feedback_result_count"], 0)
        self.assertGreaterEqual(assembly["refinement_quality"]["rejected_feedback_result_count"], 1)

    async def test_high_ambiguity_evidence_puzzle_requests_textual_clarification(self):
        old_llm_refine = settings.memory.auto_context_llm_refine
        old_k = settings.memory.auto_context_k
        old_variants = settings.memory.auto_context_query_variants
        settings.memory.auto_context_llm_refine = False
        settings.memory.auto_context_k = 4
        settings.memory.auto_context_query_variants = 4
        try:
            result = await build_auto_context(
                tenant="default",
                session_id="s-ambiguous",
                message="Develop a pathway relating food habits, pH, and something else that promotes tissue growth.",
                store=FakeStore(),
                selected_context_count=0,
                confidence_min=0.5,
                multilevel_search_fn=FakeNoisyMultilevelSearch(),
            )
        finally:
            settings.memory.auto_context_llm_refine = old_llm_refine
            settings.memory.auto_context_k = old_k
            settings.memory.auto_context_query_variants = old_variants

        assembly = result["plan"]["evidence_assembly"]
        self.assertTrue(assembly["clarification_recommended"])
        self.assertIn("opening paragraph must end", assembly["prompt_context"].lower())
        self.assertIn("candidate_frames", assembly)
        self.assertIn("evidence_puzzle", assembly)
        self.assertIn("absent example candidates", assembly["prompt_context"].lower())
        self.assertIn("named candidate", assembly["prompt_context"].lower())
        self.assertIn("only supported evidence", assembly["prompt_context"].lower())

    async def test_entity_mechanism_query_uses_generic_task_bridge(self):
        plan = await plan_auto_context(
            message="what fungi are described as playing essential roles in tumorigenesis and how it happens",
            selected_context_count=0,
            notes=[],
            action_value_hints=[],
            max_variants=4,
            allow_llm_refine=False,
        )

        queries = " ".join(item.query.lower() for item in plan.variants)
        self.assertEqual(plan.search_frame["frame"], "mechanism_or_pathway")
        self.assertIn("fungi", queries)
        self.assertIn("tumorigenesis", queries)
        self.assertIn("mechanism", queries)

    async def test_entity_mechanism_search_rejects_hits_missing_entity_anchor(self):
        old_llm_refine = settings.memory.auto_context_llm_refine
        old_k = settings.memory.auto_context_k
        old_variants = settings.memory.auto_context_query_variants
        settings.memory.auto_context_llm_refine = False
        settings.memory.auto_context_k = 4
        settings.memory.auto_context_query_variants = 4
        fake_search = FakeFungiMultilevelSearch()
        try:
            result = await build_auto_context(
                tenant="default",
                session_id="s-fungi",
                message="what fungi are described as playing essential roles in tumorigenesis and how it happens",
                store=FakeStore(),
                selected_context_count=0,
                confidence_min=0.5,
                multilevel_search_fn=fake_search,
            )
        finally:
            settings.memory.auto_context_llm_refine = old_llm_refine
            settings.memory.auto_context_k = old_k
            settings.memory.auto_context_query_variants = old_variants

        snippets = result["snippets"]
        plan = result["plan"]
        self.assertTrue(snippets)
        self.assertTrue(all("fungi" in (item.get("text") or item.get("title") or "").lower() for item in snippets))
        self.assertGreaterEqual(plan["level_reports"][0]["anchor_mismatch_result_count"], 1)
        self.assertNotIn("nf-kb", " ".join(plan["level_reports"][0]["feedback_terms_after"]).lower())

    async def test_bibliography_hit_feedback_is_rejected(self):
        report = _feedback_term_report(
            [
                {
                    "title": "Candida albicans tumorigenesis",
                    "text": "Allemailem K. Alnuqaydan A. Almatroudi A. Alrumaihi F. Khalilullah H. Khan A. Safety and Therapeutic Efficacy Pharmaceutics 2021 13 677 10.3390/pharmaceutics13050677.",
                }
            ],
            anchor_queries=["Candida albicans tumorigenesis"],
        )

        accepted = " ".join(report["accepted_terms"]).lower()
        self.assertGreaterEqual(report["rejected_result_count"], 1)
        self.assertNotIn("alnuqaydan", accepted)
        self.assertNotIn("almatroudi", accepted)
        self.assertNotIn("khan", accepted)

    async def test_evidence_assembly_quality_drops_for_meta_task_only_hits(self):
        query = (
            "For a multi-turn evaluation, give the careful biomedical framing for TME-only "
            "scope control across multi-turn conversation."
        )
        plan = await plan_auto_context(
            message=query,
            selected_context_count=0,
            notes=[],
            action_value_hints=[],
            max_variants=4,
            allow_llm_refine=False,
        )
        assembly = _evidence_assembly(
            message=query,
            plan=plan,
            snippets=[
                {
                    "paper_id": "paper-scope",
                    "sent_id": "s1",
                    "title": "Scope of the problem",
                    "text": "In turn, the scope of the problem presented is substantial.",
                }
            ],
            level_reports=[
                {"level": "title", "result_count": 1, "feedback_terms_added": []},
                {"level": "paper", "result_count": 1, "feedback_terms_added": []},
                {"level": "sentence", "result_count": 1, "feedback_terms_added": []},
            ],
        )

        puzzle = assembly["evidence_puzzle"]
        self.assertIn("tme", puzzle["candidate_nodes"])
        self.assertEqual(puzzle["edge_support_status"], "missing")
        self.assertLessEqual(assembly["assembly_quality"], 0.55)

    async def test_umbrella_only_evidence_puzzle_requests_clarification(self):
        query = "Give careful biomedical framing for TME-only scope control."
        plan = await plan_auto_context(
            message=query,
            selected_context_count=0,
            notes=[],
            action_value_hints=[],
            max_variants=4,
            allow_llm_refine=False,
        )
        assembly = _evidence_assembly(
            message=query,
            plan=plan,
            snippets=[
                {
                    "paper_id": "paper-tme",
                    "sent_id": "s1",
                    "title": "TME inflammation",
                    "text": "Growing evidence suggests TREM-1 involvement through cancer-associated inflammation and the tumor microenvironment.",
                }
            ],
            level_reports=[
                {"level": "title", "result_count": 1, "feedback_terms_added": []},
            ],
        )

        self.assertTrue(assembly["clarification_recommended"])
        self.assertIn("incidental named mechanism", assembly["prompt_context"].lower())

    async def test_mechanism_question_with_generic_relation_is_partial_not_supported(self):
        query = "what fungi are described as playing essential roles in tumorigenesis and how it happens"
        plan = await plan_auto_context(
            message=query,
            selected_context_count=0,
            notes=[],
            action_value_hints=[],
            max_variants=4,
            allow_llm_refine=False,
        )
        assembly = _evidence_assembly(
            message=query,
            plan=plan,
            snippets=[
                {
                    "paper_id": "paper-fungi",
                    "sent_id": "s1",
                    "title": "Other microbes in tumorigenesis",
                    "text": "Other microbes also play essential roles in tumorigenesis, including fungi, viruses, and bacteriophages.",
                    "relation": "play roles in",
                }
            ],
            level_reports=[
                {"level": "sentence", "result_count": 1, "feedback_terms_added": ["fungi", "tumorigenesis"]},
            ],
        )

        self.assertEqual(assembly["evidence_puzzle"]["edge_support_status"], "partial")

    async def test_general_to_particular_process_words_do_not_become_missing_nodes(self):
        query = "I am interested in the general-to-particular process of how cancer starts and develops"
        plan = await plan_auto_context(
            message=query,
            selected_context_count=0,
            notes=[],
            action_value_hints=[],
            max_variants=4,
            allow_llm_refine=False,
        )
        assembly = _evidence_assembly(
            message=query,
            plan=plan,
            snippets=[
                {
                    "paper_id": "paper-cancer",
                    "sent_id": "s1",
                    "title": "Cancer initiation and progression",
                    "text": "Cancer initiation and progression involve genomic alterations, clonal expansion, immune evasion, and tissue microenvironment changes.",
                }
            ],
            level_reports=[
                {"level": "sentence", "result_count": 1, "feedback_terms_added": ["cancer", "progression"]},
            ],
        )

        puzzle = assembly["evidence_puzzle"]
        nodes = set(puzzle["candidate_nodes"])
        self.assertIn("cancer", nodes)
        self.assertNotIn("interested", nodes)
        self.assertNotIn("general to particular", nodes)
        self.assertNotIn("general-to-particular", nodes)
        self.assertNotIn("particular", nodes)


if __name__ == "__main__":
    unittest.main()
