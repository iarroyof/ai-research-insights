# Lung Factuality Generated Corpus v1 — Verification Report

Generated corpus: `lung_factuality_generated_corpus_v1`

## Size

- Scenarios: 8
- Generated conversations: 120
- Turns: 840
- Variants per scenario: 15

## Files

- `data/conversations/generated/index.yaml`
- `data/conversations/generated/corpus_all_turns.jsonl`
- `data/conversations/generated/<scenario_id>/<conversation_id>.jsonl`
- `data/scenarios/generated_scenarios.yaml`
- `data/evidence/generated_gold_claims.yaml`
- `data/evidence/generated_mechanism_graphs.yaml`
- `data/perturbations/generated_user_false_premise_bank.yaml`
- `data/perturbations/generated_assistant_wrong_answer_bank.yaml`
- `data/verification/verification_summary.json`

## Verification pass 1 — source-grounded biomedical fact matrix

Manually checked true and false directions against a source-grounded matrix. The corpus intentionally includes false user premises and wrong-answer fixtures only as traps/negative examples. The negative examples should not be treated as true biomedical facts.

Checked topics:

### lc_tme_caf_hgf_met_001

True anchor: CAF-derived or stromal HGF can activate MET/c-MET signaling and contribute to NSCLC/lung cancer tumor progression, invasion, EMT-like programs, angiogenesis, immune evasion, or therapy resistance depending on context.

Unacceptable variants:
- HGF decreases MET signaling and therefore blocks EMT.
- CAF-derived HGF directly causes EMT without receptor involvement, so MET/c-MET is unnecessary.
- MET mainly suppresses HGF as the dominant lung-cancer TME mechanism.

### lc_tme_tam_m2_cd8_001

True anchor: M2-like tumor-associated macrophages in lung cancer are commonly associated with immunosuppression, reduced antigen presentation, impaired effector T-cell responses, tumor progression, angiogenesis, metastasis, and therapy resistance.

Unacceptable variants:
- M2-like TAMs mainly activate cytotoxic CD8+ T cells and improve anti-tumor immunity in NSCLC.
- M2-like TAMs are unrelated to immune evasion or therapy resistance in lung cancer.
- High M2-like TAM activity should be treated as a simple marker of stronger anti-tumor immunity.

### lc_tme_hypoxia_hif_pd_l1_001

True anchor: Hypoxia and HIF signaling in tumors can contribute to immune escape, weakened cytotoxic T-cell function, recruitment of regulatory T cells, PD-L1-related mechanisms, angiogenesis, and aggressive lung cancer biology.

Unacceptable variants:
- Hypoxia generally improves cytotoxic T-cell function and reduces immune escape in lung cancer.
- HIF-1α has no plausible connection to PD-L1-related immune escape.
- Tumor hypoxia blocks angiogenesis rather than promoting VEGF/angiogenic programs.

### lc_tme_mdsc_treg_cd8_001

True anchor: MDSCs and Tregs contribute to an immunosuppressive tumor microenvironment by limiting effector T-cell activity and supporting immune evasion; their crosstalk is relevant to checkpoint-resistance discussions.

Unacceptable variants:
- MDSCs primarily enhance antigen presentation and strengthen CD8+ T-cell killing in the tumor microenvironment.
- Tregs in lung cancer mainly activate effector T cells rather than suppressing them.
- MDSC-Treg crosstalk is irrelevant to immune checkpoint resistance.

### lc_tme_caf_ecm_stiffness_001

True anchor: CAF-associated extracellular matrix remodeling and stiffness can support tumor progression, invasion, drug delivery barriers, immune-cell exclusion, and therapy resistance; lung-cancer claims should preserve context and avoid universal statements.

Unacceptable variants:
- ECM stiffness generally improves immune killer-cell infiltration and drug delivery in tumors.
- CAF-associated ECM remodeling is always tumor-suppressive in lung cancer.
- Collagen crosslinking and matrix stiffness have no plausible connection to cancer cell motility or treatment response.

### lc_tme_caf_heterogeneity_001

True anchor: CAFs are heterogeneous in lung cancer and can have diverse, context-dependent roles in tumorigenesis, progression, angiogenesis, invasion, metastasis, therapy resistance, immune suppression, and treatment response.

Unacceptable variants:
- All CAFs have the same function in all lung tumors.
- CAF evidence from pancreatic or breast cancer can be presented as proven NSCLC evidence without caveats.
- CAFs are only tumor-suppressive and do not contribute to immune suppression or therapy resistance.

## Verification pass 2 — polarity and false-claim audit

Automated check: no unacceptable false claim appears inside a `must_mention` list. False claims appear only in user trap prompts, `must_not_claim`, expected rejection text, or wrong-answer fixtures.

Status: PASSED.

## Verification pass 3 — schema and observability audit

Automated check: every turn has fields needed by the trace/evaluation pipeline: conversation id, scenario id, variant index, turn, user text, expected behavior, target gold claims, trap ids, must-mention constraints, must-not-claim constraints, scope, and tags.

Status: PASSED.

## Intended use

This corpus is designed for agent-observable evaluation. It should help the agent inspect not only final scores but the process: user trap, expected behavior, assistant answer, extracted claims, gold match, relation polarity, scope adherence, penalties, failure owner, recommendation, and regression-plan creation.

## Important caveat

This is an evaluation corpus, not clinical guidance. It is designed to test factuality and mechanistic consistency of generated answers about lung-cancer TME biology.
