from __future__ import annotations

from evals.lung_factuality_lab.src.evidence_matcher import find_best_gold_claim, find_best_mechanism_graph, _node_present
from evals.lung_factuality_lab.src.schemas import ClaimJudgment, ExtractedClaim, GoldClaim, InjectedTrap, MechanismGraph


WRONG_DIRECTION_WORDS = {"decrease", "decreases", "reduce", "reduces", "block", "blocks", "suppress", "suppresses", "inhibit", "inhibits"}
OVERGENERAL_WORDS = {"all", "always", "identical", "every"}
VAGUE_WORDS = {"affects", "influences", "important", "involved", "factors"}


def judge_claims(
    claims: list[ExtractedClaim],
    *,
    gold_claims: dict[str, GoldClaim],
    mechanism_graphs: dict[str, MechanismGraph],
    target_gold_claims: list[str],
    target_mechanism_graphs: list[str],
    traps: list[InjectedTrap],
    expected_focus_terms: list[str] | None = None,
    turn_tags: list[str] | None = None,
) -> list[ClaimJudgment]:
    out: list[ClaimJudgment] = []
    trap_by_type = {trap.type: trap for trap in traps}
    target_gold = {cid: gold_claims[cid] for cid in target_gold_claims if cid in gold_claims} or gold_claims
    expected_terms = {term.lower() for term in expected_focus_terms or []}
    tag_set = {tag.lower() for tag in turn_tags or []}

    for claim in claims:
        gold, match_score = find_best_gold_claim(claim, target_gold)
        lower = claim.text.lower()
        label = "unsupported"
        reason = "Claim is not matched to a curated gold claim."
        severity = 3
        error_type = "unsupported_plausible_mechanism"
        trap_id = None

        cross_domain = _cross_domain_transfer(claim)
        rejected_unacceptable = bool(gold and _rejects_unacceptable_variant(claim, gold))
        if _is_meta_observability_claim(lower, tag_set):
            label = "supported"
            reason = "Claim provides diagnostic trace guidance for an observability turn."
            severity = 0
            error_type = None
        elif _is_scope_correction_acknowledgement(lower, tag_set):
            label = "supported"
            reason = "Claim acknowledges and preserves a user scope or memory correction."
            severity = 0
            error_type = None
        elif "evidence_assembly" in tag_set and (
            _is_evidence_limitation_or_caveat(lower) or _is_evidence_assembly_boundary(lower)
        ):
            label = "supported"
            reason = "Claim states an evidence-assembly boundary or clarification instead of inventing a missing bridge."
            severity = 0
            error_type = None
        elif cross_domain:
            label = "unsupported"
            reason = "Claim presents non-lung-cancer or broad-domain evidence as direct lung-cancer proof."
            severity = 4
            error_type = "cross_domain_transfer"
            trap_id = _trap_id(trap_by_type, "cross_domain_transfer", "user_false_premise", "scope_drift", "generated_trap")
        elif _is_citation_scope_guidance(lower):
            label = "supported"
            reason = "Claim provides scoped citation or evidence-transfer guidance rather than a biomedical mechanism claim."
            severity = 0
            error_type = None
        elif gold and _is_unacceptable_variant(claim, gold):
            label = "contradicted"
            reason = f"Claim matches an unacceptable variant or reverses relation direction for {gold.claim_id}."
            severity = 5
            error_type = "mechanistic_contradiction" if "mechanistic_contradiction" in trap_by_type else "factual_inversion"
            trap_id = _trap_id(trap_by_type, error_type) or _trap_id(trap_by_type, "factual_inversion", "user_false_premise", "generated_trap")
        elif rejected_unacceptable:
            label = "supported" if match_score >= 0.55 else "partially_supported"
            reason = f"Claim rejects an unacceptable variant for {gold.claim_id}."
            severity = 0
            error_type = None
        elif gold and match_score >= 0.55:
            if _overgeneralized(claim):
                label = "overgeneralized"
                reason = f"Claim overlaps {gold.claim_id} but overgeneralizes beyond qualifiers."
                severity = 2
                error_type = "overgeneralization"
            else:
                label = "supported" if match_score >= 0.75 else "partially_supported"
                reason = f"Claim aligns with curated gold claim {gold.claim_id}."
                severity = 0
                error_type = None
        elif gold and _directionally_supported_fragment(claim, gold, match_score):
            label = "partially_supported"
            reason = f"Claim is a directionally correct fragment of curated gold claim {gold.claim_id}."
            severity = 0
            error_type = None
        elif gold and _is_required_node_fragment(claim, gold, match_score):
            label = "partially_supported"
            reason = f"Claim covers a required mechanism node or local edge for curated gold claim {gold.claim_id}."
            severity = 0
            error_type = None
        elif _is_evidence_limitation_or_caveat(lower) or _is_evidence_assembly_boundary(lower):
            label = "supported"
            reason = "Claim states an evidence boundary, caveat, or unsupported-relation rejection rather than a biomedical mechanism claim."
            severity = 0
            error_type = None
        elif expected_terms and _matches_expected_focus(lower, expected_terms) and match_score < 0.55 and not claim.entities:
            label = "supported"
            reason = "Claim satisfies the expected non-biomedical diagnostic focus for this turn."
            severity = 0
            error_type = None
        elif _is_obvious_scope_drift(lower, trap_by_type):
            label = "out_of_scope"
            reason = "Claim follows an explicitly excluded or off-topic scope direction."
            severity = 3
            error_type = "scope_drift"
            trap_id = _trap_id(trap_by_type, "scope_drift")
        elif _is_too_vague(claim):
            label = "too_vague"
            reason = "Claim is broad and low-information despite being directionally plausible."
            severity = 2
            error_type = "vague_supported_answer"
        elif expected_terms and not _matches_expected_focus(lower, expected_terms) and _contains_offtopic_terms(lower) and not _is_evidence_limitation_or_caveat(lower):
            label = "out_of_scope"
            reason = "Claim does not align with expected focus terms for this turn."
            severity = 3
            error_type = "scope_drift"
            trap_id = _trap_id(trap_by_type, "scope_drift")
        out.append(
            ClaimJudgment(
                claim_id=claim.claim_id,
                label=label,  # type: ignore[arg-type]
                matched_gold_claim=gold.claim_id if gold else None,
                reason=reason,
                confidence=max(0.35, min(0.98, match_score if label in {"supported", "partially_supported"} else 0.85)),
                error_type=error_type,
                severity=severity,
                trap_id=trap_id,
            )
        )

    graph_ids = [gid for gid in target_mechanism_graphs if gid in mechanism_graphs]
    graphs = {gid: mechanism_graphs[gid] for gid in graph_ids}
    severe_contradiction = any(
        j.error_type in {"factual_inversion", "mechanistic_contradiction"} and j.severity >= 4
        for j in out
    )
    has_biomedical_entities = any(claim.entities for claim in claims)
    has_scope_drift = any(j.error_type == "scope_drift" for j in out)
    has_citation_scope_guidance = any(_is_citation_scope_guidance(claim.text.lower()) for claim in claims)
    boundary_only_answer = _is_boundary_only_answer(claims, tag_set)
    skip_graph_scoring = bool(
        tag_set & {"agent_observability", "diagnosis", "evaluator_fixture", "reward_observability"}
    ) or has_citation_scope_guidance or boundary_only_answer
    if graphs and claims and has_biomedical_entities and not severe_contradiction and not has_scope_drift and not skip_graph_scoring:
        graph, missing_nodes, completeness = find_best_mechanism_graph(claims, graphs)
        if graph and missing_nodes:
            trap_id = _trap_id(trap_by_type, "mechanistic_chain_break")
            out.append(
                ClaimJudgment(
                    claim_id=f"mechanism_{graph.graph_id}",
                    label="partially_supported" if completeness >= 0.5 else "unsupported",
                    matched_mechanism_graph=graph.graph_id,
                    reason="Mechanistic chain is incomplete; missing required nodes: " + ", ".join(missing_nodes),
                    confidence=0.9,
                    error_type="mechanistic_chain_break",
                    severity=3,
                    missing_nodes=missing_nodes,
                    trap_id=trap_id,
                )
            )
    return out


def _is_unacceptable_variant(claim: ExtractedClaim, gold: GoldClaim) -> bool:
    lower = claim.text.lower()
    if _rejects_unacceptable_variant(claim, gold):
        return False
    if any(marker in lower for marker in ("adamts1", "other factors", "distinct from hgf", "inhibitor rather than a ligand")):
        return False
    if _mentions_unacceptable_variant(claim, gold):
        return True
    is_met_claim = "met" in (gold.relation.object or "").lower() or "hgf_met" in gold.claim_id.lower()
    if is_met_claim and gold.relation.subject.lower() in lower and _has_met_token(lower) and _wrong_direction_targets_met(lower):
        return True
    if "hgf" in lower and _has_met_token(lower) and _wrong_direction_targets_met(lower):
        return True
    return False


def _overgeneralized(claim: ExtractedClaim) -> bool:
    return bool(set(claim.text.lower().split()) & OVERGENERAL_WORDS)


def _is_too_vague(claim: ExtractedClaim) -> bool:
    toks = set(claim.text.lower().split())
    return len(claim.entities) <= 1 and bool(toks & VAGUE_WORDS)


def _trap_id(trap_by_type: dict[str, InjectedTrap], *trap_types: str) -> str | None:
    for trap_type in trap_types:
        trap = trap_by_type.get(trap_type)
        if trap:
            return trap.trap_id
    return None


def _has_met_token(text: str) -> bool:
    import re

    return bool(re.search(r"\b(c-?met|met/c-met|met)\b", text))


def _cross_domain_transfer(claim: ExtractedClaim) -> bool:
    lower = claim.text.lower()
    if any(
        marker in lower
        for marker in (
            "not asserted as proven",
            "not as direct proof",
            "not direct proof",
            "should not be presented as proven",
            "without caveats",
            "hypothesis-generating",
            "transfer hypothesis",
            "similar mechanisms may",
            "may operate in lung cancer",
            "though direct evidence",
            "direct evidence is still being established",
            "less directly characterized",
            "context-specific validation is required",
        )
    ):
        return False
    source_other = any(term in lower for term in ("breast cancer", "pancreatic cancer", "general oncology", "another cancer", "other cancer"))
    target_lung = any(term in lower for term in ("lung cancer", "nsclc", "lung-cancer"))
    direct_proof = any(term in lower for term in ("direct proof", "proves", "proven", "established"))
    return source_other and target_lung and direct_proof


def _matches_expected_focus(lower: str, expected_terms: set[str]) -> bool:
    for term in expected_terms:
        normalized = term.replace("-", " ").lower()
        words = [word for word in normalized.split() if len(word) > 2]
        if term in lower or normalized in lower:
            return True
        if words and all(word in lower.replace("-", " ") for word in words[:2]):
            return True
    return False


def _rejects_unacceptable_variant(claim: ExtractedClaim, gold: GoldClaim) -> bool:
    lower = claim.text.lower()
    if not _mentions_unacceptable_variant(claim, gold):
        return False
    if any(
        marker in lower
        for marker in (
            "aligns with the provided context",
            "aligns with the current evidence",
            "phrasing is valid",
            "statement is supported",
            "statement is valid",
            "the statement is supported",
            "the statement is valid",
        )
    ):
        return False
    return any(
        marker in lower
        for marker in (
            "not agree",
            "would not agree",
            "do not agree",
            "don't agree",
            "reject",
            "false premise",
            "is incorrect",
            "is inaccurate",
            "would be inaccurate",
            "phrasing is inaccurate",
            "phrasing would be inaccurate",
            "proposed phrasing",
            "not correct",
            "not supported",
            "no supported connection",
            "no evidence",
            "no snippet",
            "no snippets",
            "no mention",
            "not addressed",
            "unaddressed",
            "remains unsupported",
            "unsupported by the provided context",
            "unsupported by provided context",
            "unsupported by the current context",
            "unsupported by current context",
            "requires additional evidence",
            "additional evidence",
            "insufficient support",
            "insufficient to",
            "context is insufficient",
            "not fully support",
            "does not directly link",
            "not directly link",
            "lack of evidence",
            "absence of",
            "supplied context",
            "provided context does not",
            "context does not",
            "does not support",
            "does not directly support",
            "cannot accurately",
            "cannot be accurately",
            "cannot phrase",
            "cannot state",
            "not endorse",
            "not decrease",
            "not decreases",
            "rather than decreasing",
            "rather than blocking",
            "instead",
            "avoid unsupported",
            "avoids unsupported",
            "verify that",
            "flags uncertainties",
            "conflict with context",
            "treated as authoritative",
            "guide the benchmark",
            "which evidence to prioritize",
            "conflicts with evidence",
            "conflicts with",
            "inconsistent with evidence",
            "inconsistent with",
            "direct contradiction",
        )
    )


def _mentions_unacceptable_variant(claim: ExtractedClaim, gold: GoldClaim) -> bool:
    lower = claim.text.lower()
    if any(variant.lower().rstrip(".") in lower for variant in gold.unacceptable_variants):
        return True
    return "hgf" in lower and _has_met_token(lower) and any(word in lower for word in WRONG_DIRECTION_WORDS)



def _is_required_node_fragment(claim: ExtractedClaim, gold: GoldClaim, match_score: float) -> bool:
    lower = claim.text.lower()
    if _mentions_unacceptable_variant(claim, gold) or _overgeneralized(claim):
        return False
    if not (claim.entities or claim.relation.predicate):
        return False
    required_nodes = list(gold.required_mechanism_nodes or gold.entities or [])
    if not required_nodes:
        return False
    node_hits = sum(1 for node in required_nodes if _node_present(node, lower))
    if node_hits >= 2:
        return True
    if node_hits >= 1 and (claim.relation.predicate or _has_local_mechanism_edge(lower)) and match_score >= 0.2:
        return True
    return False



def _has_local_mechanism_edge(lower: str) -> bool:
    edge_markers = (
        "->",
        "→",
        "lead to",
        "leads to",
        "linked to",
        "link to",
        "contribute to",
        "contributes to",
        "associated with",
        "become stiff",
        "increased stiffness",
        "matrix stiffness",
    )
    return any(marker in lower for marker in edge_markers)

def _is_boundary_only_answer(claims: list[ExtractedClaim], tag_set: set[str]) -> bool:
    if not claims:
        return False
    if not (tag_set & {"oversimplification_trap", "mechanistic_completeness", "evidence_assembly"}):
        return False
    boundary_count = 0
    substantive_count = 0
    for claim in claims:
        lower = claim.text.lower()
        if _is_evidence_limitation_or_caveat(lower) or _is_evidence_assembly_boundary(lower) or _is_meta_observability_claim(lower, tag_set):
            boundary_count += 1
        elif claim.entities or claim.relation.predicate:
            substantive_count += 1
    return boundary_count > 0 and substantive_count == 0

def _is_evidence_limitation_or_caveat(lower: str) -> bool:
    return any(
        marker in lower
        for marker in (
            "current response is constrained",
            "response is constrained",
            "constrained by",
            "instruction to rely",
            "provided snippets",
            "provided context does not",
            "provided context snippets do not",
            "context snippets do not",
            "context does not",
            "not addressed",
            "unaddressed",
            "remains unsupported",
            "unsupported by the provided context",
            "unsupported by provided context",
            "unsupported by the current context",
            "unsupported by current context",
            "requires additional evidence",
            "additional evidence",
            "insufficient support",
            "insufficient to",
            "context is insufficient",
            "not fully support",
            "does not directly link",
            "not directly link",
            "lack of evidence",
            "absence of",
            "supplied context",
            "no mention",
            "no evidence",
            "no supported connection",
            "no snippets",
            "no snippet",
            "not refute",
            "not explicitly supported",
            "not explicitly",
            "not directly",
            "additional context",
            "external knowledge",
            "hypothetical connections",
            "with caveats",
            "without caveats",
            "context-dependent",
            "context dependent",
            "hypothesis-generating",
            "transfer hypothesis",
            "not as direct proof",
            "not direct proof",
            "avoid unsupported",
            "avoids unsupported",
            "verify that",
            "flags uncertainties",
            "conflict with context",
            "treated as authoritative",
            "guide the benchmark",
            "which evidence to prioritize",
        )
    )


def _is_evidence_assembly_boundary(lower: str) -> bool:
    return any(
        marker in lower
        for marker in (
            "missing link",
            "missing bridge",
            "supported partial",
            "partial structure",
            "clarification needed",
            "clarify",
            "which interpretation",
            "which evidence frame",
            "not specified",
            "lack explicit",
            "without inventing",
            "available evidence supports only",
            "evidence supports only the broader direction",
            "current evidence puzzle",
            "validated relation-evidence",
            "validated relation evidence",
            "unverified",
            "caveated",
            "explicitly caveated",
            "directly opposes",
            "opposes the user's claim",
            "invalid for",
            "inaccurate for",
            "wrong for",
            "do not add a detailed mechanism",
            "falsely claims",
            "cannot be validated",
            "directly contradicted",
            "lacks any contextual basis",
            "lacks contextual basis",
            "unfounded",
            "inaccurate",
        )
    )


def _contains_offtopic_terms(lower: str) -> bool:
    return any(
        marker in lower
        for marker in (
            "fda",
            "approval timeline",
            "drug approval",
            "drug pricing",
            "pricing",
            "clinical trial endpoint",
            "patient-specific treatment",
            "regulatory",
        )
    )


def _is_citation_scope_guidance(lower: str) -> bool:
    citation_frame = any(
        marker in lower
        for marker in (
            "citation",
            "evidence",
            "proof",
            "extrapolat",
            "validated",
            "validation",
        )
    )
    scope_frame = any(
        marker in lower
        for marker in (
            "general oncology",
            "lung-cancer-specific",
            "lung cancer-specific",
            "lung cancer",
            "lung-cancer",
        )
    )
    transfer_guard = any(
        marker in lower
        for marker in (
            "unless",
            "only if",
            "should be",
            "must be",
            "distinguish",
            "without",
            "not ",
            "requires",
            "require",
            "explicitly",
            "further validation",
        )
    )
    return citation_frame and scope_frame and transfer_guard


def _is_obvious_scope_drift(lower: str, trap_by_type: dict[str, InjectedTrap]) -> bool:
    if not _contains_offtopic_terms(lower):
        return False
    return any(
        trap.type == "scope_drift" or "scope" in trap.trap_id.lower()
        for trap in trap_by_type.values()
    )


def _directionally_supported_fragment(claim: ExtractedClaim, gold: GoldClaim, match_score: float) -> bool:
    lower = claim.text.lower()
    if "caf_heterogeneity" in gold.claim_id.lower():
        return "caf" in lower and any(marker in lower for marker in ("heterogen", "subtype", "context-dependent", "context dependent", "distinct", "vary"))
    if match_score < 0.35 or not claim.entities:
        return False
    if "hgf_met" in gold.claim_id.lower() or "met" in (gold.relation.object or "").lower():
        if not ("hgf" in lower and _has_met_token(lower)):
            return False
        if _wrong_direction_targets_met(lower):
            return False
        return any(
            marker in lower
            for marker in (
                "activate",
                "activates",
                "activating",
                "increases met",
                "binds to",
                "stimulates",
                "promotes emt",
                "hgf/met signaling",
                "met-mediated",
            )
        )
    return len(claim.entities) >= 2 and match_score >= 0.4


def _wrong_direction_targets_met(lower: str) -> bool:
    import re

    if re.search(r"hgf.{0,30}(decreases|decrease|reduces|reduce|blocks|block|suppresses|suppress|inhibits|inhibit).{0,30}\b(c-?met|met/c-met|met)\b", lower):
        return True
    if re.search(r"\b(c-?met|met/c-met|met)\b.{0,30}(decreased|reduced|blocked|suppressed|inhibited)", lower):
        return True
    if "blocks emt" in lower and "hgf" in lower and _has_met_token(lower):
        return True
    return False


def _is_meta_observability_claim(lower: str, tag_set: set[str]) -> bool:
    if not tag_set & {"agent_observability", "diagnosis"}:
        return False
    return any(
        marker in lower
        for marker in (
            "trace",
            "inspect",
            "evaluator",
            "expected behavior",
            "extracted claim",
            "claim judgment",
            "failure owner",
            "reward model",
            "reward penalties",
            "source sentence",
            "source sentence ids",
            "bm25",
            "retrieval records",
            "pinned context",
            "scope constraint",
            "user directive",
            "grounded",
            "discrepanc",
        )
    )


def _is_scope_correction_acknowledgement(lower: str, tag_set: set[str]) -> bool:
    if not (tag_set & {"scope_correction", "conversation_memory", "post_correction_adherence"}):
        return False
    return any(
        marker in lower
        for marker in (
            "session scope correction",
            "scope correction",
            "constrain later retrieval",
            "constrain later",
            "from now on",
            "going forward",
            "unless you explicitly ask",
            "i will use it to constrain",
        )
    )
