from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


ClaimLabel = Literal[
    "supported",
    "partially_supported",
    "unsupported",
    "contradicted",
    "out_of_scope",
    "overgeneralized",
    "too_vague",
]


class Relation(BaseModel):
    subject: str = ""
    predicate: str = ""
    object: str = ""


class GoldClaim(BaseModel):
    claim_id: str
    domain: str
    topic: str
    claim: str
    entities: list[str] = Field(default_factory=list)
    relation: Relation = Field(default_factory=Relation)
    qualifiers: dict[str, Any] = Field(default_factory=dict)
    unacceptable_variants: list[str] = Field(default_factory=list)
    required_mechanism_nodes: list[str] = Field(default_factory=list)
    evidence_strength: str = "review_supported"
    scope: str = ""
    note: str = ""


class MechanismEdge(BaseModel):
    source: str
    relation: str
    target: str


class MechanismGraph(BaseModel):
    graph_id: str
    target_gold_claim: str = ""
    topic: str
    required_nodes: list[str] = Field(default_factory=list)
    preferred_edges: list[MechanismEdge] = Field(default_factory=list)
    common_errors: list[str] = Field(default_factory=list)


class InjectedTrap(BaseModel):
    trap_id: str
    turn: int
    type: str
    severity: int = Field(ge=1, le=5)
    wrong_claim: str = ""
    wrong_direction: str = ""
    expected_behavior: str
    target_gold_claims: list[str] = Field(default_factory=list)
    failure_if_assistant_claims: list[str] = Field(default_factory=list)


class ConversationTurn(BaseModel):
    conversation_id: str = ""
    scenario_id: str = ""
    variant_index: int | None = None
    turn: int
    role: str = "user"
    user_message: str = ""
    text: str = ""
    expected_behavior: str = ""
    target_gold_claims: list[str] = Field(default_factory=list)
    expected_focus_terms: list[str] = Field(default_factory=list)
    correction_terms: list[str] = Field(default_factory=list)
    trap_ids: list[str] = Field(default_factory=list)
    must_mention: list[str] = Field(default_factory=list)
    must_not_claim: list[str] = Field(default_factory=list)
    scope: str = ""
    tags: list[str] = Field(default_factory=list)

    @property
    def message(self) -> str:
        return self.user_message or self.text


class Scenario(BaseModel):
    scenario_id: str
    base_scenario_id: str = ""
    conversation_file: str = ""
    scenario_type: str
    domain: str
    user_profile: str
    conversation_goal: str
    expected_capabilities: list[str] = Field(default_factory=list)
    target_gold_claims: list[str] = Field(default_factory=list)
    target_mechanism_graphs: list[str] = Field(default_factory=list)
    injected_traps: list[InjectedTrap] = Field(default_factory=list)
    turns: list[ConversationTurn] = Field(default_factory=list)
    success_criteria: dict[str, Any] = Field(default_factory=dict)


class AssistantAnswer(BaseModel):
    scenario_id: str
    turn: int
    assistant: str
    answer: str
    adapter_meta: dict[str, Any] = Field(default_factory=dict)


class ExtractedClaim(BaseModel):
    claim_id: str
    turn: int
    text: str
    entities: list[str] = Field(default_factory=list)
    relation: Relation = Field(default_factory=Relation)
    polarity: str = "affirmed"
    confidence: float = 0.5


class ClaimJudgment(BaseModel):
    claim_id: str
    label: ClaimLabel
    matched_gold_claim: str | None = None
    matched_mechanism_graph: str | None = None
    reason: str
    confidence: float = 0.0
    error_type: str | None = None
    severity: int = 0
    missing_nodes: list[str] = Field(default_factory=list)
    trap_id: str | None = None


class TurnScore(BaseModel):
    turn: int
    turn_reward: float
    component_scores: dict[str, float]
    penalties_applied: list[dict[str, Any]] = Field(default_factory=list)
    interpretation: str


class TurnTrace(BaseModel):
    turn: int
    user_message: str
    assistant_answer: str
    assistant_metadata: dict[str, Any] = Field(default_factory=dict)
    expected_behavior: str = ""
    injected_traps: list[InjectedTrap] = Field(default_factory=list)
    extracted_claims: list[ExtractedClaim] = Field(default_factory=list)
    claim_judgments: list[ClaimJudgment] = Field(default_factory=list)
    scores: TurnScore
    diagnosis: dict[str, Any] = Field(default_factory=dict)


class ConversationTrace(BaseModel):
    run_id: str
    scenario_id: str
    assistant: str
    turns: list[TurnTrace] = Field(default_factory=list)
    aggregate: dict[str, Any] = Field(default_factory=dict)


class FailureItem(BaseModel):
    failure_id: str
    scenario_id: str | None = None
    turn: int
    severity: int
    category: str
    short_description: str
    expected: str
    actual: str
    detected_by_evaluator: bool
    penalized_sufficiently: bool
    root_cause: str
    recommended_action_type: str
    failure_owner: str
    trap_id: str | None = None


class FailureBoard(BaseModel):
    run_id: str
    scenario_id: str | None = None
    failure_summary: dict[str, Any]
    failures: list[FailureItem] = Field(default_factory=list)


class Recommendation(BaseModel):
    priority: str
    target: str
    problem: str
    evidence: dict[str, Any] = Field(default_factory=dict)
    recommended_change: dict[str, Any] = Field(default_factory=dict)
    validation: dict[str, Any] = Field(default_factory=dict)


class RegressionTest(BaseModel):
    regression_id: str
    created_from_failure: str
    scenario: str
    invariant: str
    expected_claim_judgment: dict[str, Any] = Field(default_factory=dict)
    expected_behavior: list[str] = Field(default_factory=list)
    max_allowed_reward: float | None = None
    min_allowed_reward: float | None = None
