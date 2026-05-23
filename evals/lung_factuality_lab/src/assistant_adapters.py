from __future__ import annotations

import json
import os
import urllib.request
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from evals.lung_factuality_lab.src.scenario_loader import LAB_ROOT, load_structured
from evals.lung_factuality_lab.src.schemas import AssistantAnswer, ConversationTurn, Scenario


class AssistantAdapter(Protocol):
    name: str

    def answer(self, scenario: Scenario, turn: ConversationTurn, history: list[dict]) -> AssistantAnswer:
        ...


@dataclass
class DummyAssistantAdapter:
    mode: str = "mixed"
    name: str = "dummy"

    def answer(self, scenario: Scenario, turn: ConversationTurn, history: list[dict]) -> AssistantAnswer:
        user = turn.message.lower()
        trap_type = ""
        traps = {trap.trap_id: trap for trap in scenario.injected_traps}
        for trap_id in turn.trap_ids:
            if trap_id in traps:
                trap_type = traps[trap_id].type
                break

        if self.mode == "correct":
            text = _correct_answer(user, scenario)
        elif self.mode == "factual_inversion" or trap_type == "factual_inversion":
            text = "Yes. HGF may reduce MET signaling and thereby block EMT in some lung cancer contexts."
        elif self.mode == "unsupported_plausible_mechanism" or trap_type == "unsupported_plausible_mechanism":
            text = "SALL4 methylation directly suppresses M2 polarization via IL-10 in lung cancer TME."
        elif self.mode == "vague_answer" or "basic" in scenario.scenario_id:
            text = "The tumor microenvironment affects cancer progression through immune and stromal effects."
        elif self.mode == "scope_drift" or trap_type == "scope_drift":
            text = "The most important issue is drug approval timelines, FDA labeling, and clinical trial endpoints."
        elif self.mode == "over_agree_with_user_false_premise":
            text = f"That premise is plausible: {turn.message}"
        elif "missing_met" in " ".join(turn.trap_ids):
            text = "CAF-derived HGF directly causes EMT and resistance through stromal remodeling."
        else:
            text = _correct_answer(user, scenario)
        return AssistantAnswer(
            scenario_id=scenario.scenario_id,
            turn=turn.turn,
            assistant=self.name if self.mode == "mixed" else f"{self.name}:{self.mode}",
            answer=text,
            adapter_meta={"mode": self.mode, "trap_ids": turn.trap_ids},
        )


@dataclass
class HttpChatAdapter:
    endpoint: str
    api_key: str = ""
    tenant_id: str = "eval-lab"
    request_timeout: float = 300.0
    name: str = "http"
    run_namespace: str = field(default_factory=lambda: uuid.uuid4().hex[:10])

    def answer(self, scenario: Scenario, turn: ConversationTurn, history: list[dict]) -> AssistantAnswer:
        payload = {
            "message": turn.message,
            "session_id": f"eval_{self.run_namespace}_{scenario.scenario_id}",
            "items": [],
            "options": {"allow_auto_context": True, "expose_memory_debug": True},
        }
        api_key = self.api_key or os.getenv("API_KEY", "")
        req = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "X-Tenant-Id": self.tenant_id,
                **({"X-API-Key": api_key} if api_key else {}),
            },
            method="POST",
        )
        chunks: list[str] = []
        event_meta: dict[str, object] = {}
        with urllib.request.urlopen(req, timeout=self.request_timeout) as res:
            for raw in res:
                line = raw.decode("utf-8", errors="ignore").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    event = json.loads(data)
                    if event.get("type") == "token":
                        chunks.append(str(event.get("data") or ""))
                    elif event.get("type") in {
                        "citations",
                        "memory_debug",
                        "reward",
                        "evidence_table",
                        "conversation_frame",
                    }:
                        event_meta[event.get("type")] = event.get("data") or {}
                    elif event.get("type") == "consistency_warning":
                        event_meta.setdefault("consistency_warning", []).append(event.get("data") or {})
                except Exception:
                    pass
        return AssistantAnswer(
            scenario_id=scenario.scenario_id,
            turn=turn.turn,
            assistant=self.name,
            answer="".join(chunks).strip(),
            adapter_meta={"endpoint": self.endpoint, "session_namespace": self.run_namespace, **event_meta},
        )


@dataclass
class WrongAnswerReplayAdapter:
    bank_path: str = "data/perturbations/assistant_wrong_answer_bank.yaml"
    name: str = "wrong_answer_replay"

    def __post_init__(self) -> None:
        path = LAB_ROOT / self.bank_path if not self.bank_path.startswith("/") else self.bank_path
        data = load_structured(path)
        self.answers = list(data if isinstance(data, list) else (data or {}).get("wrong_answers", []))

    def answer(self, scenario: Scenario, turn: ConversationTurn, history: list[dict]) -> AssistantAnswer:
        for item in self.answers:
            applies = item.get("applies_to", {})
            scenario_match = applies.get("scenario_id") in {scenario.scenario_id, scenario.base_scenario_id, turn.scenario_id}
            variant = applies.get("variant_index")
            variant_match = variant is None or turn.variant_index is None or int(variant) == int(turn.variant_index)
            if scenario_match and variant_match and int(applies.get("turn", -1)) == turn.turn:
                return AssistantAnswer(
                    scenario_id=scenario.scenario_id,
                    turn=turn.turn,
                    assistant=self.name,
                    answer=str(item.get("answer") or ""),
                    adapter_meta={
                        "wrong_answer_id": item.get("wrong_answer_id"),
                        "expected_judgment": item.get("expected_judgment", {}),
                        "expected_reward": item.get("expected_reward", {}),
                    },
                )
        return DummyAssistantAdapter(mode="correct").answer(scenario, turn, history)


@dataclass
class SavedAnswerReplayAdapter:
    replay_run_dir: str
    name: str = "saved_answer_replay"

    def __post_init__(self) -> None:
        if not self.replay_run_dir:
            raise ValueError("--answer-replay-dir is required for saved_answer_replay")
        self.root = Path(self.replay_run_dir)
        self._cache: dict[str, dict[int, AssistantAnswer]] = {}

    def answer(self, scenario: Scenario, turn: ConversationTurn, history: list[dict]) -> AssistantAnswer:
        answers = self._answers_for_scenario(scenario.scenario_id)
        if turn.turn not in answers:
            raise ValueError(f"No replayed answer for {scenario.scenario_id} turn {turn.turn} in {self.root}")
        saved = answers[turn.turn]
        return AssistantAnswer(
            scenario_id=scenario.scenario_id,
            turn=turn.turn,
            assistant=self.name,
            answer=saved.answer,
            adapter_meta={"replay_run_dir": str(self.root), "source_assistant": saved.assistant, **saved.adapter_meta},
        )

    def _answers_for_scenario(self, scenario_id: str) -> dict[int, AssistantAnswer]:
        if scenario_id in self._cache:
            return self._cache[scenario_id]
        path = self.root / scenario_id / "assistant_answers.jsonl"
        if not path.exists():
            single_path = self.root / "assistant_answers.jsonl"
            path = single_path if single_path.exists() else path
        if not path.exists():
            raise FileNotFoundError(f"Replay answer file not found: {path}")
        answers: dict[int, AssistantAnswer] = {}
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                item = AssistantAnswer(**json.loads(line))
                answers[item.turn] = item
        self._cache[scenario_id] = answers
        return answers


def build_adapter(
    name: str,
    *,
    mode: str = "mixed",
    endpoint: str = "",
    api_key: str = "",
    tenant_id: str = "eval-lab",
    request_timeout: float = 300.0,
    wrong_answer_bank: str = "",
    answer_replay_dir: str = "",
) -> AssistantAdapter:
    if name == "dummy":
        return DummyAssistantAdapter(mode=mode)
    if name in {"wrong_answer_replay", "wrong_replay"}:
        return WrongAnswerReplayAdapter(bank_path=wrong_answer_bank or "data/perturbations/assistant_wrong_answer_bank.yaml")
    if name in {"saved_answer_replay", "answer_replay", "replay"}:
        return SavedAnswerReplayAdapter(replay_run_dir=answer_replay_dir)
    if name in {"http", "target_chatbot"}:
        if not endpoint:
            raise ValueError("--endpoint is required for http/target_chatbot adapter")
        return HttpChatAdapter(endpoint=endpoint, api_key=api_key, tenant_id=tenant_id, request_timeout=request_timeout, name=name)
    raise ValueError(f"Unknown assistant adapter: {name}")


def _correct_answer(user: str, scenario: Scenario) -> str:
    if "trace evidence" in user or "evaluator disagrees" in user or "before changing code" in user:
        return (
            "The agent should inspect the user turn, expected behavior, extracted claims, relation polarity, "
            "matched gold claim, required mechanism nodes, scope constraints, penalties, turn reward, and failure owner before changing code."
        )
    if "hgf" in user or "met" in user or "hgf_met" in scenario.scenario_id:
        return (
            "CAF-derived HGF is generally described as activating MET/c-MET signaling, "
            "which can contribute to downstream signaling, EMT-associated progression, and therapy resistance. "
            "I would not agree that HGF decreases MET signaling unless specific evidence showed that context."
        )
    if "mdsc" in scenario.scenario_id or "treg" in scenario.scenario_id:
        return (
            "MDSCs and Tregs can support an immunosuppressive lung-cancer TME by limiting effector T-cell activity, "
            "promoting immune evasion, and contributing to checkpoint-resistance discussions."
        )
    if "tam" in scenario.scenario_id or "cd8" in user:
        return (
            "M2-like TAMs usually support immunosuppression, cytokine signaling, angiogenesis, and impaired cytotoxic T-cell activity; "
            "they should not be framed as mainly activating anti-tumor CD8 immunity."
        )
    if "hypoxia" in scenario.scenario_id:
        return (
            "Hypoxia can promote HIF-linked angiogenesis, metabolic adaptation, immune suppression, PD-L1-related immune escape, and selection of invasive tumor phenotypes."
        )
    if "ecm" in scenario.scenario_id or "stiffness" in user:
        return (
            "CAF-associated ECM remodeling and stiffness can create invasion, drug-delivery, and immune-infiltration barriers, "
            "but the effect should be stated with context-dependent caveats."
        )
    if (
        "cross" in scenario.scenario_id
        or "citation_drift" in scenario.scenario_id
        or "correction_scope" in scenario.scenario_id
        or "breast" in user
        or "general oncology" in user
    ):
        return (
            "CAFs are heterogeneous in lung cancer, so evidence from breast cancer, pancreatic cancer, or general oncology should be framed "
            "as background or a transfer hypothesis, not as direct proof in NSCLC without caveats."
        )
    return "TME factors include CAF signaling, immune suppression, hypoxia, angiogenesis, metabolic crosstalk, and ECM remodeling."
