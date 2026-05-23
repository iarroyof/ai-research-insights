from __future__ import annotations

import json
from pathlib import Path

from evals.lung_factuality_lab.src.scenario_loader import LAB_ROOT
from evals.lung_factuality_lab.src.schemas import ConversationTurn, Scenario


def resolve_lab_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    if path.exists():
        return path
    return LAB_ROOT / path


def load_seed_conversation(path: str | Path) -> list[ConversationTurn]:
    resolved = resolve_lab_path(path)
    turns: list[ConversationTurn] = []
    for line in resolved.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        raw.setdefault("user_message", raw.get("text", ""))
        raw.setdefault("expected_focus_terms", raw.get("must_mention", []))
        turns.append(ConversationTurn(**raw))
    return sorted(turns, key=lambda item: item.turn)


def conversation_for_scenario(scenario: Scenario) -> list[ConversationTurn]:
    if scenario.conversation_file:
        return load_seed_conversation(scenario.conversation_file)
    return scenario.turns


def write_generated_conversation(path: str | Path, turns: list[ConversationTurn]) -> None:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(
        "\n".join(turn.model_dump_json(exclude_none=True) for turn in turns) + ("\n" if turns else ""),
        encoding="utf-8",
    )

