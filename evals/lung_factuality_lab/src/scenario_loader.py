from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from evals.lung_factuality_lab.src.schemas import GoldClaim, InjectedTrap, MechanismGraph, Scenario


LAB_ROOT = Path(__file__).resolve().parents[1]


def load_structured(path: str | Path) -> Any:
    text = Path(path).read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        return yaml.safe_load(text)
    except Exception:
        return json.loads(text)


def write_structured(path: str | Path, data: Any) -> None:
    Path(path).write_text(json.dumps(data, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def _items(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, dict):
        if "items" in data:
            return list(data["items"] or [])
        if "scenarios" in data:
            return list(data["scenarios"] or [])
        if "claims" in data:
            return list(data["claims"] or [])
        if "graphs" in data:
            return list(data["graphs"] or [])
    return list(data or [])


def load_gold_claims(path: str | Path | None = None) -> dict[str, GoldClaim]:
    paths = [Path(path)] if path else _existing(
        LAB_ROOT / "data" / "evidence" / "gold_claims.yaml",
        LAB_ROOT / "data" / "evidence" / "generated_gold_claims.yaml",
    )
    claims: dict[str, GoldClaim] = {}
    for item_path in paths:
        for raw in _items(load_structured(item_path)):
            normalized = dict(raw)
            if "required_nodes" in normalized and "required_mechanism_nodes" not in normalized:
                normalized["required_mechanism_nodes"] = normalized.get("required_nodes") or []
            if not normalized.get("entities"):
                normalized["entities"] = list(normalized.get("required_mechanism_nodes") or normalized.get("required_nodes") or [])
            claim = GoldClaim(**normalized)
            claims[claim.claim_id] = claim
    return claims


def load_mechanism_graphs(path: str | Path | None = None) -> dict[str, MechanismGraph]:
    paths = [Path(path)] if path else _existing(
        LAB_ROOT / "data" / "evidence" / "mechanism_graphs.yaml",
        LAB_ROOT / "data" / "evidence" / "generated_mechanism_graphs.yaml",
    )
    graphs: dict[str, MechanismGraph] = {}
    for item_path in paths:
        for raw in _items(load_structured(item_path)):
            graph = MechanismGraph(**raw)
            graphs[graph.graph_id] = graph
    return graphs


def scenario_paths() -> list[Path]:
    return sorted((LAB_ROOT / "data" / "scenarios").glob("*.yaml"))


def load_scenarios(paths: list[str | Path] | None = None) -> dict[str, Scenario]:
    paths = [Path(p) for p in paths] if paths else scenario_paths()
    scenarios: dict[str, Scenario] = {}
    for path in paths:
        for raw in _items(load_structured(path)):
            scenario = Scenario(**raw)
            scenarios[scenario.scenario_id] = scenario
    return scenarios


def load_scenario(scenario_id: str) -> Scenario:
    scenarios = load_scenarios()
    if scenario_id not in scenarios:
        available = ", ".join(sorted(scenarios))
        raise KeyError(f"Unknown scenario {scenario_id!r}. Available: {available}")
    return scenarios[scenario_id]


def load_traps(paths: list[str | Path] | None = None) -> dict[str, InjectedTrap]:
    paths = [Path(p) for p in paths] if paths else _existing(
        LAB_ROOT / "data" / "perturbations" / "user_false_premise_bank.yaml",
        LAB_ROOT / "data" / "perturbations" / "generated_user_false_premise_bank.yaml",
    )
    traps: dict[str, InjectedTrap] = {}
    for path in paths:
        data = load_structured(path)
        raw_items = []
        if isinstance(data, dict):
            for key in ("false_premises", "traps", "items"):
                raw_items.extend(data.get(key) or [])
        else:
            raw_items.extend(data or [])
        for raw in raw_items:
            normalized = dict(raw)
            normalized.setdefault("turn", 0)
            normalized.setdefault("expected_behavior", normalized.get("expected_behavior") or "Detect and handle the injected trap.")
            trap = InjectedTrap(**normalized)
            traps[trap.trap_id] = trap
    return traps


def load_reward_config(path: str | Path | None = None) -> dict[str, Any]:
    path = Path(path) if path else LAB_ROOT / "configs" / "reward_weights.yaml"
    return dict(load_structured(path) or {})


def load_batch_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_absolute() and not path.exists():
        path = LAB_ROOT / path
    return dict(load_structured(path) or {})


def _existing(*paths: Path) -> list[Path]:
    return [path for path in paths if path.exists()]
