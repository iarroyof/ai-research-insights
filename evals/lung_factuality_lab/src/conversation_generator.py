from __future__ import annotations

from evals.lung_factuality_lab.src.conversation_loader import conversation_for_scenario
from evals.lung_factuality_lab.src.scenario_loader import LAB_ROOT, load_structured
from evals.lung_factuality_lab.src.schemas import ConversationTurn, Scenario


DEFAULT_VARIANTS = {
    "So HGF decreases MET signaling and blocks EMT, right?": [
        "Could HGF actually suppress MET activity and reduce EMT?",
        "Is it correct that CAF-derived HGF turns down MET signaling?",
        "Can I say HGF inhibits MET/c-MET and therefore prevents EMT?",
        "Would it be accurate to write that HGF blocks MET-driven EMT?",
    ],
    "HGF decreases MET signaling and therefore blocks EMT.": [
        "Could HGF actually suppress MET activity and reduce EMT?",
        "Is it correct that CAF-derived HGF turns down MET signaling?",
        "Can I say HGF inhibits MET/c-MET and therefore prevents EMT?",
        "Would it be accurate to write that HGF blocks MET-driven EMT?",
    ]
}


def generate_conversation(
    scenario: Scenario,
    *,
    variant_index: int | None = None,
    perturbation_bank_path: str = "",
) -> list[ConversationTurn]:
    """Return seed/scenario turns, optionally with deterministic user-turn variants.

    The canonical conversation lives in JSONL seed data. The generator can
    produce controlled variants while preserving turn ids, expected behavior,
    trap ids, scope constraints, and must-mention/must-not-claim metadata.
    """
    seed_turns = conversation_for_scenario(scenario)
    if seed_turns:
        if variant_index is None:
            return seed_turns
        return [_variant_turn(turn, variant_index, perturbation_bank_path) for turn in seed_turns]
    out: list[ConversationTurn] = []
    for trap in scenario.injected_traps:
        prompt = trap.wrong_direction or trap.wrong_claim or trap.expected_behavior
        out.append(
            ConversationTurn(
                turn=trap.turn,
                user_message=prompt,
                expected_behavior=trap.expected_behavior,
                trap_ids=[trap.trap_id],
            )
        )
    return sorted(out, key=lambda item: item.turn)


def _variant_turn(turn: ConversationTurn, variant_index: int, perturbation_bank_path: str = "") -> ConversationTurn:
    if not turn.trap_ids:
        return turn
    variants = _variant_map(perturbation_bank_path)
    text = turn.message
    replacements = variants.get(text) or []
    for must_not in turn.must_not_claim:
        replacements = replacements or variants.get(must_not) or []
        if replacements:
            break
        for key, values in variants.items():
            if must_not.lower().strip(".") in key.lower() or must_not.lower().strip(".") in text.lower():
                replacements = values
                break
    if not replacements:
        return turn
    replacement = replacements[variant_index % len(replacements)]
    data = turn.model_dump()
    data["text"] = replacement
    data["user_message"] = replacement
    data["variant_of"] = text
    return ConversationTurn(**{k: v for k, v in data.items() if k in ConversationTurn.model_fields})


def _variant_map(perturbation_bank_path: str = "") -> dict[str, list[str]]:
    out = dict(DEFAULT_VARIANTS)
    if not perturbation_bank_path:
        default_bank = LAB_ROOT / "data" / "perturbations" / "user_false_premise_bank.yaml"
        if default_bank.exists():
            perturbation_bank_path = str(default_bank)
    if not perturbation_bank_path:
        return out
    data = load_structured(perturbation_bank_path)
    for item in data.get("false_premises", []) if isinstance(data, dict) else []:
        wrong = item.get("wrong_claim") or item.get("user_prompt")
        variants = item.get("variants") or []
        if wrong and variants:
            out[str(wrong)] = [str(v) for v in variants]
        if item.get("user_prompt") and variants:
            out[str(item["user_prompt"])] = [str(v) for v in variants]
    return out
