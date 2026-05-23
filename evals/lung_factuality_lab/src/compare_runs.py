from __future__ import annotations

import argparse
import json
from pathlib import Path


def compare_runs(before: str, after: str, out: str) -> dict:
    before_trace = _load_trace(Path(before))
    after_trace = _load_trace(Path(after))
    comparison = {
        "before": str(before),
        "after": str(after),
        "before_avg_reward": before_trace.get("avg_reward") or before_trace.get("aggregate", {}).get("avg_reward"),
        "after_avg_reward": after_trace.get("avg_reward") or after_trace.get("aggregate", {}).get("avg_reward"),
        "scenario_count_before": before_trace.get("scenario_count", 1),
        "scenario_count_after": after_trace.get("scenario_count", 1),
    }
    if comparison["before_avg_reward"] is not None and comparison["after_avg_reward"] is not None:
        comparison["reward_delta"] = round(float(comparison["after_avg_reward"]) - float(comparison["before_avg_reward"]), 4)
    Path(out).mkdir(parents=True, exist_ok=True)
    (Path(out) / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")
    (Path(out) / "comparison.md").write_text(_markdown(comparison), encoding="utf-8")
    return comparison


def _load_trace(path: Path) -> dict:
    target = path / "conversation_trace.json" if path.is_dir() else path
    return json.loads(target.read_text(encoding="utf-8"))


def _markdown(comparison: dict) -> str:
    return "\n".join(
        [
            "# Run Comparison",
            "",
            f"- Before avg reward: {comparison.get('before_avg_reward')}",
            f"- After avg reward: {comparison.get('after_avg_reward')}",
            f"- Reward delta: {comparison.get('reward_delta')}",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two lung factuality lab runs.")
    parser.add_argument("--before", required=True)
    parser.add_argument("--after", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    compare_runs(args.before, args.after, args.out)


if __name__ == "__main__":
    main()

