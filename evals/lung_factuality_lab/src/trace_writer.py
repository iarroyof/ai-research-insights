from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from pydantic import BaseModel

from evals.lung_factuality_lab.src.scenario_loader import write_structured


def model_to_dict(item):
    if isinstance(item, BaseModel):
        return item.model_dump(mode="json")
    return item


def write_json(path: str | Path, data) -> None:
    Path(path).write_text(json.dumps(model_to_dict(data), indent=2, sort_keys=False) + "\n", encoding="utf-8")


def write_jsonl(path: str | Path, items: Iterable) -> None:
    lines = [json.dumps(model_to_dict(item), sort_keys=False) for item in items]
    Path(path).write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def write_yaml_like(path: str | Path, data) -> None:
    write_structured(path, model_to_dict(data))

