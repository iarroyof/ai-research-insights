from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from app.memory.evaluation import simulate_quality_by_session
from app.memory.store import MemoryStore
from app.search.os_client import os_client


def _fetch_docs(tenant: str, limit: int) -> list[dict[str, Any]]:
    store = MemoryStore(tenant)
    body = {
        "size": min(max(1, limit), 10000),
        "query": {
            "bool": {
                "filter": [
                    {
                        "terms": {
                            "doc_type": [
                                "message",
                                "episodic_summary",
                                "landmark",
                                "user_correction",
                            ]
                        }
                    }
                ]
            }
        },
        "sort": [{"session_id": {"order": "asc"}}, {"turn_index": {"order": "asc"}}],
    }
    res = os_client().search(index=store.index, body=body)
    return [hit.get("_source", {}) for hit in res.get("hits", {}).get("hits", [])]


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay saved conversations as memory-quality simulation seeds.")
    parser.add_argument("--tenant", default="default")
    parser.add_argument("--limit", type=int, default=2000)
    parser.add_argument("--token-budget", type=int, default=None)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    docs = _fetch_docs(args.tenant, args.limit)
    report = simulate_quality_by_session(docs, token_budget=args.token_budget)
    report["tenant"] = args.tenant
    report["source_doc_count"] = len(docs)
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
