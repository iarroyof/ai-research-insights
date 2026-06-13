#!/usr/bin/env python3
"""promote_vocab.py — Global vocabulary promotion gate (Module 5).

Standalone CLI script. NEVER runs automatically. Only invoked manually after
sentinel guard clears.

Usage:
    python3 promote_vocab.py --run-id shape8_sentinel_a_v3 [--dry-run]

Steps:
  1. Load VocabularyStore().global_candidates(min_run_count=2, min_success_rate=0.65)
  2. For each candidate:
     a. Semantic diversity check (4-gram Jaccard against existing ANCHOR_ALIASES keys)
     b. Cross-frame check (appeared in >1 search frame type)
     c. If both pass AND --dry-run is not set:
        - Patch ANCHOR_ALIASES in-memory
        - Append to configs/anchor_aliases_additions.yaml
  3. Print promotion report
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import yaml

# Allow running from the eval dir or project root
_BASE = os.path.dirname(os.path.abspath(__file__))
# Walk up to find project root containing 'services/'
_ROOT = _BASE
for _ in range(6):
    if os.path.isdir(os.path.join(_ROOT, "services")):
        break
    _ROOT = os.path.dirname(_ROOT)

# Add the API app to path
_API_APP = os.path.join(_ROOT, "services", "api")
if _API_APP not in sys.path:
    sys.path.insert(0, _API_APP)

try:
    from app.memory.vocabulary_store import VocabularyStore
    from app.memory.search_agent import ANCHOR_ALIASES, PUZZLE_NODE_STOP_TERMS
except ImportError as e:
    print(f"[ERROR] Could not import API modules: {e}", file=sys.stderr)
    print("Run this script from within the API container or with PYTHONPATH set.", file=sys.stderr)
    sys.exit(1)

# Staging file for reviewed additions
_STAGING_FILE = os.path.join(_ROOT, "configs", "anchor_aliases_additions.yaml")


# ---------------------------------------------------------------------------
# Diversity helpers
# ---------------------------------------------------------------------------

def _char_ngrams(text: str, n: int = 4) -> set[str]:
    """Return set of character n-grams from text."""
    text = text.lower().replace(" ", "_")
    if len(text) < n:
        return {text}
    return {text[i:i + n] for i in range(len(text) - n + 1)}


def _jaccard_similarity(a: str, b: str, n: int = 4) -> float:
    """Character 4-gram Jaccard similarity between two strings."""
    sa = _char_ngrams(a, n)
    sb = _char_ngrams(b, n)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


def _is_near_duplicate(candidate: str, existing_keys: list[str], threshold: float = 0.75) -> tuple[bool, str]:
    """Return (is_duplicate, closest_match) if max_similarity > threshold."""
    best_match = ""
    best_sim = 0.0
    for key in existing_keys:
        sim = _jaccard_similarity(candidate, key)
        if sim > best_sim:
            best_sim = sim
            best_match = key
    return (best_sim > threshold, best_match)


# ---------------------------------------------------------------------------
# Cross-frame check helpers
# ---------------------------------------------------------------------------

_KNOWN_FRAME_TYPES = {
    "mechanism_or_pathway",
    "evidence_question",
    "cross_domain_or_analogy",
    "general_biomedical",
}


def _get_frame_types_for_term(vs: VocabularyStore, term: str, run_id: str) -> set[str]:
    """Read run-scope records and determine which frame types the term appeared in.

    This is a best-effort approximation: the run-scope key stores aggregate
    alpha/beta but not per-frame breakdowns. As a proxy we check if the term
    appeared in multiple query shapes via the stored aliases.

    If the global_candidates record has aliases spanning multiple frame-typical
    vocabularies, we count those. Otherwise we return {'unknown'}.
    """
    # Read global_candidate data for this term's aliases
    gc_keys = vs._scan_prefix(f"vocabstore:global_candidate:shared:{term}")
    if not gc_keys:
        return {"unknown"}

    data: dict = {}
    r = vs._get_redis() if hasattr(vs, "_get_redis") else None
    if r is not None:
        try:
            raw = r.hgetall(gc_keys[0])
            data = {
                (k.decode() if isinstance(k, bytes) else k):
                (v.decode() if isinstance(v, bytes) else v)
                for k, v in raw.items()
            }
        except Exception:
            pass

    # Approximate cross-frame inference: if a term has high alpha (many successes)
    # across multiple run_ids, we consider it cross-frame.  Strict per-frame
    # tracking would require instrumenting build_auto_context with frame_id logging,
    # which is deferred to a future work package.  For now, use a conservative
    # heuristic: terms from >= 2 run_ids are treated as cross-frame.
    run_ids_raw = data.get("run_ids", "")
    run_ids = [r for r in str(run_ids_raw).split(",") if r]
    if len(run_ids) >= 2:
        # Cross-frame heuristic: multiple runs imply different search contexts
        return {"mechanism_or_pathway", "evidence_question"}
    return {"unknown"}


def _is_single_frame(candidate: str, run_id: str, vs: VocabularyStore) -> bool:
    """Return True if term only appeared in a single frame type (frame-specific)."""
    frame_types = _get_frame_types_for_term(vs, candidate, run_id)
    return len(frame_types) <= 1


# ---------------------------------------------------------------------------
# Main promotion logic
# ---------------------------------------------------------------------------

def _load_staging_file() -> dict:
    """Load existing staging file or return empty dict."""
    if not os.path.exists(_STAGING_FILE):
        return {"additions": [], "metadata": {"last_updated": None}}
    try:
        with open(_STAGING_FILE, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {"additions": [], "metadata": {}}
    except Exception:
        return {"additions": [], "metadata": {}}


def _write_staging_file(staging: dict) -> None:
    """Write updated staging file."""
    os.makedirs(os.path.dirname(_STAGING_FILE), exist_ok=True)
    staging["metadata"]["last_updated"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with open(_STAGING_FILE, "w", encoding="utf-8") as f:
        yaml.dump(staging, f, default_flow_style=False, allow_unicode=True)


def run_promotion(run_id: str, dry_run: bool) -> None:
    """Execute the promotion gate for a given run_id."""
    os.environ.setdefault("VOCAB_STORE_ENABLED", "true")

    vs = VocabularyStore()
    if not vs.enabled():
        print("[ERROR] VocabularyStore is not enabled or Redis is unavailable.")
        print("Set VOCAB_STORE_ENABLED=true and ensure Redis is reachable.")
        sys.exit(1)

    candidates = vs.global_candidates(min_run_count=2, min_success_rate=0.65)
    if not candidates:
        print("[INFO] No global candidates found meeting criteria.")
        return

    existing_keys = list(ANCHOR_ALIASES.keys())
    staging = _load_staging_file()
    existing_staging_terms = {entry["term"] for entry in staging.get("additions", [])}

    promoted: list[str] = []
    blocked_near_dup: list[tuple[str, str]] = []
    blocked_single_frame: list[str] = []
    blocked_stopword: list[str] = []
    pending_holdout: list[str] = []

    for cand in candidates:
        term = cand["term"]

        # Gate 0: stopword check
        if term in PUZZLE_NODE_STOP_TERMS:
            blocked_stopword.append(term)
            continue

        # Gate 1: diversity check
        is_dup, closest = _is_near_duplicate(term, existing_keys)
        if is_dup:
            blocked_near_dup.append((term, closest))
            continue

        # Gate 2: cross-frame check
        if _is_single_frame(term, run_id, vs):
            blocked_single_frame.append(term)
            continue

        # Passed all gates
        if dry_run:
            pending_holdout.append(term)
        else:
            # Patch ANCHOR_ALIASES in-memory (runtime patch — container only)
            aliases_set = set(cand.get("aliases", [term]))
            aliases_set.add(term)
            ANCHOR_ALIASES[term] = aliases_set
            print(f"  [PATCH] ANCHOR_ALIASES['{term}'] = {sorted(aliases_set)}")

            # Append to staging file for code-review
            if term not in existing_staging_terms:
                staging.setdefault("additions", []).append({
                    "term": term,
                    "aliases": sorted(aliases_set),
                    "success_rate": cand["success_rate"],
                    "run_count": cand["run_count"],
                    "run_ids": cand["run_ids"],
                    "promoted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "promoted_by_run": run_id,
                })
            promoted.append(term)

    if not dry_run and promoted:
        _write_staging_file(staging)

    # ------------------------------------------------------------------
    # Print report
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print(f"  Vocabulary Promotion Report  (run_id={run_id})")
    print("=" * 60)
    print(f"  Mode: {'DRY-RUN (no changes made)' if dry_run else 'LIVE'}")
    print()

    if dry_run and pending_holdout:
        print(f"  Pending holdout validation ({len(pending_holdout)} terms):")
        for t in pending_holdout:
            print(f"    + {t}")
    elif promoted:
        print(f"  Promoted to ANCHOR_ALIASES ({len(promoted)} terms):")
        for t in promoted:
            print(f"    + {t}")
        print(f"\n  Staging file: {_STAGING_FILE}")
        print("  (Requires code review + semantic_drift_holdout before merging to source)")
    else:
        print("  No terms promoted.")

    if blocked_near_dup:
        print(f"\n  Blocked (near-duplicate, {len(blocked_near_dup)} terms):")
        for t, closest_match in blocked_near_dup:
            print(f"    - {t}  (similar to: {closest_match})")

    if blocked_single_frame:
        print(f"\n  Blocked (single-frame only, {len(blocked_single_frame)} terms):")
        for t in blocked_single_frame:
            print(f"    - {t}")

    if blocked_stopword:
        print(f"\n  Blocked (PUZZLE_NODE_STOP_TERMS, {len(blocked_stopword)} terms):")
        for t in blocked_stopword:
            print(f"    - {t}")

    print()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Global vocabulary promotion gate (Module 5).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--run-id",
        required=True,
        help="Eval run ID (e.g. shape8_sentinel_a_v3)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Print what would be promoted without modifying ANCHOR_ALIASES or staging file.",
    )
    args = parser.parse_args()
    run_promotion(args.run_id, args.dry_run)


if __name__ == "__main__":
    main()
