# services/api/app/services/zero_shot.py
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Dict

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline,
)


# Global singleton
_NLI = None

MODEL_ID = "facebook/bart-large-mnli"


def _cache_root() -> Path:
    """Root of the HF cache inside the container (bind-mounted)."""
    return Path(os.getenv("HF_HOME", "/models"))


def _resolve_local_snapshot(model_id: str = MODEL_ID) -> Path:
    """
    Locate a local snapshot directory for the given model inside HF cache, without network calls.

    Expected layout:
      /models/models--facebook--bart-large-mnli/snapshots/<rev>/{config.json, tokenizer.json, pytorch_model.bin, ...}
    """
    root = _cache_root()
    org, name = model_id.split("/", 1)
    cache_dir = root / f"models--{org}--{name}"
    snapshots_dir = cache_dir / "snapshots"
    if not snapshots_dir.exists():
        raise FileNotFoundError(f"HF cache snapshots dir not found: {snapshots_dir}")

    # newest first
    candidates = sorted(
        (p for p in snapshots_dir.iterdir() if p.is_dir()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No snapshots under {snapshots_dir}")

    for snap in candidates:
        if (snap / "config.json").exists():
            return snap

    # Fallback to the newest; load will error with a clear message if invalid.
    return candidates[0]


def _build_offline_pipeline() -> pipeline:
    """Build a zero-shot pipeline from local snapshot only (no network)."""
    snapshot_dir = _resolve_local_snapshot(MODEL_ID)

    tok = AutoTokenizer.from_pretrained(snapshot_dir, local_files_only=True)
    mdl = AutoModelForSequenceClassification.from_pretrained(
        snapshot_dir, local_files_only=True
    )

    return pipeline(
        task="zero-shot-classification",
        model=mdl,
        tokenizer=tok,
        device=-1,  # CPU by default; set to 0 if your api image has CUDA PyTorch
    )


def _build_online_pipeline() -> pipeline:
    """
    Online fallback: download the model into the mounted cache and build the pipeline.
    This runs only if the offline path fails.
    """
    cache_dir = _cache_root()
    # Let transformers handle download; ensure it caches under /models
    return pipeline(
        task="zero-shot-classification",
        model=MODEL_ID,
        device=-1,
        # Ensure we write into the mounted cache and don't go elsewhere
        model_kwargs={"cache_dir": str(cache_dir)},
        tokenizer_kwargs={"cache_dir": str(cache_dir)},
    )


def _get_nli() -> pipeline:
    global _NLI
    if _NLI is not None:
        return _NLI

    # 1) Try strictly offline
    try:
        _NLI = _build_offline_pipeline()
        return _NLI
    except Exception as offline_err:
        # 2) Fallback online (will fill the cache for next runs)
        try:
            _NLI = _build_online_pipeline()
            return _NLI
        except Exception as online_err:
            # Bubble up a clear error
            raise RuntimeError(
                f"Failed to load zero-shot model offline ({offline_err}) "
                f"and online fallback also failed ({online_err})."
            )


def score_labels(texts: List[str], labels: List[str]) -> List[Dict[str, float]]:
    """
    For each input text, return {label: probability}. Multi-label mode.
    """
    if not texts:
        return []
    if not labels:
        return [{} for _ in texts]

    nli = _get_nli()
    results = nli(texts, candidate_labels=labels, multi_label=True)

    if isinstance(results, dict):
        results = [results]

    out: List[Dict[str, float]] = []
    for r in results:
        labs = r["labels"]
        scores = r["scores"]
        out.append({labs[i]: float(scores[i]) for i in range(len(labs))})
    return out

