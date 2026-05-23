# services/api/app/services/zero_shot.py
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, List, Dict

import httpx

from app.services.provider_metrics import record_provider_call
from app.services.provider_queue import provider_slot


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


def _load_transformers() -> tuple[Any, Any, Any]:
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
    except ImportError as exc:
        raise RuntimeError(
            "Local zero-shot classification requires the full/local-ML API image "
            "with transformers installed. Hosted/no-GPU mode intentionally omits "
            "that dependency."
        ) from exc
    return AutoTokenizer, AutoModelForSequenceClassification, pipeline


def _build_offline_pipeline() -> Any:
    """Build a zero-shot pipeline from local snapshot only (no network)."""
    AutoTokenizer, AutoModelForSequenceClassification, pipeline = _load_transformers()
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


def _build_online_pipeline() -> Any:
    """
    Online fallback: download the model into the mounted cache and build the pipeline.
    This runs only if the offline path fails.
    """
    cache_dir = _cache_root()
    _, _, pipeline = _load_transformers()
    # Let transformers handle download; ensure it caches under /models
    return pipeline(
        task="zero-shot-classification",
        model=MODEL_ID,
        device=-1,
        # Ensure we write into the mounted cache and don't go elsewhere
        model_kwargs={"cache_dir": str(cache_dir)},
        tokenizer_kwargs={"cache_dir": str(cache_dir)},
    )


def _get_nli() -> Any:
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


def _clean_token(value: str | None) -> str:
    if not value:
        return ""
    value = value.strip().strip('"').strip("'")
    if "hf_" in value:
        value = "hf_" + value.split("hf_", 1)[1]
    return value.strip()


def _hf_zero_shot_url() -> str:
    base = os.getenv(
        "ZERO_SHOT_HF_API_BASE_URL",
        os.getenv("HF_API_BASE_URL", "https://router.huggingface.co/hf-inference/models"),
    ).rstrip("/")
    model = os.getenv("ZERO_SHOT_MODEL", MODEL_ID).strip() or MODEL_ID
    return f"{base}/{model}"


def _retry_config() -> tuple[int, float]:
    retries = int(os.getenv("ZERO_SHOT_HF_API_MAX_RETRIES", "2"))
    backoff = float(os.getenv("ZERO_SHOT_HF_API_RETRY_BACKOFF_SEC", "2.0"))
    return max(0, retries), max(0.0, backoff)


def _retry_delay(attempt: int, backoff: float, response: httpx.Response | None = None) -> float:
    if response is not None:
        retry_after = response.headers.get("retry-after")
        if retry_after:
            try:
                return max(0.0, float(retry_after))
            except ValueError:
                pass
    return backoff * (2 ** max(0, attempt - 1))


def _is_retryable_status(status_code: int) -> bool:
    return status_code in {408, 409, 425, 429, 500, 502, 503, 504}


def _batch_size() -> int:
    try:
        return max(1, int(os.getenv("ZERO_SHOT_HF_API_BATCH_SIZE", os.getenv("ZERO_SHOT_BATCH_SIZE", "8"))))
    except ValueError:
        return 8


def _chunks(items: List[str], size: int) -> List[List[str]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def _parse_zero_shot_response(data: Any) -> Dict[str, float]:
    if isinstance(data, list):
        if not data:
            return {}
        if all(isinstance(item, dict) and "label" in item and "score" in item for item in data):
            return {str(item["label"]): float(item["score"]) for item in data}
        if len(data) == 1:
            return _parse_zero_shot_response(data[0])
    if not isinstance(data, dict):
        raise RuntimeError(f"Unexpected zero-shot HF response type: {type(data).__name__}")
    if "label" in data and "score" in data:
        return {str(data["label"]): float(data["score"])}
    labels = data.get("labels")
    scores = data.get("scores")
    if not isinstance(labels, list) or not isinstance(scores, list):
        raise RuntimeError("Unexpected zero-shot HF response payload; missing labels/scores")
    return {str(label): float(score) for label, score in zip(labels, scores)}


def _parse_zero_shot_batch_response(data: Any, expected_count: int) -> List[Dict[str, float]]:
    if expected_count <= 1:
        return [_parse_zero_shot_response(data)]
    if isinstance(data, list) and len(data) == expected_count:
        return [_parse_zero_shot_response(item) for item in data]
    raise RuntimeError(
        f"Unexpected zero-shot HF batch response; expected {expected_count} items, "
        f"got {type(data).__name__}"
    )


def _score_labels_hf_api(texts: List[str], labels: List[str]) -> List[Dict[str, float]]:
    token = _clean_token(os.getenv("ZERO_SHOT_HF_API_TOKEN") or os.getenv("HF_API_TOKEN"))
    if not token:
        raise RuntimeError("ZERO_SHOT_PROVIDER=hf_api requires HF_API_TOKEN")

    timeout = float(os.getenv("ZERO_SHOT_HF_API_TIMEOUT_SEC", os.getenv("HF_API_TIMEOUT_SEC", "45")))
    headers = {"Authorization": f"Bearer {token}"}
    url = _hf_zero_shot_url()
    max_retries, backoff = _retry_config()
    batch_size = _batch_size()

    out: List[Dict[str, float]] = []
    queue_timeout = float(os.getenv("ZERO_SHOT_HF_API_QUEUE_TIMEOUT_SEC", os.getenv("HF_PROVIDER_QUEUE_TIMEOUT_SEC", "30")))
    with provider_slot("hf_zero_shot", timeout_sec=queue_timeout) as queue, httpx.Client(timeout=timeout) as client:
        for batch in _chunks(texts, batch_size):
            started = time.monotonic()
            retry_count = 0
            payload = {
                "inputs": batch[0] if len(batch) == 1 else batch,
                "parameters": {
                    "candidate_labels": labels,
                    "multi_label": True,
                },
            }
            for attempt in range(max_retries + 1):
                try:
                    response = client.post(url, headers=headers, json=payload)
                    if _is_retryable_status(response.status_code) and attempt < max_retries:
                        if not queue.consume_retry():
                            response.raise_for_status()
                        retry_count += 1
                        time.sleep(_retry_delay(attempt + 1, backoff, response))
                        continue
                    response.raise_for_status()
                    out.extend(_parse_zero_shot_batch_response(response.json(), len(batch)))
                    record_provider_call(
                        "hf_zero_shot",
                        status="success",
                        latency_sec=time.monotonic() - started,
                        retries=retry_count,
                    )
                    break
                except (httpx.TimeoutException, httpx.NetworkError) as exc:
                    if attempt >= max_retries:
                        record_provider_call(
                            "hf_zero_shot",
                            status="failure",
                            latency_sec=time.monotonic() - started,
                            retries=retry_count,
                            error=str(exc),
                        )
                        raise RuntimeError(
                            "HF zero-shot request failed after retries; "
                            "the router may be unavailable or the model may still be cold."
                        ) from exc
                    if not queue.consume_retry():
                        record_provider_call(
                            "hf_zero_shot",
                            status="failure",
                            latency_sec=time.monotonic() - started,
                            retries=retry_count,
                            error="shared retry budget exhausted",
                        )
                        raise RuntimeError("HF zero-shot retry budget exhausted") from exc
                    retry_count += 1
                    time.sleep(_retry_delay(attempt + 1, backoff))
                except httpx.HTTPStatusError as exc:
                    record_provider_call(
                        "hf_zero_shot",
                        status="failure",
                        latency_sec=time.monotonic() - started,
                        retries=retry_count,
                        error=str(exc),
                    )
                    raise
    return out


def _score_labels_local(texts: List[str], labels: List[str]) -> List[Dict[str, float]]:
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


def score_labels(texts: List[str], labels: List[str]) -> List[Dict[str, float]]:
    """
    For each input text, return {label: probability}. Multi-label mode.
    """
    if not texts:
        return []
    if not labels:
        return [{} for _ in texts]

    provider = os.getenv("ZERO_SHOT_PROVIDER", "hf_api").strip().lower()
    if provider in {"hf", "hf_api", "huggingface"}:
        return _score_labels_hf_api(texts, labels)
    if provider == "auto":
        if _clean_token(os.getenv("ZERO_SHOT_HF_API_TOKEN") or os.getenv("HF_API_TOKEN")):
            return _score_labels_hf_api(texts, labels)
        return _score_labels_local(texts, labels)
    if provider == "local":
        return _score_labels_local(texts, labels)
    raise RuntimeError(f"Unsupported ZERO_SHOT_PROVIDER={provider!r}")
