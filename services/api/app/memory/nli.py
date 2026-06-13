from __future__ import annotations

import json
import os
import re
import time
import asyncio
from typing import Any, Dict, List

import httpx

from app.clients.llm import LLMClient
from app.config import settings
from app.memory.rewards import lexical_overlap, terms
from app.services.provider_metrics import record_provider_call
from app.services.provider_queue import async_provider_slot


def triple_claim(triple: Dict[str, Any]) -> str:
    subject = triple.get("subject") or ""
    relation = triple.get("relation") or triple.get("predicate") or ""
    obj = triple.get("object") or ""
    return " ".join(x for x in [subject, relation, obj] if x).strip()


def origin_sentence(triple: Dict[str, Any]) -> str:
    return (
        triple.get("origin_sentence")
        or triple.get("sentence_text")
        or triple.get("text")
        or triple.get("source_sentence")
        or ""
    )


def _entity_overlap(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    aa = set(terms(f"{a.get('subject', '')} {a.get('object', '')}"))
    bb = set(terms(f"{b.get('subject', '')} {b.get('object', '')}"))
    if not aa or not bb:
        return lexical_overlap(triple_claim(a), triple_claim(b))
    return len(aa & bb) / max(1, len(aa | bb))


def _heuristic_nli(premise: str, hypothesis: str) -> Dict[str, float | str]:
    overlap = lexical_overlap(premise, hypothesis)
    negators = {"no", "not", "never", "without", "absent", "negative", "inhibits", "fails"}
    p_neg = bool(set(terms(premise)) & negators)
    h_neg = bool(set(terms(hypothesis)) & negators)
    contradiction = min(1.0, 0.25 + overlap) if overlap > 0.15 and p_neg != h_neg else 0.0
    entailment = max(0.0, min(1.0, overlap - contradiction * 0.5))
    neutral = max(0.0, 1.0 - entailment - contradiction)
    label = "entailment" if entailment >= max(contradiction, neutral) else "contradiction" if contradiction >= neutral else "neutral"
    return {
        "label": label,
        "entailment": round(entailment, 4),
        "contradiction": round(contradiction, 4),
        "neutral": round(neutral, 4),
        "provider": "heuristic",
    }


async def _http_nli(premise: str, hypothesis: str) -> Dict[str, Any]:
    if not settings.memory.nli_endpoint:
        return _heuristic_nli(premise, hypothesis)
    payload = {
        "premise": premise,
        "hypothesis": hypothesis,
        "model": settings.memory.nli_model,
    }
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(settings.memory.nli_endpoint, json=payload)
        resp.raise_for_status()
        data = resp.json()
    label = str(data.get("label") or data.get("prediction") or "neutral").lower()
    scores = data.get("scores") or data.get("probabilities") or data
    return {
        "label": label,
        "entailment": float(scores.get("entailment", 0.0) or 0.0),
        "contradiction": float(scores.get("contradiction", 0.0) or 0.0),
        "neutral": float(scores.get("neutral", 0.0) or 0.0),
        "provider": "http",
        "model": settings.memory.nli_model,
    }


def _clean_hf_token(value: str | None) -> str:
    if not value:
        return ""
    match = re.search(r"hf_[A-Za-z0-9]+", value)
    return match.group(0) if match else value.strip()


def _labels_from_scores(items: Any) -> Dict[str, float]:
    scores = {"entailment": 0.0, "contradiction": 0.0, "neutral": 0.0}
    if isinstance(items, list):
        # HF text-classification may return [[{label, score}, ...]] or [{...}].
        if items and isinstance(items[0], list):
            items = items[0]
        for item in items:
            if not isinstance(item, dict):
                continue
            label = str(item.get("label") or "").lower()
            score = float(item.get("score", 0.0) or 0.0)
            if "entail" in label or label in {"label_1", "1"}:
                scores["entailment"] = max(scores["entailment"], score)
            elif "contrad" in label or label in {"label_0", "0"}:
                scores["contradiction"] = max(scores["contradiction"], score)
            elif "neutral" in label or label in {"label_2", "2"}:
                scores["neutral"] = max(scores["neutral"], score)
    return scores


def _labels_list_from_response(data: Any, expected: int) -> list[Dict[str, float]]:
    """
    Normalize HF text-classification responses for single or batched inputs.

    The router commonly returns:
      - single input: [{label, score}, ...] or [[{label, score}, ...]]
      - batched input: [[{label, score}, ...], [{label, score}, ...]]
    """
    expected = max(1, expected)
    if isinstance(data, list) and data and all(isinstance(item, list) for item in data):
        out = [_labels_from_scores(item) for item in data[:expected]]
    elif expected == 1:
        out = [_labels_from_scores(data)]
    else:
        out = [_labels_from_scores(data)]
    while len(out) < expected:
        out.append({"entailment": 0.0, "contradiction": 0.0, "neutral": 1.0})
    return out[:expected]


def _hf_retry_config() -> tuple[int, float]:
    retries = int(os.getenv("NLI_HF_API_MAX_RETRIES", os.getenv("HF_API_MAX_RETRIES", "2")))
    backoff = float(os.getenv("NLI_HF_API_RETRY_BACKOFF_SEC", os.getenv("HF_API_RETRY_BACKOFF_SEC", "2.0")))
    return max(0, retries), max(0.0, backoff)


def _hf_batch_size() -> int:
    raw = os.getenv("NLI_HF_API_BATCH_SIZE")
    if raw is None:
        raw = str(getattr(settings.memory, "nli_hf_api_batch_size", 8))
    try:
        return max(1, int(raw))
    except ValueError:
        return 8


def _nli_panel_models() -> list[str]:
    configured = str(getattr(settings.memory, "nli_panel_models", "") or "")
    items = [item.strip() for item in configured.split(",") if item.strip()]
    primary = str(getattr(settings.memory, "nli_model", "") or "").strip()
    if primary:
        items.insert(0, primary)
    return list(dict.fromkeys(items))


def _retryable_status(status_code: int) -> bool:
    return status_code in {408, 409, 425, 429, 500, 502, 503, 504}


def _retry_delay(attempt: int, backoff: float, response: httpx.Response | None = None) -> float:
    if response is not None:
        retry_after = response.headers.get("retry-after")
        if retry_after:
            try:
                return max(0.0, float(retry_after))
            except ValueError:
                pass
    return backoff * (2 ** max(0, attempt - 1))


async def _hf_api_nli_batch_for_model(pairs: list[tuple[str, str]], model: str) -> list[Dict[str, Any]]:
    if not pairs:
        return []
    token = _clean_hf_token(settings.memory.hf_api_token)
    if not token:
        print("[WARN] HF_API_TOKEN missing, using heuristic NLI fallback")
        return [_heuristic_nli(premise, hypothesis) for premise, hypothesis in pairs]

    base = settings.memory.hf_api_base_url.rstrip("/")
    url = f"{base}/{model}"
    inputs = [f"{hypothesis} [SEP] {premise}" for premise, hypothesis in pairs]
    payload = {
        "inputs": inputs[0] if len(inputs) == 1 else inputs,
        "parameters": {
            "return_all_scores": True,
        },
        "options": {
            "wait_for_model": True,
        },
    }
    headers = {"Authorization": f"Bearer {token}"}
    max_retries, backoff = _hf_retry_config()
    queue_timeout = float(os.getenv("NLI_HF_API_QUEUE_TIMEOUT_SEC", os.getenv("HF_PROVIDER_QUEUE_TIMEOUT_SEC", "30")))

    started = time.monotonic()
    retry_count = 0
    try:
        async with async_provider_slot("hf_biomed_nli", timeout_sec=queue_timeout) as queue, httpx.AsyncClient(timeout=settings.memory.hf_api_timeout_sec) as client:
            for attempt in range(max_retries + 1):
                try:
                    resp = await client.post(url, json=payload, headers=headers)
                    if resp.status_code in (400, 422) and len(pairs) == 1:
                        # Fallback for older serverless endpoints that treat a two-item list
                        # as a sequence pair instead of a batch.
                        fallback_payload = dict(payload)
                        premise, hypothesis = pairs[0]
                        fallback_payload["inputs"] = [hypothesis, premise]
                        resp = await client.post(url, json=fallback_payload, headers=headers)
                    if _retryable_status(resp.status_code) and attempt < max_retries:
                        if not queue.consume_retry():
                            resp.raise_for_status()
                        retry_count += 1
                        await asyncio.sleep(_retry_delay(attempt + 1, backoff, resp))
                        continue
                    resp.raise_for_status()
                    data = resp.json()
                    break
                except (httpx.TimeoutException, httpx.NetworkError):
                    if attempt >= max_retries:
                        raise
                    if not queue.consume_retry():
                        raise RuntimeError("HF BioNLI retry budget exhausted")
                    retry_count += 1
                    await asyncio.sleep(_retry_delay(attempt + 1, backoff))
            else:
                data = []
    except Exception as exc:
        record_provider_call(
            "hf_biomed_nli",
            status="failure",
            latency_sec=time.monotonic() - started,
            retries=retry_count,
            error=str(exc),
        )
        raise

    score_items = _labels_list_from_response(data, len(pairs))
    record_provider_call(
        "hf_biomed_nli",
        status="success",
        latency_sec=time.monotonic() - started,
        retries=retry_count,
    )
    out: list[dict[str, Any]] = []
    for scores in score_items:
        label = max(scores, key=lambda k: scores[k])
        out.append(
            {
                "label": label,
                **scores,
                "provider": "hf_api",
                "model": model,
            }
        )
    return out


async def _hf_api_nli_batch(pairs: list[tuple[str, str]]) -> list[Dict[str, Any]]:
    return await _hf_api_nli_batch_for_model(pairs, settings.memory.nli_model)


def _aggregate_nli_panel(per_pair: list[list[Dict[str, Any]]]) -> list[Dict[str, Any]]:
    out: list[Dict[str, Any]] = []
    for panel in per_pair:
        successes = [
            item for item in panel
            if not item.get("error") and item.get("provider") != "heuristic_panel_fallback"
        ]
        usable = successes or [item for item in panel if not item.get("error")]
        if not usable:
            usable = [_heuristic_nli("", "")]
        entailments = [float(item.get("entailment", 0.0) or 0.0) for item in usable]
        contradictions = [float(item.get("contradiction", 0.0) or 0.0) for item in usable]
        neutrals = [float(item.get("neutral", 0.0) or 0.0) for item in usable]
        labels = [str(item.get("label") or "neutral") for item in usable]
        scores = {
            "entailment": round(sum(entailments) / max(1, len(entailments)), 4),
            "contradiction": round(max(contradictions) if contradictions else 0.0, 4),
            "neutral": round(sum(neutrals) / max(1, len(neutrals)), 4),
        }
        label = max(scores, key=lambda key: scores[key])
        agreement = labels.count(label) / max(1, len(labels))
        out.append(
            {
                "label": label,
                **scores,
                "provider": "nli_panel",
                "model": "panel",
                "panel_success_count": len(successes),
                "panel_size": len(panel),
                "panel_agreement": round(agreement, 4),
                "panel": panel,
            }
        )
    return out


async def _hf_api_nli_panel_batch(pairs: list[tuple[str, str]]) -> list[Dict[str, Any]]:
    if not pairs:
        return []
    models = _nli_panel_models()
    if not models:
        return await _hf_api_nli_batch(pairs)
    per_pair: list[list[Dict[str, Any]]] = [[] for _ in pairs]
    for model in models:
        try:
            results = await _hf_api_nli_batch_for_model(pairs, model)
        except Exception as exc:
            results = [
                {
                    "label": "neutral",
                    "entailment": 0.0,
                    "contradiction": 0.0,
                    "neutral": 1.0,
                    "provider": "hf_api",
                    "model": model,
                    "error": exc.__class__.__name__,
                }
                for _ in pairs
            ]
        for idx, result in enumerate(results[: len(pairs)]):
            per_pair[idx].append(result)
    min_successes = max(1, int(getattr(settings.memory, "nli_panel_min_successes", 1) or 1))
    for idx, panel in enumerate(per_pair):
        successes = [item for item in panel if not item.get("error")]
        if len(successes) < min_successes:
            premise, hypothesis = pairs[idx]
            fallback = _heuristic_nli(premise, hypothesis)
            fallback["provider"] = "heuristic_panel_fallback"
            fallback["model"] = "heuristic"
            panel.append(fallback)
    return _aggregate_nli_panel(per_pair)


async def _hf_api_nli(premise: str, hypothesis: str) -> Dict[str, Any]:
    return (await _hf_api_nli_batch([(premise, hypothesis)]))[0]


async def _llm_nli(premise: str, hypothesis: str) -> Dict[str, Any]:
    messages = [
        {
            "role": "system",
            "content": (
                "Classify biomedical natural-language inference. Return compact JSON only with keys "
                "label, entailment, contradiction, neutral. Label must be entailment, contradiction, or neutral."
            ),
        },
        {
            "role": "user",
            "content": f"Premise/source sentence:\n{premise[:1200]}\n\nHypothesis/claim:\n{hypothesis[:500]}",
        },
    ]
    try:
        text = await LLMClient().chat_once(
            messages,
            provider=settings.llm.context_manager_provider,
            max_tokens=120,
        )
        start = text.find("{")
        end = text.rfind("}")
        data = json.loads(text[start : end + 1]) if start >= 0 and end > start else {}
        label = str(data.get("label") or "neutral").lower()
        return {
            "label": label,
            "entailment": float(data.get("entailment", 0.0) or 0.0),
            "contradiction": float(data.get("contradiction", 0.0) or 0.0),
            "neutral": float(data.get("neutral", 0.0) or 0.0),
            "provider": "llm",
        }
    except Exception as e:
        print(f"[WARN] LLM NLI failed, using heuristic fallback: {e}")
        return _heuristic_nli(premise, hypothesis)


async def classify_nli(premise: str, hypothesis: str) -> Dict[str, Any]:
    provider = settings.memory.nli_provider
    if provider == "hf_api":
        if bool(getattr(settings.memory, "nli_panel_enabled", False)):
            return (await _hf_api_nli_panel_batch([(premise, hypothesis)]))[0]
        return await _hf_api_nli(premise, hypothesis)
    if provider == "http":
        return await _http_nli(premise, hypothesis)
    if provider == "llm":
        return await _llm_nli(premise, hypothesis)
    return _heuristic_nli(premise, hypothesis)


async def classify_nli_batch(pairs: list[tuple[str, str]]) -> list[Dict[str, Any]]:
    provider = settings.memory.nli_provider
    if not pairs:
        return []
    if provider == "hf_api":
        batch_size = _hf_batch_size()
        out: list[dict[str, Any]] = []
        for start in range(0, len(pairs), batch_size):
            batch = pairs[start : start + batch_size]
            if bool(getattr(settings.memory, "nli_panel_enabled", False)):
                out.extend(await _hf_api_nli_panel_batch(batch))
            else:
                out.extend(await _hf_api_nli_batch(batch))
        return out
    return [await classify_nli(premise, hypothesis) for premise, hypothesis in pairs]


async def score_answer_triples(
    answer_triples: List[Dict[str, Any]],
    retrieved_triplets: List[Dict[str, Any]],
    *,
    max_pairs: int = 8,
) -> List[Dict[str, Any]]:
    if not settings.memory.nli_enabled or not answer_triples or not retrieved_triplets:
        return []

    pairs: list[tuple[float, Dict[str, Any], Dict[str, Any]]] = []
    for claim in answer_triples:
        claim_text = triple_claim(claim)
        if not claim_text:
            continue
        for evidence in retrieved_triplets:
            premise = origin_sentence(evidence)
            if not premise:
                continue
            score = _entity_overlap(claim, evidence)
            if score <= 0:
                continue
            pairs.append((score, claim, evidence))

    pairs.sort(key=lambda x: x[0], reverse=True)
    out: list[dict] = []
    seen: set[tuple[str, str]] = set()
    selected_pairs: list[tuple[Dict[str, Any], Dict[str, Any], str, str]] = []
    for _, claim, evidence in pairs:
        claim_text = triple_claim(claim)
        premise = origin_sentence(evidence)
        key = (claim_text.lower(), premise[:180].lower())
        if key in seen:
            continue
        seen.add(key)
        selected_pairs.append((claim, evidence, premise, claim_text))
        if len(selected_pairs) >= max_pairs:
            break

    nli_results = await classify_nli_batch([(premise, claim_text) for _, _, premise, claim_text in selected_pairs])
    for (claim, evidence, premise, claim_text), nli in zip(selected_pairs, nli_results):
        out.append(
            {
                "claim": claim_text,
                "premise": premise,
                "evidence_triple": {
                    "subject": evidence.get("subject"),
                    "relation": evidence.get("relation") or evidence.get("predicate"),
                    "object": evidence.get("object"),
                    "origin_sentence": premise,
                    "paper_id": evidence.get("paper_id") or evidence.get("article_id"),
                    "confidence": evidence.get("confidence"),
                },
                **nli,
            }
        )
    return out
