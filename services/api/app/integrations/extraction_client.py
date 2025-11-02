# services/api/app/integrations/extraction_client.py
from __future__ import annotations

from typing import List, Dict, Any, Iterable, Tuple
import httpx

from app.core.settings import settings
from app.services.zero_shot import score_labels  # used by extract_rows_for_csv
from app.services.triples_csv import to_rows, HEADER  # used by extract_rows_for_csv


def _base_url() -> str:
    """
    Resolve the active extractor base URL from config.
    Supports multiple providers (e.g., 'rebel', 'stanford').
    """
    prov = settings.extraction.provider
    providers = settings.extraction.providers or {}
    if prov not in providers:
        raise RuntimeError(f"Extractor provider '{prov}' not configured")
    return providers[prov].base_url.rstrip("/")


def _default_timeout() -> int:
    """
    REBEL may need a longer timeout on first warm-up.
    """
    prov = settings.extraction.provider
    return 300 if prov == "rebel" else 60


def _ensure_text_key(triple: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize keys to {subject, relation, object, confidence?}
    Some adapters might use slightly different key names.
    """
    out = {
        "subject": triple.get("subject") or triple.get("subj") or triple.get("s") or "",
        "relation": triple.get("relation") or triple.get("rel") or triple.get("r"),
        "object": triple.get("object") or triple.get("obj") or triple.get("o") or "",
    }
    # optional confidence
    if "confidence" in triple:
        out["confidence"] = triple["confidence"]
    return out


def _normalize_rebel(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Normalize REBEL response to a flat list of triples.
    Expected shape (per-sentence):
      {
        "count": int,
        "elapsed_sec": float,
        "results": [
          {"text": "...", "tuples": [ either dict triples or 3-item lists ], ...},
          ...
        ]
      }
    """
    triples: List[Dict[str, Any]] = []
    for item in payload.get("results", []) or []:
        text = item.get("text", "")
        tuples = item.get("tuples", []) or []
        for t in tuples:
            if isinstance(t, dict):
                norm = _ensure_text_key(t)
            elif isinstance(t, (list, tuple)) and len(t) >= 3:
                # [subject, relation, object, (optional confidence)]
                norm = {
                    "subject": t[0],
                    "relation": t[1],
                    "object": t[2],
                }
                if len(t) >= 4:
                    norm["confidence"] = t[3]
            else:
                # Unknown element; skip
                continue
            norm["sentence_text"] = text
            triples.append(norm)
    return triples


def _normalize_stanford(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Normalize CoreNLP adapter response to a flat list of triples.
    Accepts both:
      { "triples": [ {subject,relation,object,confidence?}, ... ] }
      or per-sentence shape:
      { "results": [ {"text": "...", "triples":[...]}, ... ] }
    """
    triples: List[Dict[str, Any]] = []
    if "results" in payload:
        for item in payload.get("results", []) or []:
            text = item.get("text", "")
            for t in item.get("triples", []) or []:
                norm = _ensure_text_key(t)
                norm["sentence_text"] = text
                triples.append(norm)
        return triples

    for t in payload.get("triples", []) or []:
        norm = _ensure_text_key(t)
        # sentence_text may be absent; keep it missing rather than guessing
        triples.append(norm)
    return triples


async def extract_triples(
    sentences: List[str],
    *,
    timeout_sec: int | None = None,
    num_extractions: int | None = None,
) -> Dict[str, Any]:
    """
    Call the active extractor adapter with a batch of sentences and return
    a normalized triple list.

    Returns:
      {
        "triples": [ {subject, relation, object, confidence?, sentence_text?}, ... ],
        "count": <int>,
        "provider": <str>
      }
    """
    if not sentences:
        return {"triples": [], "count": 0, "provider": settings.extraction.provider}

    url = f"{_base_url()}/extract"
    payload: Dict[str, Any] = {"sentences": sentences}
    if num_extractions is not None:
        payload["num_extractions"] = num_extractions

    timeout = timeout_sec or _default_timeout()
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()

    prov = settings.extraction.provider
    if prov == "rebel":
        triples = _normalize_rebel(data)
    else:
        # default normalization (stanford)
        triples = _normalize_stanford(data)

    return {"triples": triples, "count": len(triples), "provider": prov}


async def extract_rows_for_csv(
    article_id: str,
    sentence_texts: List[str],
    labels: List[str] | None = None,
    add_probs: bool = True,
) -> Tuple[List[List[str]], List[str]]:
    """
    Helper to go straight from sentences → normalized CSV rows (your schema),
    optionally adding BART-MNLI probabilities for subject/object.

    Returns:
      (rows, header)
    """
    result = await extract_triples(sentence_texts)
    triples: List[Dict[str, Any]] = result["triples"]
    labs = labels or settings.classification.labels

    # Group triples by sentence_text if present; otherwise, we
    # conservatively assign no sentence grouping (emit what we have).
    by_sent: Dict[str, List[Dict[str, Any]]] = {}
    have_sentence_text = any("sentence_text" in t for t in triples)
    if have_sentence_text:
        for t in triples:
            by_sent.setdefault(t.get("sentence_text", ""), []).append(t)
    else:
        # No per-sentence tag in triples: treat all as belonging to each given sentence
        # (or emit once with the first sentence). Here we assign to the matching input
        # if extractor preserved order; otherwise fall back to a single group.
        if sentence_texts:
            by_sent[sentence_texts[0]] = triples[:]
        else:
            by_sent[""] = triples[:]

    rows: List[List[str]] = []
    if add_probs:
        def probs_fn(texts: List[str]) -> List[Dict[str, float]]:
            return score_labels(texts, labs)

    for sent in sentence_texts or list(by_sent.keys()):
        group = by_sent.get(sent, [])
        if not group:
            continue
        if add_probs:
            rows.extend(
                to_rows(
                    article_id,
                    sent,
                    group,
                    probs_fn=probs_fn,
                    labels=labs,
                )
            )
        else:
            rows.extend(
                to_rows(
                    article_id,
                    sent,
                    group,
                    probs_fn=None,
                    labels=labs,
                )
            )

    return rows, HEADER

