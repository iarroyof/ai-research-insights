# services/api/app/services/triples_csv.py
from __future__ import annotations
from typing import List, Dict, Any, Callable, Iterable

HEADER = [
    "article_id","sentence_text","subject","relation","object",
    "subject_probably_EBio","subject_probably_NGen","subject_probably_otro",
    "object_probably_EBio","object_probably_NGen","object_probably_otro",
]

def to_rows(
    article_id: str,
    sentence_text: str,
    triples: Iterable[Dict[str, Any]],
    probs_fn: Callable[[List[str]], List[Dict[str,float]]] | None = None,
    labels: List[str] = ("EBio","NGen","otro"),
) -> List[List[str]]:
    rows: List[List[str]] = []
    if probs_fn:
        subj_texts = [t.get("subject","") for t in triples]
        obj_texts  = [t.get("object","")  for t in triples]
        subj_probs = probs_fn(subj_texts)
        obj_probs  = probs_fn(obj_texts)
    else:
        subj_probs = [{} for _ in triples]
        obj_probs  = [{} for _ in triples]

    for (t, sp, op) in zip(triples, subj_probs, obj_probs):
        s = t.get("subject",""); r = t.get("relation",""); o = t.get("object","")
        row = [
            article_id, sentence_text, s, r, o,
            f"{sp.get(labels[0], '') or '':}".rstrip(),
            f"{sp.get(labels[1], '') or '':}".rstrip(),
            f"{sp.get(labels[2], '') or '':}".rstrip(),
            f"{op.get(labels[0], '') or '':}".rstrip(),
            f"{op.get(labels[1], '') or '':}".rstrip(),
            f"{op.get(labels[2], '') or '':}".rstrip(),
        ]
        rows.append(row)
    return rows
