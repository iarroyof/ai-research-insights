from __future__ import annotations
import re
from typing import List, Dict, Tuple

_TRIPLET_SPLIT = re.compile(r"<triplet>", re.IGNORECASE)
TAG = r"\s*<\s*{tag}\s*>\s*"

_PATTERNS: List[Tuple[str, re.Pattern]] = [
    ("std_all",
     re.compile(
        TAG.format(tag="subj") + r"(?P<subj>.*?)" +
        TAG.format(tag="obj")  + r"(?P<obj>.*?)"  +
        TAG.format(tag="rel")  + r"(?P<rel>.+?)\s*$",
        re.IGNORECASE | re.DOTALL
     )),
    ("std_norel",
     re.compile(
        TAG.format(tag="subj") + r"(?P<subj>.*?)" +
        TAG.format(tag="obj")  + r"(?P<obj>.+?)\s*$",
        re.IGNORECASE | re.DOTALL
     )),
    ("label_after_no_rel",
     re.compile(
        r"^(?P<subj>.*?)" + TAG.format(tag="subj") +
        r"(?P<obj>.*?)"  + TAG.format(tag="obj")  +
        r"(?P<rel>.+?)\s*$",
        re.IGNORECASE | re.DOTALL
     )),
]

def _clean(s: str) -> str:
    s = s.replace("</s>", "").replace("<s>", "").strip()
    return re.sub(r"\s+", " ", s)

def parse_generation_to_tuples(generation: str) -> List[Dict[str, str]]:
    if not generation:
        return []
    triples: List[Dict[str, str]] = []
    for chunk in _TRIPLET_SPLIT.split(generation):
        chunk = chunk.strip()
        if not chunk:
            continue
        for _, pat in _PATTERNS:
            m = pat.search(chunk)
            if m:
                subj = _clean(m.group("subj"))
                obj  = _clean(m.group("obj"))
                rel  = _clean(m.group("rel")) if "rel" in m.groupdict() else ""
                if subj and obj:
                    triples.append({"subject": subj, "relation": rel or "related_to", "object": obj})
                break
    return triples
