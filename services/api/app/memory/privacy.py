from __future__ import annotations

import re
from typing import Dict


REDACTIONS: list[tuple[str, str]] = [
    (r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", "[email]"),
    (r"\b(?:\+?\d[\d\s().-]{7,}\d)\b", "[phone]"),
    (r"https?://\S+|www\.\S+", "[url]"),
    (r"\b\d{1,3}(?:\.\d{1,3}){3}\b", "[ip]"),
    (r"\b(?:api[_-]?key|token|secret|password|passwd|pwd)\s*[:=]\s*\S+", "[secret]"),
    (r"\b[A-Za-z]:\\[^\s]+|/(?:home|Users|mnt|var|etc)/[^\s]+", "[path]"),
]


def redact_query(text: str, *, max_len: int = 240) -> Dict[str, object]:
    redacted = text or ""
    changed = False
    for pattern, repl in REDACTIONS:
        updated = re.sub(pattern, repl, redacted, flags=re.IGNORECASE)
        changed = changed or updated != redacted
        redacted = updated
    redacted = " ".join(redacted.split())[:max_len]
    return {
        "query": redacted,
        "redacted": changed,
        "safe_for_web": bool(redacted and "[secret]" not in redacted),
    }

