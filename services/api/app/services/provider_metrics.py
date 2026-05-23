from __future__ import annotations

import threading
from collections import defaultdict
from dataclasses import dataclass, field
from time import monotonic
from typing import Any


@dataclass
class ProviderMetric:
    calls: int = 0
    successes: int = 0
    failures: int = 0
    retries: int = 0
    total_latency_sec: float = 0.0
    last_status: str = ""
    last_error: str = ""
    last_updated_monotonic: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        avg = self.total_latency_sec / self.calls if self.calls else 0.0
        return {
            "calls": self.calls,
            "successes": self.successes,
            "failures": self.failures,
            "retries": self.retries,
            "avg_latency_sec": round(avg, 4),
            "last_status": self.last_status,
            "last_error": self.last_error,
            "last_updated_monotonic": round(self.last_updated_monotonic, 4),
        }


_LOCK = threading.Lock()
_METRICS: dict[str, ProviderMetric] = defaultdict(ProviderMetric)


def record_provider_call(
    name: str,
    *,
    status: str,
    latency_sec: float = 0.0,
    retries: int = 0,
    error: str = "",
) -> None:
    with _LOCK:
        metric = _METRICS[name]
        metric.calls += 1
        metric.total_latency_sec += max(0.0, latency_sec)
        metric.retries += max(0, retries)
        metric.last_status = status
        metric.last_error = error[:300]
        metric.last_updated_monotonic = monotonic()
        if status == "success":
            metric.successes += 1
        else:
            metric.failures += 1


def snapshot_provider_metrics() -> dict[str, dict[str, Any]]:
    with _LOCK:
        return {name: metric.to_dict() for name, metric in sorted(_METRICS.items())}


def reset_provider_metrics() -> None:
    with _LOCK:
        _METRICS.clear()
