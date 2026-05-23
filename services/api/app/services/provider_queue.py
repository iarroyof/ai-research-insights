from __future__ import annotations

import asyncio
import os
import threading
import time
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from typing import Iterator

from app.services.provider_metrics import record_provider_call


@dataclass
class RetryBudget:
    limit: int
    window_sec: float
    used: int = 0
    window_started: float = field(default_factory=time.monotonic)

    def consume(self) -> bool:
        now = time.monotonic()
        if now - self.window_started >= self.window_sec:
            self.used = 0
            self.window_started = now
        if self.used >= self.limit:
            return False
        self.used += 1
        return True


class ProviderQueue:
    def __init__(self, name: str, max_concurrency: int, retry_budget: int, retry_window_sec: float):
        self.name = name
        self.semaphore = threading.BoundedSemaphore(max(1, max_concurrency))
        self.retry_budget = RetryBudget(max(0, retry_budget), max(1.0, retry_window_sec))
        self.lock = threading.Lock()

    def acquire(self, timeout_sec: float) -> float:
        started = time.monotonic()
        ok = self.semaphore.acquire(timeout=max(0.0, timeout_sec))
        waited = time.monotonic() - started
        if not ok:
            record_provider_call(self.name, status="queue_timeout", latency_sec=waited)
            raise TimeoutError(f"Provider queue timeout for {self.name}")
        if waited > 0.001:
            record_provider_call(self.name, status="queued", latency_sec=waited)
        return waited

    def release(self) -> None:
        self.semaphore.release()

    def consume_retry(self) -> bool:
        with self.lock:
            ok = self.retry_budget.consume()
        if not ok:
            record_provider_call(self.name, status="retry_budget_exhausted")
        return ok


_QUEUES: dict[str, ProviderQueue] = {}
_LOCK = threading.Lock()


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def provider_queue(name: str) -> ProviderQueue:
    env_prefix = name.upper().replace("-", "_")
    max_concurrency = _env_int(f"{env_prefix}_MAX_CONCURRENCY", _env_int("HF_PROVIDER_MAX_CONCURRENCY", 4))
    retry_budget = _env_int(f"{env_prefix}_RETRY_BUDGET", _env_int("HF_PROVIDER_RETRY_BUDGET", 60))
    retry_window = _env_float(f"{env_prefix}_RETRY_WINDOW_SEC", _env_float("HF_PROVIDER_RETRY_WINDOW_SEC", 60.0))
    key = f"{name}:{max_concurrency}:{retry_budget}:{retry_window}"
    with _LOCK:
        queue = _QUEUES.get(key)
        if queue is None:
            queue = ProviderQueue(name, max_concurrency, retry_budget, retry_window)
            _QUEUES[key] = queue
        return queue


@contextmanager
def provider_slot(name: str, timeout_sec: float = 30.0) -> Iterator[ProviderQueue]:
    queue = provider_queue(name)
    queue.acquire(timeout_sec)
    try:
        yield queue
    finally:
        queue.release()


@asynccontextmanager
async def async_provider_slot(name: str, timeout_sec: float = 30.0):
    queue = provider_queue(name)
    await asyncio.to_thread(queue.acquire, timeout_sec)
    try:
        yield queue
    finally:
        queue.release()


def reset_provider_queues() -> None:
    with _LOCK:
        _QUEUES.clear()
