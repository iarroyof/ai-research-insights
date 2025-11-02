# services/api/app/clients/llm.py
import asyncio
from typing import AsyncGenerator
import httpx


class LLMClient:
    def __init__(self):
        # Lazy import: only load settings when the client is instantiated
        from app.config import settings
        self.base = settings.llm.base_url
        self.model = settings.llm.model
        self.max_tokens = settings.llm.max_output_tokens

    async def chat_stream(self, messages: list[dict]):
        url = f"{self.base}/chat/completions"
        async with httpx.AsyncClient(timeout=120) as client:
            async with client.stream(
                "POST",
                url,
                json={
                    "model": self.model,
                    "messages": messages,
                    "stream": True,
                    "max_tokens": self.max_tokens,
                },
            ) as r:
                async for line in r.aiter_lines():
                    if not line:
                        continue
                    if line.startswith("data: "):
                        yield line[6:]


# ----------------------------------------------------------------------
# Deterministic alias for summarize/conditioned.py
# ----------------------------------------------------------------------
async def stream_completion(prompt: str, **kwargs) -> AsyncGenerator[str, None]:
    """
    Deterministic backward-compatible stream completion for summarize/conditioned.py.
    It simply wraps LLMClient.chat_stream() with the user prompt.
    """
    client = LLMClient()
    async for chunk in client.chat_stream([{"role": "user", "content": prompt}]):
        yield chunk
