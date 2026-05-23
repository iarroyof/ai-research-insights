# services/api/app/clients/llm.py
from typing import Any, AsyncGenerator
import re
import httpx

class LLMClient:
    def __init__(self):
        # Lazy import: only load settings when the client is instantiated
        from app.config import settings
        self.settings = settings.llm
        self.base = settings.llm.base_url
        self.model = settings.llm.model
        self.max_tokens = settings.llm.max_output_tokens

    def _provider_config(self, provider: str | None = None) -> dict[str, Any]:
        provider = provider or "local"
        if provider == "nvidia":
            api_key = self._clean_api_key(self.settings.nvidia_api_key or "")
            return {
                "base_url": self.settings.nvidia_base_url.rstrip("/"),
                "model": self.settings.context_manager_model or self.settings.nvidia_model,
                "api_key": api_key or "not-used",
                "max_tokens": self.settings.nvidia_max_tokens,
            }
        return {
            "base_url": self.base.rstrip("/"),
            "model": self.model,
            "api_key": "not-used",
            "max_tokens": self.max_tokens,
        }

    @staticmethod
    def _clean_api_key(value: str) -> str:
        match = re.search(r"nvapi-[^\s]+", value or "")
        return match.group(0) if match else (value or "").strip()

    def _completion_payload(
        self,
        messages: list[dict],
        *,
        model: str,
        max_tokens: int,
        stream: bool,
        include_optional: bool = True,
        provider: str | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": stream,
            "max_tokens": max_tokens,
        }
        if include_optional:
            payload["temperature"] = self.settings.temperature
            payload["top_p"] = self.settings.top_p
            if provider == "nvidia":
                if self.settings.nvidia_reasoning_effort:
                    payload["reasoning_effort"] = self.settings.nvidia_reasoning_effort
                if self.settings.nvidia_enable_thinking is not None:
                    payload["extra_body"] = {
                        "chat_template_kwargs": {
                            "enable_thinking": self.settings.nvidia_enable_thinking,
                        }
                    }
        return payload

    async def chat_stream(self, messages: list[dict]):
        provider = self.settings.chat_provider or "local"
        cfg = self._provider_config(provider)
        url = f"{cfg['base_url']}/chat/completions"
        headers = {"Authorization": f"Bearer {cfg['api_key']}"} if provider == "nvidia" else {}
        payload = self._completion_payload(
            messages,
            model=cfg["model"],
            max_tokens=cfg["max_tokens"],
            stream=True,
            include_optional=True,
            provider=provider,
        )
        async with httpx.AsyncClient(timeout=120) as client:
            stream = client.stream("POST", url, json=payload, headers=headers)
            async with stream as r:
                if r.status_code in (400, 422):
                    await r.aread()
                    payload = self._completion_payload(
                        messages,
                        model=cfg["model"],
                        max_tokens=cfg["max_tokens"],
                        stream=True,
                        include_optional=False,
                        provider=provider,
                    )
                    stream = client.stream("POST", url, json=payload, headers=headers)
                    async with stream as retry:
                        retry.raise_for_status()
                        async for line in retry.aiter_lines():
                            if not line:
                                continue
                            if line.startswith("data: "):
                                yield line[6:]
                    return
                r.raise_for_status()
                async for line in r.aiter_lines():
                    if not line:
                        continue
                    if line.startswith("data: "):
                        yield line[6:]

    async def chat_once(
        self,
        messages: list[dict],
        *,
        provider: str | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """
        Non-streaming completion for policy/reflection calls.

        NVIDIA NIM/build endpoints are OpenAI-compatible, but model-specific
        schemas differ. We first send configured optional controls, then retry
        without them when a model rejects unexpected arguments.
        """
        provider = provider or self.settings.context_manager_provider or "local"
        cfg = self._provider_config(provider)
        url = f"{cfg['base_url']}/chat/completions"
        headers = {"Authorization": f"Bearer {cfg['api_key']}"}
        payload = self._completion_payload(
            messages,
            model=cfg["model"],
            max_tokens=max_tokens or cfg["max_tokens"],
            stream=False,
            include_optional=True,
            provider=provider,
        )

        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(url, json=payload, headers=headers)
            if resp.status_code in (400, 422):
                payload = self._completion_payload(
                    messages,
                    model=cfg["model"],
                    max_tokens=max_tokens or cfg["max_tokens"],
                    stream=False,
                    include_optional=False,
                    provider=provider,
                )
                resp = await client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()

        try:
            message = data["choices"][0].get("message") or {}
            return message.get("content") or ""
        except Exception:
            return ""

# ----------------------------------------------------------------------
# Deterministic alias for summarize/conditioned.py
# ----------------------------------------------------------------------
async def stream_completion(prompt: str, **kwargs) -> AsyncGenerator[str, None]:
    """
    Deterministic backward-compatible stream completion for summarize/conditioned.py.
    It simply wraps LLMClient.chat_stream() with the user prompt.
    
    Parses OpenAI-format streaming chunks and yields only the text content.
    """
    import json
    
    client = LLMClient()
    async for chunk in client.chat_stream([{"role": "user", "content": prompt}]):
        # Skip [DONE] signal
        if chunk == "[DONE]":
            break
        
        try:
            # Parse the JSON chunk
            data = json.loads(chunk)
            
            # Extract text content from OpenAI format
            if "choices" in data and len(data["choices"]) > 0:
                delta = data["choices"][0].get("delta", {})
                content = delta.get("content", "")
                if content:
                    yield content
        except json.JSONDecodeError:
            # Skip malformed chunks
            continue
