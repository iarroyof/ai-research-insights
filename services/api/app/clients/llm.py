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

    def _provider_config(
        self,
        provider: str | None = None,
        *,
        model: str | None = None,
        api_format: str | None = None,
    ) -> dict[str, Any]:
        provider = provider or "local"
        if provider == "nvidia":
            api_key = self._clean_api_key(self.settings.nvidia_api_key or "")
            return {
                "base_url": self.settings.nvidia_base_url.rstrip("/"),
                "model": model or self.settings.context_manager_model or self.settings.nvidia_model,
                "api_key": api_key or "not-used",
                "max_tokens": self.settings.nvidia_max_tokens,
                "api_format": api_format or "openai_chat",
            }
        return {
            "base_url": self.base.rstrip("/"),
            "model": model or self.model,
            "api_key": "not-used",
            "max_tokens": self.max_tokens,
            "api_format": api_format or "openai_chat",
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

    def _headers_for_provider(self, provider: str | None, cfg: dict[str, Any]) -> dict[str, str]:
        if provider == "nvidia":
            return {"Authorization": f"Bearer {cfg['api_key']}"}
        return {}

    def _messages_payload(
        self,
        messages: list[dict],
        *,
        model: str,
        max_tokens: int,
        stream: bool,
        include_optional: bool = True,
    ) -> dict[str, Any]:
        system_parts: list[str] = []
        body_messages: list[dict] = []
        for message in messages:
            role = message.get("role")
            content = message.get("content", "")
            if role == "system":
                system_parts.append(str(content))
            elif role in {"user", "assistant"}:
                body_messages.append({"role": role, "content": content})
        payload: dict[str, Any] = {
            "model": model,
            "messages": body_messages,
            "stream": stream,
            "max_tokens": max_tokens,
        }
        if system_parts:
            payload["system"] = "\n\n".join(system_parts)
        if include_optional:
            payload["temperature"] = self.settings.temperature
            payload["top_p"] = self.settings.top_p
        return payload

    async def list_models(self, provider: str | None = None) -> list[dict[str, Any]]:
        provider = provider or self.settings.chat_provider or "local"
        cfg = self._provider_config(provider)
        url = f"{cfg['base_url']}/models"
        headers = self._headers_for_provider(provider, cfg)
        try:
            async with httpx.AsyncClient(timeout=20) as client:
                resp = await client.get(url, headers=headers)
                resp.raise_for_status()
                data = resp.json()
        except Exception:
            return []

        raw_items = data.get("data", data if isinstance(data, list) else [])
        models: list[dict[str, Any]] = []
        for item in raw_items:
            if isinstance(item, str):
                model_id = item
            elif isinstance(item, dict):
                model_id = item.get("id") or item.get("model") or item.get("name")
            else:
                continue
            if model_id:
                models.append(
                    {
                        "provider": provider,
                        "model": str(model_id),
                        "api_format": "openai_chat",
                        "available": True,
                        "source": "provider",
                    }
                )
        return models

    async def model_catalog(self) -> list[dict[str, Any]]:
        discovered = await self.list_models(self.settings.chat_provider or "nvidia")
        configured: list[dict[str, Any]] = []
        if self.settings.nvidia_model:
            configured.append(
                {
                    "provider": "nvidia",
                    "model": self.settings.nvidia_model,
                    "api_format": "openai_chat",
                    "available": any(m["model"] == self.settings.nvidia_model for m in discovered),
                    "source": "configured",
                }
            )
        if self.model:
            configured.append(
                {
                    "provider": "local",
                    "model": self.model,
                    "api_format": "openai_chat",
                    "available": False,
                    "source": "configured",
                }
            )

        presets = [
            "nvidia/llama-3.3-nemotron-super-49b-v1.5",
            "nvidia/llama-3.1-nemotron-70b-instruct",
            "nvidia/llama-3.1-nemotron-nano-vl-8b-v1",
            "meta/llama-3.1-8b-instruct",
        ]
        for model_id in presets:
            if not any(item["model"] == model_id for item in discovered + configured):
                configured.append(
                    {
                        "provider": "nvidia",
                        "model": model_id,
                        "api_format": "openai_chat",
                        "available": any(m["model"] == model_id for m in discovered),
                        "source": "preset",
                    }
                )

        merged: dict[tuple[str, str, str], dict[str, Any]] = {}
        for item in discovered + configured:
            key = (item["provider"], item["model"], item.get("api_format", "openai_chat"))
            current = merged.get(key, {})
            merged[key] = {**current, **item, "available": bool(current.get("available") or item.get("available"))}
        return sorted(
            merged.values(),
            key=lambda item: (
                not item.get("available", False),
                "nemotron" not in item.get("model", "").lower(),
                item.get("provider", ""),
                item.get("model", ""),
            ),
        )

    async def chat_stream(
        self,
        messages: list[dict],
        *,
        provider: str | None = None,
        model: str | None = None,
        api_format: str | None = None,
    ):
        provider = provider or self.settings.chat_provider or "local"
        cfg = self._provider_config(provider, model=model, api_format=api_format)
        api_format = cfg.get("api_format", "openai_chat")
        url = f"{cfg['base_url']}/chat/completions"
        headers = self._headers_for_provider(provider, cfg)
        payload = self._completion_payload(
            messages,
            model=cfg["model"],
            max_tokens=cfg["max_tokens"],
            stream=True,
            include_optional=True,
            provider=provider,
        )
        if api_format == "anthropic_messages":
            url = f"{cfg['base_url']}/messages"
            payload = self._messages_payload(
                messages,
                model=cfg["model"],
                max_tokens=cfg["max_tokens"],
                stream=True,
                include_optional=True,
            )
        async with httpx.AsyncClient(timeout=120) as client:
            stream = client.stream("POST", url, json=payload, headers=headers)
            async with stream as r:
                if r.status_code in (400, 422):
                    await r.aread()
                    if api_format == "anthropic_messages":
                        payload = self._messages_payload(
                            messages,
                            model=cfg["model"],
                            max_tokens=cfg["max_tokens"],
                            stream=True,
                            include_optional=False,
                        )
                    else:
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
                                yield self._normalize_stream_chunk(line[6:], api_format)
                    return
                r.raise_for_status()
                async for line in r.aiter_lines():
                    if not line:
                        continue
                    if line.startswith("data: "):
                        yield self._normalize_stream_chunk(line[6:], api_format)

    @staticmethod
    def _normalize_stream_chunk(chunk: str, api_format: str) -> str:
        if api_format != "anthropic_messages" or chunk == "[DONE]":
            return chunk
        try:
            import json

            data = json.loads(chunk)
            if data.get("type") == "content_block_delta":
                delta = data.get("delta", {})
                text = delta.get("text") or ""
                if text:
                    return json.dumps({"choices": [{"delta": {"content": text}}]})
        except Exception:
            pass
        return "{}"

    async def chat_once(
        self,
        messages: list[dict],
        *,
        provider: str | None = None,
        model: str | None = None,
        api_format: str | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """
        Non-streaming completion for policy/reflection calls.

        NVIDIA NIM/build endpoints are OpenAI-compatible, but model-specific
        schemas differ. We first send configured optional controls, then retry
        without them when a model rejects unexpected arguments.
        """
        provider = provider or self.settings.context_manager_provider or "local"
        cfg = self._provider_config(provider, model=model, api_format=api_format)
        api_format = cfg.get("api_format", "openai_chat")
        url = f"{cfg['base_url']}/chat/completions"
        headers = self._headers_for_provider(provider, cfg)
        payload = self._completion_payload(
            messages,
            model=cfg["model"],
            max_tokens=max_tokens or cfg["max_tokens"],
            stream=False,
            include_optional=True,
            provider=provider,
        )
        if api_format == "anthropic_messages":
            url = f"{cfg['base_url']}/messages"
            payload = self._messages_payload(
                messages,
                model=cfg["model"],
                max_tokens=max_tokens or cfg["max_tokens"],
                stream=False,
                include_optional=True,
            )

        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(url, json=payload, headers=headers)
            if resp.status_code in (400, 422):
                if api_format == "anthropic_messages":
                    payload = self._messages_payload(
                        messages,
                        model=cfg["model"],
                        max_tokens=max_tokens or cfg["max_tokens"],
                        stream=False,
                        include_optional=False,
                    )
                else:
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
            if api_format == "anthropic_messages":
                parts = data.get("content", [])
                return "".join(part.get("text", "") for part in parts if isinstance(part, dict))
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
