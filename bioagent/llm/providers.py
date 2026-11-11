"""LLM providers over plain HTTP (no vendor SDKs, so the air-gapped install is small).

- anthropic          Anthropic Messages API
- openai             OpenAI Chat Completions
- openai_compatible  vLLM, DeepSeek, Together, LM Studio, Ollama's /v1 ...
- ollama             Ollama native /api/chat with JSON-schema constrained output
- mock               deterministic offline provider for tests and demos
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any

import httpx

from ..config import ModelSpec


@dataclass
class LLMResponse:
    text: str
    model: str
    tokens_in: int = 0
    tokens_out: int = 0
    latency_s: float = 0.0


class ProviderError(RuntimeError):
    pass


class Provider:
    def __init__(self, spec: ModelSpec):
        self.spec = spec

    @property
    def name(self) -> str:
        return f"{self.spec.provider}:{self.spec.model}"

    def complete(self, system: str, messages: list[dict], json_schema: dict | None, ctx: dict) -> LLMResponse:
        raise NotImplementedError

    def _key(self, default_env: str) -> str:
        env = self.spec.api_key_env or default_env
        key = os.environ.get(env)
        if not key:
            raise ProviderError(f"environment variable {env} is not set for {self.name}")
        return key

    def _post(self, url: str, headers: dict, body: dict) -> dict:
        try:
            r = httpx.post(url, headers=headers, json=body, timeout=self.spec.timeout_s)
        except httpx.HTTPError as e:
            raise ProviderError(f"{self.name}: {e}") from e
        if r.status_code >= 400:
            raise ProviderError(f"{self.name}: HTTP {r.status_code}: {r.text[:500]}")
        return r.json()


class AnthropicProvider(Provider):
    def complete(self, system, messages, json_schema, ctx):
        t0 = time.time()
        url = (self.spec.base_url or "https://api.anthropic.com").rstrip("/") + "/v1/messages"
        body = {
            "model": self.spec.model,
            "max_tokens": self.spec.max_tokens,
            "temperature": self.spec.temperature,
            "system": system,
            "messages": messages,
        }
        data = self._post(url, {"x-api-key": self._key("ANTHROPIC_API_KEY"), "anthropic-version": "2023-06-01"}, body)
        text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
        u = data.get("usage", {})
        return LLMResponse(text, self.spec.model, u.get("input_tokens", 0), u.get("output_tokens", 0), time.time() - t0)


class OpenAICompatibleProvider(Provider):
    default_base = "https://api.openai.com/v1"
    default_env = "OPENAI_API_KEY"

    def complete(self, system, messages, json_schema, ctx):
        t0 = time.time()
        base = (self.spec.base_url or self.default_base).rstrip("/")
        headers = {}
        if self.spec.provider == "openai" or self.spec.api_key_env:
            headers["Authorization"] = f"Bearer {self._key(self.default_env)}"
        body: dict[str, Any] = {
            "model": self.spec.model,
            "temperature": self.spec.temperature,
            "max_tokens": self.spec.max_tokens,
            "messages": [{"role": "system", "content": system}, *messages],
        }
        if json_schema is not None:
            body["response_format"] = {"type": "json_object"}
        try:
            data = self._post(base + "/chat/completions", headers, body)
        except ProviderError as e:
            if json_schema is not None and "response_format" in str(e):
                body.pop("response_format")
                data = self._post(base + "/chat/completions", headers, body)
            else:
                raise
        text = data["choices"][0]["message"].get("content") or ""
        u = data.get("usage") or {}
        return LLMResponse(text, self.spec.model, u.get("prompt_tokens", 0), u.get("completion_tokens", 0), time.time() - t0)


class OllamaProvider(Provider):
    def complete(self, system, messages, json_schema, ctx):
        t0 = time.time()
        base = (self.spec.base_url or "http://localhost:11434").rstrip("/")
        body: dict[str, Any] = {
            "model": self.spec.model,
            "stream": False,
            "messages": [{"role": "system", "content": system}, *messages],
            "options": {"temperature": self.spec.temperature, "num_predict": self.spec.max_tokens},
        }
        if json_schema is not None:
            body["format"] = json_schema  # Ollama >=0.5 constrains decoding to the schema
        data = self._post(base + "/api/chat", {}, body)
        text = data.get("message", {}).get("content", "")
        return LLMResponse(text, self.spec.model, data.get("prompt_eval_count", 0), data.get("eval_count", 0), time.time() - t0)


def make_provider(spec: ModelSpec) -> Provider:
    if spec.provider == "anthropic":
        return AnthropicProvider(spec)
    if spec.provider in ("openai", "openai_compatible"):
        return OpenAICompatibleProvider(spec)
    if spec.provider == "ollama":
        return OllamaProvider(spec)
    if spec.provider == "mock":
        from .mock import MockProvider

        return MockProvider(spec)
    raise ValueError(spec.provider)
