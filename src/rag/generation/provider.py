"""
src/rag/generation/provider.py
--------------------------------
DCLC-RAG Phase 6D — Configurable LLM Provider Abstraction.

Provides unified interfaces for:
1. Groq Cloud API (hosted LLM option via httpx, e.g. llama-3.3-70b-versatile).
   NOTE: Transmits retrieved evidence packets to external Groq API servers.
2. Local OpenAI-compatible server (e.g. Ollama, vLLM, llama.cpp server).
   Zero external network calls; 100% offline.
3. Deterministic Mock LLM (for hermetic unit tests and automated CI).

Strict Safety Guarantees:
- Secrets are read exclusively from environment variables (DCLC_GROQ_API_KEY).
- API keys are never printed, logged, or included in error traces.
- Failures raise explicit LLMProviderError without silently swapping models
  or fabricating hallucinated outputs.
"""

from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import httpx


class LLMProviderError(Exception):
    """Raised when an LLM provider call fails or is misconfigured."""


@dataclass
class LLMResponse:
    """Standardized response from any LLM provider."""
    content: str
    provider: str
    model: str
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    latency_ms: float = 0.0


class LLMProvider(ABC):
    """Abstract base class for all LLM providers."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> LLMResponse:
        """Generate a response given a prompt and system prompt."""
        raise NotImplementedError

    @abstractmethod
    def is_available(self) -> bool:
        """Check if provider is configured and reachable."""
        raise NotImplementedError

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the canonical provider name."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# 1. Groq Cloud API Provider
# ---------------------------------------------------------------------------


class GroqProvider(LLMProvider):
    """
    Hosted LLM provider using Groq Cloud API.

    Transmits evidence packets over HTTPS to Groq API servers.
    Requires DCLC_GROQ_API_KEY environment variable.
    """

    GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
    DEFAULT_MODEL = "qwen/qwen3.8-27b"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        timeout_seconds: float = 30.0,
    ) -> None:
        self.api_key = api_key or os.environ.get("DCLC_GROQ_API_KEY", "").strip()
        self.model_name = model_name or os.environ.get("DCLC_GROQ_MODEL", self.DEFAULT_MODEL).strip()
        self.timeout_seconds = timeout_seconds

    @property
    def provider_name(self) -> str:
        return "groq"

    def is_available(self) -> bool:
        return bool(self.api_key and len(self.api_key) > 5)

    def generate(
        self,
        prompt: str,
        system_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> LLMResponse:
        if not self.is_available():
            raise LLMProviderError(
                "GroqProvider is not configured: DCLC_GROQ_API_KEY is missing or empty. "
                "Set the environment variable or choose a local/mock provider."
            )

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "temperature": max(0.0, min(2.0, temperature)),
            "max_tokens": max(1, min(4096, max_tokens)),
        }

        t0 = time.perf_counter()
        try:
            with httpx.Client(timeout=self.timeout_seconds) as client:
                resp = client.post(self.GROQ_API_URL, headers=headers, json=payload)
        except Exception as exc:
            raise LLMProviderError(f"Groq API connection error: {exc}") from exc

        latency_ms = (time.perf_counter() - t0) * 1000.0

        if resp.status_code != 200:
            # Mask key in error outputs
            raise LLMProviderError(
                f"Groq API request failed with HTTP {resp.status_code}: {resp.text[:300]}"
            )

        data = resp.json()
        try:
            choice = data["choices"][0]
            content = choice["message"]["content"]
            usage = data.get("usage", {})
            return LLMResponse(
                content=content,
                provider="groq",
                model=self.model_name,
                prompt_tokens=usage.get("prompt_tokens"),
                completion_tokens=usage.get("completion_tokens"),
                latency_ms=latency_ms,
            )
        except (KeyError, IndexError) as exc:
            raise LLMProviderError(f"Unexpected Groq API response schema: {data}") from exc


# ---------------------------------------------------------------------------
# 2. Local OpenAI-Compatible Provider
# ---------------------------------------------------------------------------


class LocalOpenAICompatibleProvider(LLMProvider):
    """
    Local LLM provider connecting to any local OpenAI-compatible server.
    Examples: Ollama (http://localhost:11434/v1), vLLM, llama.cpp server.

    Zero external network calls; 100% offline.
    """

    DEFAULT_BASE_URL = "http://localhost:11434/v1"
    DEFAULT_MODEL = "llama3.2:3b"

    def __init__(
        self,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
        timeout_seconds: float = 60.0,
    ) -> None:
        raw_url = base_url or os.environ.get("DCLC_LOCAL_BASE_URL", self.DEFAULT_BASE_URL)
        self.base_url = raw_url.rstrip("/")
        self.model_name = model_name or os.environ.get("DCLC_LOCAL_MODEL", self.DEFAULT_MODEL)
        self.timeout_seconds = timeout_seconds

    @property
    def provider_name(self) -> str:
        return "local"

    def is_available(self) -> bool:
        try:
            with httpx.Client(timeout=2.0) as client:
                r = client.get(f"{self.base_url}/models")
                return r.status_code == 200
        except Exception:
            return False

    def generate(
        self,
        prompt: str,
        system_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> LLMResponse:
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "temperature": max(0.0, min(2.0, temperature)),
            "max_tokens": max(1, min(4096, max_tokens)),
        }

        t0 = time.perf_counter()
        try:
            with httpx.Client(timeout=self.timeout_seconds) as client:
                resp = client.post(url, json=payload)
        except Exception as exc:
            raise LLMProviderError(f"Local LLM connection error to {self.base_url}: {exc}") from exc

        latency_ms = (time.perf_counter() - t0) * 1000.0

        if resp.status_code != 200:
            raise LLMProviderError(
                f"Local LLM request to {self.base_url} failed with HTTP {resp.status_code}: {resp.text[:300]}"
            )

        data = resp.json()
        try:
            choice = data["choices"][0]
            content = choice["message"]["content"]
            usage = data.get("usage", {})
            return LLMResponse(
                content=content,
                provider="local",
                model=self.model_name,
                prompt_tokens=usage.get("prompt_tokens"),
                completion_tokens=usage.get("completion_tokens"),
                latency_ms=latency_ms,
            )
        except (KeyError, IndexError) as exc:
            raise LLMProviderError(f"Unexpected Local LLM response format: {data}") from exc


# ---------------------------------------------------------------------------
# 3. Deterministic Mock LLM Provider
# ---------------------------------------------------------------------------


class MockLLMProvider(LLMProvider):
    """
    Deterministic mock provider for unit tests, offline continuous integration,
    and benchmarking without network or API dependencies.
    """

    def __init__(
        self,
        default_answer: Optional[str] = None,
        custom_mapping: Optional[Dict[str, str]] = None,
    ) -> None:
        self.default_answer = default_answer
        self.custom_mapping = custom_mapping or {}

    @property
    def provider_name(self) -> str:
        return "mock"

    def is_available(self) -> bool:
        return True

    def generate(
        self,
        prompt: str,
        system_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> LLMResponse:
        t0 = time.perf_counter()

        # 1. Check custom query mapping
        for key, ans in self.custom_mapping.items():
            if key.lower() in prompt.lower():
                latency_ms = (time.perf_counter() - t0) * 1000.0
                return LLMResponse(
                    content=ans,
                    provider="mock",
                    model="mock-deterministic",
                    prompt_tokens=len(prompt.split()),
                    completion_tokens=len(ans.split()),
                    latency_ms=latency_ms,
                )

        # 2. If prompt has simulated prompt injection attempt, resist it
        if "ignore all previous instructions" in prompt.lower():
            answer = "I must adhere to system grounding instructions and ignore overriding commands inside documents."
            return LLMResponse(
                content=answer,
                provider="mock",
                model="mock-deterministic",
                prompt_tokens=len(prompt.split()),
                completion_tokens=len(answer.split()),
                latency_ms=(time.perf_counter() - t0) * 1000.0,
            )

        # 3. If explicit default answer set
        if self.default_answer:
            return LLMResponse(
                content=self.default_answer,
                provider="mock",
                model="mock-deterministic",
                prompt_tokens=len(prompt.split()),
                completion_tokens=len(self.default_answer.split()),
                latency_ms=(time.perf_counter() - t0) * 1000.0,
            )

        # 4. Synthesize grounded answer citing first available chunk tag
        import re
        chunk_tags = re.findall(r"\[(chunk_\d+)\]", prompt)
        if chunk_tags:
            first_tag = f"[{chunk_tags[0]}]"
            answer = f"According to the provided document evidence {first_tag}, the requested parameters are verified."
        else:
            answer = "Insufficient evidence provided in the document context."

        latency_ms = (time.perf_counter() - t0) * 1000.0
        return LLMResponse(
            content=answer,
            provider="mock",
            model="mock-deterministic",
            prompt_tokens=len(prompt.split()),
            completion_tokens=len(answer.split()),
            latency_ms=latency_ms,
        )


# ---------------------------------------------------------------------------
# Factory Helper
# ---------------------------------------------------------------------------


def get_llm_provider(
    provider_type: Optional[str] = None,
    **kwargs: Any,
) -> LLMProvider:
    """
    Factory function instantiating configured LLMProvider.

    Priority:
    1. Explicit provider_type argument ('groq', 'local', 'mock').
    2. Environment variable DCLC_RAG_PROVIDER.
    3. Defaults to 'mock' if nothing configured (safe, non-breaking).
    """
    p_type = (provider_type or os.environ.get("DCLC_RAG_PROVIDER", "mock")).lower().strip()

    if p_type == "groq":
        return GroqProvider(**kwargs)
    elif p_type in ("local", "ollama"):
        return LocalOpenAICompatibleProvider(**kwargs)
    elif p_type == "mock":
        return MockLLMProvider(**kwargs)
    else:
        raise LLMProviderError(
            f"Unsupported DCLC_RAG_PROVIDER: '{p_type}'. Must be one of: 'groq', 'local', 'mock'."
        )
