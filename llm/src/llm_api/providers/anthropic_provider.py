from __future__ import annotations

import importlib
from typing import Any

from llm_api.models import DEFAULT_MAX_OUTPUT_TOKENS, PromptBundle, ProviderResponse
from llm_api.providers.base import BaseProvider


class AnthropicProvider(BaseProvider):
    provider_name = "anthropic"
    env_var_name = "ANTHROPIC_API_KEY"

    def __init__(self, *, model: str, temperature: float, timeout: int, max_output_tokens: int | None = None):
        super().__init__(
            model=model,
            temperature=temperature,
            timeout=timeout,
            max_output_tokens=max_output_tokens or DEFAULT_MAX_OUTPUT_TOKENS,
        )
        try:
            anthropic_module = importlib.import_module("anthropic")
        except ImportError as exc:
            raise RuntimeError(
                "The 'anthropic' package is required. Install dependencies from pyproject.toml first."
            ) from exc

        self._client = anthropic_module.Anthropic(api_key=self.api_key, timeout=self.timeout)

    def generate(self, prompt: PromptBundle) -> ProviderResponse:
        response = self._client.messages.create(
            model=self.model,
            temperature=self.temperature,
            system=prompt.system_prompt,
            messages=[{"role": "user", "content": prompt.user_prompt}],
            max_tokens=self.max_output_tokens,
        )
        raw_payload = self._to_serializable(response)
        text = self._extract_output_text(raw_payload)
        input_tokens, output_tokens, total_tokens = self._extract_usage(raw_payload.get("usage", {}))
        return ProviderResponse(
            text=text.strip(),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            raw_payload=raw_payload,
        )

    def _extract_output_text(self, payload: dict[str, Any] | list[Any] | str) -> str:
        if isinstance(payload, str):
            return payload
        if isinstance(payload, list):
            return "\n".join(self._extract_output_text(item) for item in payload if item)

        content = payload.get("content", [])
        parts: list[str] = []
        for item in content:
            if item.get("type") == "text" and item.get("text"):
                parts.append(item["text"])
        if parts:
            return "\n".join(parts)
        return str(payload)

    def _extract_usage(self, usage_payload: Any) -> tuple[int | None, int | None, int | None]:
        if isinstance(usage_payload, str):
            return None, None, None
        usage_dict = self._to_serializable(usage_payload)
        if isinstance(usage_dict, str):
            return None, None, None
        input_tokens = usage_dict.get("input_tokens")
        output_tokens = usage_dict.get("output_tokens")
        total_tokens = None
        if input_tokens is not None and output_tokens is not None:
            total_tokens = input_tokens + output_tokens
        return input_tokens, output_tokens, total_tokens
