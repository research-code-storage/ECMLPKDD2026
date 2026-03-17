from __future__ import annotations

import importlib
from typing import Any

from llm_api.models import PromptBundle, ProviderResponse
from llm_api.providers.base import BaseProvider


class OpenAIProvider(BaseProvider):
    provider_name = "openai"
    env_var_name = "OPENAI_API_KEY"

    def __init__(self, *, model: str, temperature: float, timeout: int, max_output_tokens: int | None = None):
        super().__init__(
            model=model,
            temperature=temperature,
            timeout=timeout,
            max_output_tokens=max_output_tokens,
        )
        try:
            openai_module = importlib.import_module("openai")
        except ImportError as exc:
            raise RuntimeError(
                "The 'openai' package is required. Install dependencies from pyproject.toml first."
            ) from exc

        self._client = openai_module.OpenAI(api_key=self.api_key, timeout=self.timeout)

    def generate(self, prompt: PromptBundle) -> ProviderResponse:
        response = self._client.responses.create(
            model=self.model,
            temperature=self.temperature,
            input=[
                {
                    "role": "system",
                    "content": [{"type": "input_text", "text": prompt.system_prompt}],
                },
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": prompt.user_prompt}],
                },
            ],
        )
        raw_payload = self._to_serializable(response)
        text = getattr(response, "output_text", None) or self._extract_output_text(raw_payload)
        usage = getattr(response, "usage", None)
        input_tokens, output_tokens, total_tokens = self._extract_usage(usage)
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

        output_items = payload.get("output", [])
        fragments: list[str] = []
        for item in output_items:
            for content in item.get("content", []):
                if content.get("type") in {"output_text", "text"} and content.get("text"):
                    fragments.append(content["text"])
        if fragments:
            return "\n".join(fragments)
        return str(payload)

    def _extract_usage(self, usage: Any) -> tuple[int | None, int | None, int | None]:
        if usage is None:
            return None, None, None
        usage_payload = self._to_serializable(usage)
        if isinstance(usage_payload, str):
            return None, None, None
        input_tokens = usage_payload.get("input_tokens")
        output_tokens = usage_payload.get("output_tokens")
        total_tokens = usage_payload.get("total_tokens")
        return input_tokens, output_tokens, total_tokens
