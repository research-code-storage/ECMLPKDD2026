from __future__ import annotations

import importlib
from typing import Any

from llm_api.models import PromptBundle, ProviderResponse
from llm_api.providers.base import BaseProvider


class GeminiProvider(BaseProvider):
    provider_name = "gemini"
    env_var_name = "GEMINI_API_KEY"

    def __init__(self, *, model: str, temperature: float, timeout: int, max_output_tokens: int | None = None):
        super().__init__(
            model=model,
            temperature=temperature,
            timeout=timeout,
            max_output_tokens=max_output_tokens,
        )
        try:
            genai_module = importlib.import_module("google.genai")
            types_module = importlib.import_module("google.genai.types")
        except ImportError as exc:
            raise RuntimeError(
                "The 'google-genai' package is required. Install dependencies from pyproject.toml first."
            ) from exc

        self._genai = genai_module
        self._types = types_module
        self._client = genai_module.Client(api_key=self.api_key)

    def generate(self, prompt: PromptBundle) -> ProviderResponse:
        response = self._client.models.generate_content(
            model=self.model,
            contents=f"{prompt.system_prompt}\n\n{prompt.user_prompt}",
            config=self._types.GenerateContentConfig(
                temperature=self.temperature,
            ),
        )
        raw_payload = self._to_serializable(response)
        text = getattr(response, "text", None) or self._extract_output_text(raw_payload)
        input_tokens, output_tokens, total_tokens = self._extract_usage(raw_payload.get("usage_metadata", {}))
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

        candidates = payload.get("candidates", [])
        fragments: list[str] = []
        for candidate in candidates:
            content = candidate.get("content", {})
            for part in content.get("parts", []):
                text = part.get("text")
                if text:
                    fragments.append(text)
        if fragments:
            return "\n".join(fragments)
        return str(payload)

    def _extract_usage(self, usage_payload: Any) -> tuple[int | None, int | None, int | None]:
        usage_dict = self._to_serializable(usage_payload)
        if isinstance(usage_dict, str):
            return None, None, None
        input_tokens = usage_dict.get("prompt_token_count") or usage_dict.get("input_tokens")
        output_tokens = usage_dict.get("candidates_token_count") or usage_dict.get("output_tokens")
        total_tokens = usage_dict.get("total_token_count") or usage_dict.get("total_tokens")
        return input_tokens, output_tokens, total_tokens
