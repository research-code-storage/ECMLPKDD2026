from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any

from llm_api.models import PromptBundle, ProviderResponse


class BaseProvider(ABC):
    provider_name: str = "base"
    env_var_name: str = ""

    def __init__(self, *, model: str, temperature: float, timeout: int, max_output_tokens: int | None = None):
        self.model = model
        self.temperature = temperature
        self.timeout = timeout
        self.max_output_tokens = max_output_tokens
        self.api_key = self._require_env_var(self.env_var_name)

    @abstractmethod
    def generate(self, prompt: PromptBundle) -> ProviderResponse:
        raise NotImplementedError

    @staticmethod
    def _require_env_var(env_var_name: str) -> str:
        value = os.environ.get(env_var_name)
        if not value:
            raise RuntimeError(f"Environment variable '{env_var_name}' is required.")
        return value

    @staticmethod
    def _to_serializable(payload: Any) -> dict[str, Any] | list[Any] | str:
        if payload is None:
            return ""
        model_dump = getattr(payload, "model_dump", None)
        if callable(model_dump):
            return model_dump(mode="python")
        to_dict = getattr(payload, "to_dict", None)
        if callable(to_dict):
            return to_dict()
        if isinstance(payload, (dict, list, str)):
            return payload
        if hasattr(payload, "__dict__"):
            return dict(payload.__dict__)
        return str(payload)
