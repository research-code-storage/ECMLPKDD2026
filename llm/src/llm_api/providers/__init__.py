from .anthropic_provider import AnthropicProvider
from .gemini_provider import GeminiProvider
from .openai_provider import OpenAIProvider


def build_provider(
    provider: str,
    *,
    model: str,
    temperature: float,
    timeout: int,
    max_output_tokens: int | None = None,
):
    provider_map = {
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
        "gemini": GeminiProvider,
    }
    try:
        provider_cls = provider_map[provider]
    except KeyError as exc:
        raise ValueError(f"Unsupported provider: {provider}") from exc
    return provider_cls(
        model=model,
        temperature=temperature,
        timeout=timeout,
        max_output_tokens=max_output_tokens,
    )


__all__ = [
    "AnthropicProvider",
    "GeminiProvider",
    "OpenAIProvider",
    "build_provider",
]
