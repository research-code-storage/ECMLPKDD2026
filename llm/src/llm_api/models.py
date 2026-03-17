from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

SAFE_STOP_ERROR_TEXT = (
    "Error: The meaning of positive/negative target classes or target value direction is unclear. "
    "Please provide a clear target definition."
)
SCHEMA_VERSION = "1.0"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_RETRIES = 2
DEFAULT_TIMEOUT = 600
DEFAULT_MAX_OUTPUT_TOKENS = 65536
DEFAULT_PROVIDER_MODELS: dict[str, str] = {
    "openai": "gpt-5.4",
    "anthropic": "claude-sonnet-4-6",
    "gemini": "gemini-3.1-pro-preview",
}
SUPPORTED_PROVIDERS = tuple(DEFAULT_PROVIDER_MODELS.keys())


@dataclass(frozen=True)
class PromptBundle:
    system_prompt: str
    user_prompt: str
    combined_prompt: str
    prompt_hash: str


@dataclass(frozen=True)
class ProviderResponse:
    text: str
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    raw_payload: dict[str, Any] | list[Any] | str


@dataclass(frozen=True)
class Step4RunConfig:
    dataset_name: str
    provider: str
    model: str
    temperature: float = DEFAULT_TEMPERATURE
    max_retries: int = DEFAULT_MAX_RETRIES
    timeout: int = DEFAULT_TIMEOUT
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS
    overwrite: bool = False
    dry_run: bool = False
    log_dir: Path = Path("logs") / "step4_api"
    input_dir: Path = Path("input_json")
    output_root: Path = Path("sc_json")
    prompt_path: Path = Path("prompt") / "sc_prompt.md"


@dataclass(frozen=True)
class ValidationResult:
    normalized_output: dict[str, Any]
    safe_stop: bool = False


@dataclass(frozen=True)
class Step4LogRecord:
    timestamp: str
    dataset_name: str
    input_path: str
    provider: str
    model: str
    temperature: float
    success: bool
    safe_stop: bool
    latency_sec: float | None
    retry_count: int
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    estimated_cost_usd: float | None
    raw_response_path: str | None
    normalized_output_path: str | None
    sc_output_path: str | None
    error_type: str | None
    error_message: str | None
    prompt_hash: str
    input_json_hash: str
    schema_version: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
