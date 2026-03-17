from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from llm_api.logging_utils import (  # noqa: E402
    append_log_record,
    compact_timestamp,
    save_normalized_output,
    save_raw_response,
    sha256_text,
    utc_timestamp,
)
from llm_api.models import (  # noqa: E402
    DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_MAX_RETRIES,
    DEFAULT_PROVIDER_MODELS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TIMEOUT,
    SCHEMA_VERSION,
    SUPPORTED_PROVIDERS,
    Step4LogRecord,
    Step4RunConfig,
)
from llm_api.pricing import estimate_cost_usd  # noqa: E402
from llm_api.prompting import build_step4_prompt  # noqa: E402
from llm_api.providers import build_provider  # noqa: E402
from llm_api.retry import call_with_retry  # noqa: E402
from llm_api.schema import SafeStopError, SchemaValidationError, validate_sign_constraint_output  # noqa: E402


def normalize_name(name: str) -> str:
    normalized = re.sub(r'[\\/*?:"<>|]', "", name)
    normalized = normalized.replace(" ", "_").lower().strip("_")
    return normalized


def get_input_path(dataset_name: str, input_dir: Path) -> Path:
    return input_dir / f"{dataset_name}_input.json"


def get_sc_output_path(dataset_name: str, output_root: Path, provider: str, model: str) -> Path:
    return output_root / provider / model / f"{dataset_name}_sc.json"


def resolve_provider_models(
    provider: str | None,
    model: str | None,
) -> list[tuple[str, str]]:
    provider_name = provider or "all"
    if provider_name == "all":
        if model is not None:
            raise ValueError("--model can only be used together with a single --provider.")
        return [(name, DEFAULT_PROVIDER_MODELS[name]) for name in SUPPORTED_PROVIDERS]

    if provider_name not in SUPPORTED_PROVIDERS:
        raise ValueError(f"Unsupported provider: {provider_name}")
    return [(provider_name, model or DEFAULT_PROVIDER_MODELS[provider_name])]


def load_input_json(input_path: Path) -> tuple[dict[str, Any], str]:
    input_text = input_path.read_text(encoding="utf-8")
    input_data = json.loads(input_text)
    if not isinstance(input_data, dict):
        raise ValueError("Input JSON root must be an object.")
    return input_data, input_text


def run_single_provider(config: Step4RunConfig) -> bool:
    input_path = get_input_path(config.dataset_name, config.input_dir)
    if not input_path.exists():
        print(f"Error: input_json file not found: {input_path}")
        print("  -> Please run Step 3 first.")
        return False

    output_path = get_sc_output_path(
        dataset_name=config.dataset_name,
        output_root=config.output_root,
        provider=config.provider,
        model=config.model,
    )
    if output_path.exists() and not config.overwrite:
        print(f"Skip: sign-constraint file already exists -> {output_path}")
        return True

    input_data, input_text = load_input_json(input_path)
    prompt_bundle = build_step4_prompt(input_data=input_data, prompt_path=config.prompt_path)
    input_json_hash = sha256_text(input_text)

    if config.dry_run:
        print(
            f"[dry-run] dataset={config.dataset_name}, provider={config.provider}, model={config.model}, "
            f"prompt_hash={prompt_bundle.prompt_hash}"
        )
        print(f"[dry-run] output path -> {output_path}")
        return True

    provider_client = build_provider(
        config.provider,
        model=config.model,
        temperature=config.temperature,
        timeout=config.timeout,
        max_output_tokens=config.max_output_tokens,
    )

    timestamp_for_log = utc_timestamp()
    timestamp_for_path = compact_timestamp()
    raw_response_path: Path | None = None
    normalized_output_path: Path | None = None
    sc_output_path: Path | None = None
    estimated_cost_usd: float | None = None
    error_type: str | None = None
    error_message: str | None = None
    retry_count = 0
    latency_sec: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    safe_stop = False
    success = False

    print(f"[Step 4] provider={config.provider}, model={config.model}, dataset={config.dataset_name}")
    start_time = time.perf_counter()
    try:
        response, retry_count = call_with_retry(
            lambda: provider_client.generate(prompt_bundle),
            max_retries=config.max_retries,
        )
        latency_sec = round(time.perf_counter() - start_time, 3)
        input_tokens = response.input_tokens
        output_tokens = response.output_tokens
        total_tokens = response.total_tokens
        estimated_cost_usd = estimate_cost_usd(
            provider=config.provider,
            model=config.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
        raw_response_path = save_raw_response(
            config.log_dir,
            timestamp=timestamp_for_path,
            dataset_name=config.dataset_name,
            provider=config.provider,
            model=config.model,
            raw_payload=response.raw_payload,
        )

        validation_result = validate_sign_constraint_output(
            result_text=response.text,
            input_data=input_data,
        )
        normalized_output = validation_result.normalized_output
        normalized_output_path = save_normalized_output(
            config.log_dir,
            timestamp=timestamp_for_path,
            dataset_name=config.dataset_name,
            provider=config.provider,
            model=config.model,
            normalized_output=normalized_output,
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(normalized_output, indent=4, ensure_ascii=False),
            encoding="utf-8",
        )
        sc_output_path = output_path
        success = True
        print(f"Success: saved sign constraints -> {output_path}")
    except SafeStopError as exc:
        latency_sec = round(time.perf_counter() - start_time, 3)
        safe_stop = True
        error_type = "safe_stop"
        error_message = str(exc)
        print(f"Safe-stop: provider={config.provider}, dataset={config.dataset_name}")
    except SchemaValidationError as exc:
        latency_sec = round(time.perf_counter() - start_time, 3)
        error_type = "schema_validation_error"
        error_message = str(exc)
        print(f"Schema validation failed: {exc}")
    except Exception as exc:
        latency_sec = round(time.perf_counter() - start_time, 3)
        error_type = exc.__class__.__name__
        error_message = str(exc)
        print(
            f"Provider call failed: provider={config.provider}, model={config.model}, "
            f"dataset={config.dataset_name}, error={exc}"
        )

    append_log_record(
        config.log_dir,
        Step4LogRecord(
            timestamp=timestamp_for_log,
            dataset_name=config.dataset_name,
            input_path=str(input_path),
            provider=config.provider,
            model=config.model,
            temperature=config.temperature,
            success=success,
            safe_stop=safe_stop,
            latency_sec=latency_sec,
            retry_count=retry_count,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            estimated_cost_usd=estimated_cost_usd,
            raw_response_path=str(raw_response_path) if raw_response_path else None,
            normalized_output_path=str(normalized_output_path) if normalized_output_path else None,
            sc_output_path=str(sc_output_path) if sc_output_path else None,
            error_type=error_type,
            error_message=error_message,
            prompt_hash=prompt_bundle.prompt_hash,
            input_json_hash=input_json_hash,
            schema_version=SCHEMA_VERSION,
        ),
    )
    return success


def run_step4_generate_sign_constraints(
    dataset_name: str,
    *,
    provider: str | None = None,
    model: str | None = None,
    temperature: float = DEFAULT_TEMPERATURE,
    overwrite: bool = False,
    max_retries: int = DEFAULT_MAX_RETRIES,
    timeout: int = DEFAULT_TIMEOUT,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    log_dir: str | Path = Path("logs") / "step4_api",
    dry_run: bool = False,
) -> bool:
    name = normalize_name(dataset_name)
    provider_models = resolve_provider_models(provider=provider, model=model)

    all_success = True
    for provider_name, resolved_model in provider_models:
        config = Step4RunConfig(
            dataset_name=name,
            provider=provider_name,
            model=resolved_model,
            temperature=temperature,
            overwrite=overwrite,
            max_retries=max_retries,
            timeout=timeout,
            max_output_tokens=max_output_tokens,
            dry_run=dry_run,
            log_dir=Path(log_dir),
            input_dir=Path("input_json"),
            output_root=Path("sc_json"),
            prompt_path=Path("prompt") / "sc_prompt.md",
        )
        success = run_single_provider(config)
        all_success = all_success and success

    return all_success


def main() -> None:
    import argparse

    os.chdir(PROJECT_ROOT)

    parser = argparse.ArgumentParser(description="Run Step 4 using multi-provider LLM APIs.")
    parser.add_argument(
        "dataset_name",
        type=str,
        help="Dataset name (must be enclosed in double quotes)",
    )
    parser.add_argument(
        "--provider",
        choices=[*SUPPORTED_PROVIDERS, "all"],
        default="all",
        help="Target provider. Default: all supported providers.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override the model name. Only valid when --provider is a single provider.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help="Sampling temperature. Default: 0.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing sign-constraint outputs.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=DEFAULT_MAX_RETRIES,
        help="Maximum number of retries for transient communication errors.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help="Provider timeout in seconds.",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=DEFAULT_MAX_OUTPUT_TOKENS,
        help="Maximum output tokens requested from the provider.",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("logs") / "step4_api",
        help="Directory for Step 4 API logs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build prompts and output paths without calling any API.",
    )

    args = parser.parse_args()

    try:
        success = run_step4_generate_sign_constraints(
            dataset_name=args.dataset_name,
            provider=args.provider,
            model=args.model,
            temperature=args.temperature,
            overwrite=args.overwrite,
            max_retries=args.max_retries,
            timeout=args.timeout,
            max_output_tokens=args.max_output_tokens,
            log_dir=args.log_dir,
            dry_run=args.dry_run,
        )
    except ValueError as exc:
        print(f"Error: {exc}")
        raise SystemExit(2) from exc

    if not success:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
