from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .models import Step4LogRecord

CSV_FIELDNAMES = [
    "timestamp",
    "dataset_name",
    "input_path",
    "provider",
    "model",
    "temperature",
    "success",
    "safe_stop",
    "latency_sec",
    "retry_count",
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "estimated_cost_usd",
    "raw_response_path",
    "normalized_output_path",
    "sc_output_path",
    "error_type",
    "error_message",
    "prompt_hash",
    "input_json_hash",
    "schema_version",
]


def utc_timestamp() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def compact_timestamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def save_raw_response(
    log_dir: Path,
    *,
    timestamp: str,
    dataset_name: str,
    provider: str,
    model: str,
    raw_payload: dict[str, Any] | list[Any] | str,
) -> Path:
    raw_dir = log_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    safe_model = model.replace("/", "_")
    if isinstance(raw_payload, (dict, list)):
        path = raw_dir / f"{timestamp}_{dataset_name}_{provider}_{safe_model}.json"
        path.write_text(json.dumps(raw_payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    path = raw_dir / f"{timestamp}_{dataset_name}_{provider}_{safe_model}.txt"
    path.write_text(str(raw_payload), encoding="utf-8")
    return path


def save_normalized_output(
    log_dir: Path,
    *,
    timestamp: str,
    dataset_name: str,
    provider: str,
    model: str,
    normalized_output: dict[str, Any],
) -> Path:
    normalized_dir = log_dir / "normalized"
    normalized_dir.mkdir(parents=True, exist_ok=True)
    safe_model = model.replace("/", "_")
    path = normalized_dir / f"{timestamp}_{dataset_name}_{provider}_{safe_model}.json"
    path.write_text(
        json.dumps(normalized_output, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


def append_log_record(log_dir: Path, record: Step4LogRecord) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = log_dir / "requests.jsonl"
    summary_csv_path = log_dir / "summary.csv"

    with jsonl_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")

    row = record.to_dict()
    needs_header = not summary_csv_path.exists()
    with summary_csv_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDNAMES)
        if needs_header:
            writer.writeheader()
        writer.writerow({key: row.get(key) for key in CSV_FIELDNAMES})
