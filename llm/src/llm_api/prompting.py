from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .models import PromptBundle

EXTRA_CONSTRAINTS = [
    "Respond in English only.",
    "Output only a valid JSON object.",
    "Do not wrap the JSON in Markdown code fences.",
    "Do not include any analysis, notes, preamble, or text before/after the JSON.",
    "Preserve all top-level fields exactly.",
    "Preserve the original features order exactly.",
    "Do not add, remove, or rename any feature.",
    "Do not modify any existing name, description, info, target, or variable_info text.",
    "sign_constraint must be one of -1, 0, or 1.",
    "reason must be a non-empty English sentence.",
    "Keep each reason extremely short: exactly one sentence, ideally 5-12 words.",
    "Use the shortest sufficient causal justification for each reason.",
]


def build_step4_prompt(
    input_data: dict[str, Any],
    prompt_path: Path,
) -> PromptBundle:
    base_prompt = prompt_path.read_text(encoding="utf-8").strip()
    constraints = "\n".join(f"- {item}" for item in EXTRA_CONSTRAINTS)
    system_prompt = (
        f"{base_prompt}\n\n"
        "Additional implementation constraints:\n"
        f"{constraints}"
    )
    serialized_input = json.dumps(input_data, ensure_ascii=False, indent=2)
    user_prompt = f"Input JSON:\n{serialized_input}"
    combined_prompt = f"{system_prompt}\n\n{user_prompt}"
    prompt_hash = hashlib.sha256(combined_prompt.encode("utf-8")).hexdigest()
    return PromptBundle(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        combined_prompt=combined_prompt,
        prompt_hash=prompt_hash,
    )
