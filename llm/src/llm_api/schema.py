from __future__ import annotations

import json
from typing import Any

from .models import SAFE_STOP_ERROR_TEXT, ValidationResult

EXPECTED_TOP_LEVEL_KEYS = ("info", "target", "variable_info", "features")
EXPECTED_FEATURE_KEYS = ("name", "description", "sign_constraint", "reason")
ALLOWED_SIGN_VALUES = {-1, 0, 1}


class SchemaValidationError(ValueError):
    """Raised when an LLM output does not satisfy the Step 4 schema."""


class SafeStopError(SchemaValidationError):
    """Raised when the model returns the expected safe-stop sentinel."""


def _extract_json_object_text(result_text: str) -> str:
    stripped = result_text.strip()
    if not stripped:
        return stripped

    start = stripped.find("{")
    if start == -1:
        return stripped

    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(stripped)):
        char = stripped[index]

        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return stripped[start : index + 1]

    return stripped


def validate_sign_constraint_output(
    result_text: str,
    input_data: dict[str, Any],
) -> ValidationResult:
    stripped = result_text.strip()
    if stripped == SAFE_STOP_ERROR_TEXT:
        raise SafeStopError(SAFE_STOP_ERROR_TEXT)

    json_candidate = _extract_json_object_text(stripped)

    try:
        output_data = json.loads(json_candidate)
    except json.JSONDecodeError as exc:
        raise SchemaValidationError(f"Generated output is not valid JSON: {exc}") from exc

    if not isinstance(output_data, dict):
        raise SchemaValidationError("Generated JSON root must be an object.")

    actual_top_level_keys = tuple(output_data.keys())
    if actual_top_level_keys != EXPECTED_TOP_LEVEL_KEYS:
        raise SchemaValidationError(
            "Generated JSON must preserve the exact top-level keys and order. "
            f"Expected {list(EXPECTED_TOP_LEVEL_KEYS)}, got {list(actual_top_level_keys)}."
        )

    for key in ("info", "target", "variable_info"):
        if output_data.get(key) != input_data.get(key):
            raise SchemaValidationError(
                f"Top-level field '{key}' must be preserved exactly from the input JSON."
            )

    input_features = input_data.get("features")
    output_features = output_data.get("features")
    if not isinstance(input_features, list):
        raise SchemaValidationError("Input JSON must contain a 'features' list.")
    if not isinstance(output_features, list):
        raise SchemaValidationError("Generated JSON must contain a 'features' list.")
    if len(output_features) != len(input_features):
        raise SchemaValidationError(
            "Generated JSON must preserve the number of features exactly. "
            f"Expected {len(input_features)}, got {len(output_features)}."
        )

    normalized_features: list[dict[str, Any]] = []
    for index, (input_feature, output_feature) in enumerate(
        zip(input_features, output_features, strict=True),
        start=1,
    ):
        if not isinstance(input_feature, dict):
            raise SchemaValidationError(f"Input feature #{index} is not a JSON object.")
        if not isinstance(output_feature, dict):
            raise SchemaValidationError(f"Generated feature #{index} must be a JSON object.")

        actual_feature_keys = tuple(output_feature.keys())
        if actual_feature_keys != EXPECTED_FEATURE_KEYS:
            raise SchemaValidationError(
                "Each feature must preserve the exact keys and order "
                f"{list(EXPECTED_FEATURE_KEYS)}; got {list(actual_feature_keys)} at index {index}."
            )

        expected_name = input_feature.get("name")
        actual_name = output_feature.get("name")
        if actual_name != expected_name:
            raise SchemaValidationError(
                "Feature order or feature names do not match the input JSON. "
                f"Expected '{expected_name}', got '{actual_name}' at index {index}."
            )

        expected_description = input_feature.get("description")
        actual_description = output_feature.get("description")
        if actual_description != expected_description:
            raise SchemaValidationError(
                f"Feature '{expected_name}' must preserve its description exactly."
            )

        raw_sign = output_feature.get("sign_constraint")
        try:
            sign_value = int(raw_sign)
        except (TypeError, ValueError) as exc:
            raise SchemaValidationError(
                f"Feature '{expected_name}' has a non-integer sign_constraint: {raw_sign!r}"
            ) from exc

        if sign_value not in ALLOWED_SIGN_VALUES:
            raise SchemaValidationError(
                f"Feature '{expected_name}' has invalid sign_constraint {sign_value}. "
                "Allowed values are -1, 0, 1."
            )

        reason = output_feature.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            raise SchemaValidationError(
                f"Feature '{expected_name}' must have a non-empty string reason."
            )

        normalized_feature = {
            "name": expected_name,
            "description": expected_description,
            "sign_constraint": sign_value,
            "reason": reason.strip(),
        }
        normalized_features.append(normalized_feature)

    normalized_output = {
        "info": input_data["info"],
        "target": input_data["target"],
        "variable_info": input_data["variable_info"],
        "features": normalized_features,
    }
    return ValidationResult(normalized_output=normalized_output, safe_stop=False)
