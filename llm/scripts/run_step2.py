import importlib.util
import re
from pathlib import Path

import pandas as pd

MODULE_PATH = Path(__file__).resolve().parents[1] / "src" / "gemini_cli_processor.py"
spec = importlib.util.spec_from_file_location("gemini_cli_processor", MODULE_PATH)
if spec is None or spec.loader is None:
    raise ImportError(f"Failed to load gemini_cli_processor: {MODULE_PATH}")
gemini_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gemini_module)

GeminiCLIProcessor = gemini_module.GeminiCLIProcessor
check_gemini_rules = gemini_module.check_gemini_rules
DEFAULT_GEMINI_CLI_COMMAND = gemini_module.DEFAULT_GEMINI_CLI_COMMAND
format_column_info = gemini_module.format_column_info

ASSETS_DIR = Path("dataset_assets")


def normalize_name(name: str) -> str:
    normalized = re.sub(r'[\\/*?:"<>|]', "", name)
    normalized = normalized.replace(" ", "_").lower().strip("_")
    return normalized


def run_step2_preprocess(dataset_name: str) -> bool:
    check_gemini_rules()
    name = normalize_name(dataset_name)
    info_path = Path("Generalized_dataset_info") / f"{name}_info.txt"
    if not info_path.exists():
        print(f"Error: Abstraction file not found: {info_path}")
        print("  -> Please run Step 1 first.")
        return False

    raw_source_path = ASSETS_DIR / "raw_data" / f"{name}_raw.csv"
    if not raw_source_path.exists():
        print(f"Error: Raw data file not found before preprocessing: {raw_source_path}")
        return False

    output_dir = Path("preprocessed_dat") / name
    output_dir.mkdir(parents=True, exist_ok=True)

    preprocessed_path = output_dir / "preprocessed.csv"
    report_path = output_dir / "preprocessing_report.txt"
    schema_path = output_dir / "schema.json"
    script_path = output_dir / "preprocessing_script.py"

    for existing_path in [preprocessed_path, report_path, schema_path, script_path]:
        if existing_path.exists():
            existing_path.unlink()
    print(f"Overwriting existing preprocessing artifacts: {output_dir}")

    df_raw = pd.read_csv(raw_source_path)
    resolved_target_name = df_raw.columns[-1] if len(df_raw.columns) > 0 else None
    target_unique_values = []
    if resolved_target_name and resolved_target_name in df_raw.columns:
        target_unique_values = df_raw[resolved_target_name].dropna().unique().tolist()

    info_text = info_path.read_text(encoding="utf-8").strip()

    input_info = f"""
[Input Information]
- Data file path: {raw_source_path.resolve()}
- Dataset name: {name}
- Target column name: {resolved_target_name if resolved_target_name else "unknown"}
- Unique values in target column: {target_unique_values}
- Column information (name and dtype):
{format_column_info(df_raw, resolved_target_name)}
- Dataset description:
{info_text}

[Output Destination]
- Output directory: {output_dir.resolve()}
- Preprocessed file: preprocessed.csv
- Report file: preprocessing_report.txt
- Schema file: schema.json
"""

    prompt_path = Path("prompt") / "preprocess_prompt.md"
    processor = GeminiCLIProcessor(
        prompt_file=str(prompt_path),
        input_dir="",
        output_dir="",
        cli_command=DEFAULT_GEMINI_CLI_COMMAND,
        validate_json=False,
    )

    success = processor.generate_and_run_python_script(
        input_text=input_info,
        output_dir=output_dir,
        script_filename=script_path.name,
        required_output_files=[preprocessed_path.name],
    )
    if not success:
        return False

    print(f"Preprocessing outputs saved in: {output_dir}")

    return True


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run step 2")
    parser.add_argument(
        "dataset_name",
        type=str,
        help="Dataset name (must be enclosed in double quotes)",
    )

    args = parser.parse_args()

    success = run_step2_preprocess(dataset_name=args.dataset_name)
    if not success:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
