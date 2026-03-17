import importlib.util
import re
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "src" / "gemini_cli_processor.py"
spec = importlib.util.spec_from_file_location("gemini_cli_processor", MODULE_PATH)
if spec is None or spec.loader is None:
    raise ImportError(f"Failed to load gemini_cli_processor: {MODULE_PATH}")
gemini_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gemini_module)

AbstractionProcessor = gemini_module.AbstractionProcessor
check_gemini_rules = gemini_module.check_gemini_rules

ASSETS_DIR = Path("dataset_assets")


def normalize_name(name: str) -> str:
    normalized = re.sub(r'[\\/*?:"<>|]', "", name)
    normalized = normalized.replace(" ", "_").lower().strip("_")
    return normalized


def resolve_overview_path(dataset_name: str) -> Path | None:
    overview_dir = ASSETS_DIR / "dataset_overview"
    candidates = [overview_dir / f"{dataset_name}_overview.txt"]
    for path in candidates:
        if path.exists():
            return path
    return None


def run_step1_generate_abstraction(dataset_name: str, overwrite: bool = False) -> bool:
    check_gemini_rules()
    name = normalize_name(dataset_name)
    overview_dir = ASSETS_DIR / "dataset_overview"
    info_dir = Path("Generalized_dataset_info")

    overview_path = resolve_overview_path(name)
    if overview_path is None:
        print("Error: Overview file not found.")
        print(f"  Required file: {overview_dir / f'{name}_overview.txt'}")
        return False

    output_filename = overview_path.name.replace("_overview", "_info")
    info_path = info_dir / output_filename
    if info_path.exists() and not overwrite:
        print(f"Skipping because output file already exists: {info_path}")
        return True
    if info_path.exists() and overwrite:
        print(f"Overwriting existing abstraction file: {info_path}")

    processor = AbstractionProcessor(
        input_dir=str(overview_dir), output_dir=str(info_dir)
    )
    result = processor.process_file(overview_path)
    if not result:
        print("Error: Failed to generate abstraction.")
        return False

    info_path.parent.mkdir(parents=True, exist_ok=True)
    info_path.write_text(result, encoding="utf-8")
    print(f"Abstraction saved to: {info_path}")
    return True


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run step 1")
    parser.add_argument(
        "dataset_name",
        type=str,
        help="Dataset name (must be enclosed in double quotes)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the existing abstraction output if it already exists.",
    )

    args = parser.parse_args()

    success = run_step1_generate_abstraction(
        dataset_name=args.dataset_name,
        overwrite=args.overwrite,
    )
    if not success:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
