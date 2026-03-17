import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Literal, Optional

import pandas as pd


SAFE_STOP_ERROR_TEXT = (
    "Error: The meaning of positive/negative target classes or target value direction is unclear. "
    "Please provide a clear target definition."
)

DEFAULT_GEMINI_CLI_COMMAND = os.environ.get(
    "GEMINI_CLI_COMMAND", "gemini -m precise-mode"
)


def check_gemini_rules() -> None:
    global_rules_path = Path.home() / ".gemini" / "GEMINI.md"
    if global_rules_path.exists():
        content = global_rules_path.read_text(encoding="utf-8").strip()
        if content:
            raise RuntimeError(
                f"Global rules file is not empty: {global_rules_path}\n"
                "Please empty it or delete/rename it before running the workflow."
            )

    project_rules_path = Path.cwd() / ".gemini" / "GEMINI.md"
    if project_rules_path.exists():
        content = project_rules_path.read_text(encoding="utf-8").strip()
        if content:
            raise RuntimeError(
                f"Project-specific rules file is not empty: {project_rules_path}\n"
                "Please empty it or delete/rename it before running the workflow."
            )


def format_column_info(df: pd.DataFrame, target_name: str | None) -> str:
    lines: list[str] = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        if pd.api.types.is_numeric_dtype(df[col]):
            inferred_type = "Continuous"
        else:
            inferred_type = "Categorical"

        role = "Target" if target_name and col == target_name else "Feature"
        lines.append(f"  - {col}: {inferred_type} ({dtype}) [{role}]")

    return "\n".join(lines)


class GeminiCLIProcessor:
    def __init__(
        self,
        prompt_file: str,
        input_dir: str,
        output_dir: str,
        input_pattern: str = "*.txt",
        output_naming: Literal["replace", "parent_dir", "same"] = "replace",
        output_replace_from: str = "summary",
        output_replace_to: str = "info",
        sleep_time: int = 3,
        cli_command: str = DEFAULT_GEMINI_CLI_COMMAND,
        max_retries: int = 5,
        timeout: int = 600,
        validate_json: bool = False,
    ):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.input_pattern = input_pattern
        self.output_naming = output_naming
        self.output_replace_from = output_replace_from
        self.output_replace_to = output_replace_to
        self.sleep_time = sleep_time
        self.cli_command = cli_command
        self.max_retries = max_retries
        self.timeout = timeout
        self.validate_json = validate_json

        with open(prompt_file, "r", encoding="utf-8") as f:
            self.prompt_content = f.read()

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _get_output_filename(self, input_path: Path) -> str:
        if self.output_naming == "parent_dir":
            parent_name = input_path.parent.name
            return f"{parent_name}.json"
        elif self.output_naming == "replace":
            return input_path.name.replace(
                self.output_replace_from, self.output_replace_to
            )
        else:
            return input_path.name

    def _remove_thought_blocks(self, text: str) -> str:
        cleaned = re.sub(r"thought\n.*?thought\n", "", text, flags=re.DOTALL)
        cleaned = cleaned.lstrip("\n")
        return cleaned

    def _remove_code_block_markers(self, text: str) -> str:
        pattern = r"```(?:\w+)?\n(.*?)```"
        match = re.search(pattern, text, re.DOTALL)

        if match:
            return match.group(1).strip()
        else:
            cleaned = re.sub(r"^```\w*\n?", "", text.strip())
            cleaned = re.sub(r"\n?```\s*$", "", cleaned)
            return cleaned.strip()

    def _run_gemini(self, full_input: str, label: str) -> Optional[str]:
        for attempt in range(1, self.max_retries + 1):
            attempt_start = time.time()
            print(f"[Gemini] Start: {label} (attempt {attempt}/{self.max_retries})")
            try:
                result = subprocess.run(
                    self.cli_command,
                    input=full_input,
                    text=True,
                    capture_output=True,
                    shell=True,
                    encoding="utf-8",
                    timeout=self.timeout,
                )

                if result.returncode != 0:
                    error_msg = result.stderr or result.stdout or "(no output)"
                    elapsed = time.time() - attempt_start
                    print(
                        f"[Gemini] Failed: {label} (attempt {attempt}/{self.max_retries}, {elapsed:.1f}s)"
                    )
                    print(
                        f"[attempt {attempt}/{self.max_retries}] Error ({label}):\n{error_msg}"
                    )
                else:
                    if not result.stdout.strip():
                        stderr_msg = result.stderr.strip() or "(no stderr)"
                        elapsed = time.time() - attempt_start
                        print(
                            f"[Gemini] Empty response: {label} (attempt {attempt}/{self.max_retries}, {elapsed:.1f}s)"
                        )
                        print(
                            f"[attempt {attempt}/{self.max_retries}] Empty response ({label}): "
                            f"stdout is empty. stderr: {stderr_msg}"
                        )
                    else:
                        output = self._remove_thought_blocks(result.stdout)
                        output = self._remove_code_block_markers(output)

                        if not output.strip():
                            elapsed = time.time() - attempt_start
                            print(
                                f"[Gemini] Empty after cleanup: {label} (attempt {attempt}/{self.max_retries}, {elapsed:.1f}s)"
                            )
                            print(
                                f"[attempt {attempt}/{self.max_retries}] Empty after cleanup ({label}): "
                                f"raw stdout (first 200 chars): {result.stdout[:200]!r}"
                            )
                        elif output.strip() == SAFE_STOP_ERROR_TEXT:
                            elapsed = time.time() - attempt_start
                            print(
                                f"[Gemini] Safe-stop: {label} (attempt {attempt}/{self.max_retries}, {elapsed:.1f}s)"
                            )
                            return output.strip()
                        elif not self.validate_json:
                            elapsed = time.time() - attempt_start
                            print(
                                f"[Gemini] Success: {label} (attempt {attempt}/{self.max_retries}, {elapsed:.1f}s)"
                            )
                            return output
                        else:
                            try:
                                json.loads(output)
                                elapsed = time.time() - attempt_start
                                print(
                                    f"[Gemini] Success: {label} (attempt {attempt}/{self.max_retries}, {elapsed:.1f}s)"
                                )
                                return output
                            except json.JSONDecodeError as je:
                                elapsed = time.time() - attempt_start
                                print(
                                    f"[Gemini] Invalid JSON: {label} (attempt {attempt}/{self.max_retries}, {elapsed:.1f}s)"
                                )
                                print(
                                    f"[attempt {attempt}/{self.max_retries}] JSON parse error ({label}): {je}"
                                )

            except subprocess.TimeoutExpired:
                elapsed = time.time() - attempt_start
                print(
                    f"[Gemini] Timeout: {label} (attempt {attempt}/{self.max_retries}, {elapsed:.1f}s)"
                )
                print(
                    f"[attempt {attempt}/{self.max_retries}] Timeout ({label}): exceeded {self.timeout} seconds"
                )
            except Exception as e:
                elapsed = time.time() - attempt_start
                print(
                    f"[Gemini] Exception: {label} (attempt {attempt}/{self.max_retries}, {elapsed:.1f}s)"
                )
                print(
                    f"[attempt {attempt}/{self.max_retries}] Exception ({label}): {e}"
                )

            if attempt < self.max_retries:
                wait = self.sleep_time * attempt
                print(f"  -> Retrying in {wait} seconds...")
                time.sleep(wait)

        print(
            f"Processing failed: {label} (tried {self.max_retries} times without success)"
        )
        return None

    def process_file(self, input_path: Path) -> Optional[str]:
        with open(input_path, "r", encoding="utf-8") as f:
            content = f.read()

        full_input = (
            self.prompt_content
            + "\n\n"
            + "Language requirement: Respond in English only."
            + "\n"
            + "For JSON outputs, all natural-language string values must be English."
            + "\n\n"
            + content
        )
        return self._run_gemini(full_input, label=input_path.name)

    def process_text(self, text: str) -> Optional[str]:
        full_input = (
            self.prompt_content
            + "\n\n"
            + "Language requirement: Respond in English only."
            + "\n"
            + "For JSON outputs, all natural-language string values must be English."
            + "\n\n"
            + text
        )
        return self._run_gemini(full_input, label="<text input>")

    def generate_and_run_python_script(
        self,
        input_text: str,
        output_dir: str | Path,
        script_filename: str = "preprocessing_script.py",
        required_output_files: list[str] | None = None,
    ) -> bool:
        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)

        script_code = self.process_text(input_text)
        if not script_code:
            print("Error: Failed to generate Python script via Gemini.")
            return False

        script_path = output_dir_path / script_filename
        script_path.write_text(script_code, encoding="utf-8")
        print(f"Generated script saved to: {script_path}")

        result = subprocess.run(
            [sys.executable, script_path.name],
            capture_output=True,
            text=True,
            cwd=output_dir_path,
        )

        if result.returncode != 0:
            print("Error during generated script execution:")
            print(result.stderr)
            print(result.stdout)
            return False

        print(result.stdout)

        for filename in required_output_files or []:
            expected_path = output_dir_path / filename
            if not expected_path.exists():
                print(f"Error: Required output file not found: {expected_path}")
                return False

        return True

    def process_all(self, skip_existing: bool = True) -> int:
        input_files = list(self.input_dir.glob(self.input_pattern))
        print(f"Number of target files: {len(input_files)}")

        processed_count = 0

        for input_path in input_files:
            output_filename = self._get_output_filename(input_path)
            output_path = self.output_dir / output_filename

            if skip_existing and output_path.exists():
                print(f"Skip: output file already exists -> {output_path}")
                continue

            result = self.process_file(input_path)

            if result:
                with open(output_path, "w", encoding="utf-8") as out:
                    out.write(result)
                processed_count += 1

            time.sleep(self.sleep_time)

        print(f"Processing complete. Check folder: {self.output_dir}")
        return processed_count


class AbstractionProcessor(GeminiCLIProcessor):
    def __init__(
        self,
        prompt_file: str = "./prompt/abstraction_prompt.md",
        input_dir: str = "./dataset_summary",
        output_dir: str = "./Generalized_dataset_info",
        sleep_time: int = 2,
    ):
        super().__init__(
            prompt_file=prompt_file,
            input_dir=input_dir,
            output_dir=output_dir,
            input_pattern="*.txt",
            output_naming="replace",
            output_replace_from="summary",
            output_replace_to="info",
            sleep_time=sleep_time,
            validate_json=False,
        )


class SignConstraintProcessor(GeminiCLIProcessor):
    def __init__(
        self,
        prompt_file: str = "./prompt/sc_prompt.md",
        input_dir: str = "./input_json",
        output_dir: str = "./sc_json",
        sleep_time: int = 2,
    ):
        super().__init__(
            prompt_file=prompt_file,
            input_dir=input_dir,
            output_dir=output_dir,
            input_pattern="*.json",
            output_naming="replace",
            output_replace_from="_input",
            output_replace_to="_sc",
            sleep_time=sleep_time,
            validate_json=True,
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process files using Gemini CLI.")
    parser.add_argument(
        "mode",
        choices=["abstraction", "sign_constraint"],
        help="Processing mode (abstraction, sign_constraint)",
    )

    args = parser.parse_args()

    if args.mode == "abstraction":
        processor = AbstractionProcessor()
    else:
        processor = SignConstraintProcessor()

    processor.process_all()
