import json
import re
from pathlib import Path
from typing import Dict

import pandas as pd

ASSETS_DIR = Path("dataset_assets")


def normalize_name(name: str) -> str:
    normalized = re.sub(r'[\\/*?:"<>|]', "", name)
    normalized = normalized.replace(" ", "_").lower().strip("_")
    return normalized


def resolve_target_name(
    dataset_name: str,
    preprocessed_df: pd.DataFrame,
    report_text: str,
) -> str:
    raw_data_path = ASSETS_DIR / "raw_data" / f"{dataset_name}_raw.csv"
    if raw_data_path.exists():
        raw_columns = pd.read_csv(raw_data_path, nrows=0).columns.tolist()
        if raw_columns:
            raw_target_name = raw_columns[-1]
            if raw_target_name in preprocessed_df.columns:
                return raw_target_name

    report_patterns = [
        r"^Target column name:\s*(.+?)\s*$",
        r"^Target column:\s*(.+?)\s*$",
        r"^Target variable:\s*(.+?)\s*$",
    ]
    for line in report_text.splitlines():
        stripped = line.strip()
        for pattern in report_patterns:
            match = re.match(pattern, stripped)
            if match:
                candidate = match.group(1).strip()
                if candidate in preprocessed_df.columns:
                    return candidate

    if len(preprocessed_df.columns) == 0:
        raise ValueError("The preprocessed dataset has no columns.")

    raise ValueError(
        "Failed to resolve the target column from the raw dataset header or preprocessing report. "
        f"Available preprocessed columns: {preprocessed_df.columns.tolist()}"
    )


class InputJSONBuilder:
    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        self.desc_dict: Dict[str, str] = {}

    @staticmethod
    def _parse_variable_info(
        variable_info_text: str,
    ) -> Dict[str, str]:
        desc_dict: Dict[str, str] = {}
        for line in variable_info_text.splitlines():
            line = line.strip()
            if not line or ":" not in line:
                continue
            name, desc = line.split(":", 1)
            name = name.strip()
            desc = desc.strip()
            if name:
                desc_dict[name] = desc
        return desc_dict

    def _get_feature_description(self, feature_name: str) -> str:
        if feature_name in self.desc_dict:
            return self.desc_dict[feature_name]

        sorted_keys = sorted(self.desc_dict.keys(), key=len, reverse=True)
        for original_key in sorted_keys:
            if feature_name.startswith(original_key):
                suffix = feature_name[len(original_key) :]
                if suffix.startswith("_"):
                    category_name = suffix[1:]
                    base_desc = self.desc_dict[original_key]
                    if base_desc:
                        return f"{base_desc} (Category: {category_name})"
                    return f"(Category: {category_name})"

        return ""

    @staticmethod
    def _extract_target_mapping(report_text: str) -> str | None:
        mapping_info: str | None = None
        semantics_info: str | None = None

        for line in report_text.splitlines():
            if "Target mapping:" in line and mapping_info is None:
                mapping_info = line.split("Target mapping:", 1)[1].strip()
            if "Target semantics:" in line and semantics_info is None:
                semantics_info = line.split("Target semantics:", 1)[1].strip()

        if mapping_info and semantics_info:
            return f"{mapping_info} | {semantics_info}"
        if mapping_info:
            return mapping_info
        if semantics_info:
            return semantics_info
        return None

    def create(
        self,
        feature_names: list[str],
        target_name: str,
        info_text: str,
        variable_info_text: str,
        variable_description_map: Dict[str, str],
        report_text: str,
        output_path: Path,
    ) -> Path:
        self.desc_dict = dict(variable_description_map or {})
        if not self.desc_dict:
            self.desc_dict = self._parse_variable_info(variable_info_text)

        target_mapping_info = self._extract_target_mapping(report_text)
        target_desc = self.desc_dict.get(target_name, "")

        if target_mapping_info:
            target_field = f"{target_name} ({target_mapping_info})"
        elif target_desc:
            target_field = f"{target_name} ({target_desc})"
        else:
            target_field = target_name

        features_list = []
        for feat in feature_names:
            if feat == target_name:
                continue
            features_list.append(
                {
                    "name": feat,
                    "description": self._get_feature_description(feat),
                    "sign_constraint": "",
                    "reason": "",
                }
            )

        output_data = {
            "info": info_text.strip(),
            "target": target_field,
            "variable_info": variable_info_text.strip(),
            "features": features_list,
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(output_data, indent=4, ensure_ascii=False), encoding="utf-8"
        )
        print(f"Success: Created JSON file -> {output_path}")
        print(f" - Number of processed features: {len(features_list)}")
        return output_path


def run_step3_create_input_json(dataset_name: str) -> bool:
    name = normalize_name(dataset_name)
    info_path = Path("Generalized_dataset_info") / f"{name}_info.txt"
    preprocessed_path = Path("preprocessed_dat") / name / "preprocessed.csv"
    report_path = Path("preprocessed_dat") / name / "preprocessing_report.txt"
    variable_info_asset_path = (
        ASSETS_DIR / "additional_variable_info" / f"{name}_additional_variable_info.txt"
    )
    variable_description_path = (
        ASSETS_DIR / "variable_description" / f"{name}_variable_description.json"
    )
    output_path = Path("input_json") / f"{name}_input.json"

    if not info_path.exists():
        print(f"Error: Abstraction file not found: {info_path}")
        print("  -> Please run Step 1 first.")
        return False
    if not preprocessed_path.exists():
        print(f"Error: Preprocessed file not found: {preprocessed_path}")
        print("  -> Please run Step 2 first.")
        return False
    if not report_path.exists():
        print(f"Error: Preprocessing report not found: {report_path}")
        print("  -> Please run Step 2 first.")
        return False
    df = pd.read_csv(preprocessed_path)

    info_text = info_path.read_text(encoding="utf-8")
    variable_info_text = (
        variable_info_asset_path.read_text(encoding="utf-8")
        if variable_info_asset_path.exists()
        else ""
    )
    variable_description_map = (
        json.loads(variable_description_path.read_text(encoding="utf-8"))
        if variable_description_path.exists()
        else {}
    )
    report_text = report_path.read_text(encoding="utf-8")

    try:
        resolved_target_name = resolve_target_name(
            dataset_name=name,
            preprocessed_df=df,
            report_text=report_text,
        )
    except ValueError as exc:
        print(f"Error: {exc}")
        print("  -> Please verify the preprocessing output and target column handling.")
        return False

    creator = InputJSONBuilder(dataset_name=name)
    creator.create(
        feature_names=df.columns.tolist(),
        target_name=resolved_target_name,
        info_text=info_text,
        variable_info_text=variable_info_text,
        variable_description_map=variable_description_map,
        report_text=report_text,
        output_path=output_path,
    )
    return True


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run step 3")
    parser.add_argument(
        "dataset_name",
        type=str,
        help="Dataset name (must be enclosed in double quotes)",
    )

    args = parser.parse_args()

    success = run_step3_create_input_json(dataset_name=args.dataset_name)
    if not success:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
