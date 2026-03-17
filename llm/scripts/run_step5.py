from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROVIDER_ORDER = ("openai", "anthropic", "gemini")


@dataclass(frozen=True)
class ConstraintArtifact:
    provider: str
    model: str
    path: Path
    vector: np.ndarray

    @property
    def label(self) -> str:
        return f"{self.provider}:{self.model}"

    @property
    def mat_output_dir(self) -> Path:
        return Path("mat_data") / self.provider / self.model


def normalize_name(name: str) -> str:
    normalized = re.sub(r'[\\/*?:"<>|]', "", name)
    normalized = normalized.replace(" ", "_").lower().strip("_")
    return normalized


def load_input_json(dataset_name: str, input_json_dir: Path) -> dict:
    input_path = input_json_dir / f"{dataset_name}_input.json"
    if not input_path.exists():
        raise FileNotFoundError(f"input_json file not found: {input_path}")
    return json.loads(input_path.read_text(encoding="utf-8"))


def resolve_target_name(
    dataset_name: str,
    input_json_dir: Path,
    df_columns: list[str] | None = None,
) -> str | None:
    try:
        input_data = load_input_json(dataset_name=dataset_name, input_json_dir=input_json_dir)
        target_field = str(input_data.get("target", "")).strip()
        if target_field:
            inferred = target_field.split(" (", 1)[0].strip()
            if inferred:
                return inferred
    except Exception:
        pass

    if df_columns:
        return df_columns[-1]
    return None


def create_cvec_sgn(feature_names: list[str], sc_json_path: Path) -> np.ndarray:
    if not sc_json_path.exists():
        raise FileNotFoundError(f"Sign-constraint JSON file not found: {sc_json_path}")

    sign_constraints = json.loads(sc_json_path.read_text(encoding="utf-8"))
    features = sign_constraints.get("features", [])

    cvec_sgn: list[float] = []
    missing_features: list[str] = []

    for feature_name in feature_names:
        matched = next((item for item in features if item.get("name") == feature_name), None)
        if matched is None:
            missing_features.append(feature_name)
            continue
        cvec_sgn.append(float(matched.get("sign_constraint", 0)))

    if missing_features:
        raise KeyError(
            f"The following features were not found in the JSON file: {missing_features}"
        )

    return np.array(cvec_sgn, dtype=np.float64)


def discover_constraint_artifacts(dataset_name: str, sc_root: Path, feature_names: list[str]) -> list[ConstraintArtifact]:
    artifacts: list[ConstraintArtifact] = []

    for provider in DEFAULT_PROVIDER_ORDER:
        provider_dir = sc_root / provider
        if not provider_dir.exists():
            continue

        for model_dir in sorted(path for path in provider_dir.iterdir() if path.is_dir()):
            sc_path = model_dir / f"{dataset_name}_sc.json"
            if not sc_path.exists():
                continue
            vector = create_cvec_sgn(feature_names=feature_names, sc_json_path=sc_path)
            artifacts.append(
                ConstraintArtifact(
                    provider=provider,
                    model=model_dir.name,
                    path=sc_path,
                    vector=vector,
                )
            )

    return artifacts


def save_mat_file(
    X: pd.DataFrame,
    y: np.ndarray,
    cvec_sgn: np.ndarray,
    output_path: Path,
    dataset_name: str,
) -> Path:
    output_path.mkdir(parents=True, exist_ok=True)
    file_path = output_path / f"uci_{dataset_name}.mat"

    data_to_save = {
        "X_dat": X.values,
        "y_dat": y,
        "cvec_sgn": cvec_sgn,
    }
    scipy.io.savemat(file_path, data_to_save)
    print(f"Saved MAT file: {file_path}")
    return file_path


def run_step5_export_mat(dataset_name: str) -> bool:
    name = normalize_name(dataset_name)
    preprocessed_path = Path("preprocessed_dat") / name / "preprocessed.csv"
    sc_root = Path("sc_json")

    if not preprocessed_path.exists():
        print(f"Error: Preprocessed file not found: {preprocessed_path}")
        print("  -> Please run Step 2 first.")
        return False
    if not sc_root.exists():
        print(f"Error: Sign-constraint directory not found: {sc_root}")
        print("  -> Please run Step 4 first.")
        return False

    df = pd.read_csv(preprocessed_path)
    resolved_target = resolve_target_name(
        dataset_name=name,
        input_json_dir=Path("input_json"),
        df_columns=df.columns.tolist(),
    )
    if resolved_target is None or resolved_target not in df.columns:
        print(f"Error: Target variable '{resolved_target}' does not exist in the data.")
        print(f"  Available columns: {df.columns.tolist()}")
        return False

    X = df.drop(columns=[resolved_target])
    y = df[resolved_target].values.reshape(1, -1)

    try:
        artifacts = discover_constraint_artifacts(
            dataset_name=name,
            sc_root=sc_root,
            feature_names=X.columns.tolist(),
        )
    except (FileNotFoundError, KeyError, ValueError) as exc:
        print(f"Error: {exc}")
        return False

    if not artifacts:
        print(f"Error: No sign-constraint JSON files were found for dataset '{name}'.")
        print("  -> Please run Step 4 first.")
        return False

    generated_mat_paths: list[Path] = []
    for artifact in artifacts:
        mat_path = save_mat_file(
            X=X,
            y=y,
            cvec_sgn=artifact.vector,
            output_path=artifact.mat_output_dir,
            dataset_name=name,
        )
        generated_mat_paths.append(mat_path)

    print("Export Summary:")
    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")
    print(f"  cvec_sgn shape: {(len(X.columns),)}")
    print(f"  Generated MAT files: {len(generated_mat_paths)}")
    for artifact, mat_path in zip(artifacts, generated_mat_paths, strict=True):
        print(f"    - {artifact.label}: {mat_path}")
    return True


def main() -> None:
    import argparse

    os.chdir(PROJECT_ROOT)

    parser = argparse.ArgumentParser(description="Run step 5")
    parser.add_argument(
        "dataset_name",
        type=str,
        help="Dataset name (must be enclosed in double quotes)",
    )

    args = parser.parse_args()

    success = run_step5_export_mat(dataset_name=args.dataset_name)
    if not success:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
