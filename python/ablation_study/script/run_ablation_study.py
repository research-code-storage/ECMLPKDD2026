from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ablation_constraints import load_preprocessed_dataset_with_constraints
from src.ablation_excel_writer import write_ablation_excel
from src.kernels import get_kernel_function
from src.pfgd_train import train_pfgd
from src.preprocessing import prepro_sckmtwo_blocks_light
from src.splitters import h34_gen_lmat_tra

METHODS: list[str] = ["SF", "Perm", "Flip", "gemini", "gpt", "claude"]
KERNELS: list[str] = ["lin", "rbf"]


def _configure_warning_filters() -> None:
    warnings.filterwarnings(
        "ignore",
        message="X does not have valid feature names, but LGBMClassifier was fitted with feature names",
        category=UserWarning,
    )


def _percv_acc(sco: np.ndarray, y: np.ndarray) -> float:
    sco = np.asarray(sco, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    n_valid = int(np.sum(y != 0))
    if n_valid == 0:
        return float("nan")
    return float(np.sum(sco * y > 0) / n_valid)


def run_kernel_percv(
    X_all: np.ndarray,
    y_all: np.ndarray,
    cvec_sgn: np.ndarray,
    lmat_tra: np.ndarray,
    mode_sgncon: str,
    typ_kern: str,
    *,
    eta_nlr: float,
    gam_rbf: float,
    lamn: float,
    b_prime: np.ndarray,
    s_prime: np.ndarray,
    gam_sm: float,
    nepochs: int,
    qpmeth1: str,
) -> np.ndarray:
    ncvs_rnd = lmat_tra.shape[0]
    fh_kern = get_kernel_function(typ_kern, eta_nlr=eta_nlr, gam_rbf=gam_rbf)
    accs = np.empty(ncvs_rnd, dtype=float)

    for cv in range(ncvs_rnd):
        l_tra = lmat_tra[cv]
        l_tst = ~l_tra
        X_tra, X_tst = X_all[:, l_tra], X_all[:, l_tst]
        y_tra, y_tst = y_all[l_tra], y_all[l_tst]

        ntras_cv = y_tra.size
        lam1 = lamn / ntras_cv

        cvec_for_prepro = np.zeros_like(cvec_sgn) if mode_sgncon == "sf" else cvec_sgn

        K_xx, K_qx, c1, c2, sigvec1, K_qx_ekm, K_xx_ekm = prepro_sckmtwo_blocks_light(
            X_tra,
            y_tra,
            X_tst,
            fh_kern,
            cvec_for_prepro,
            s1=b_prime,
            s2=s_prime,
        )

        res = train_pfgd(
            K_xx=K_xx,
            K_qx=K_qx,
            c1=c1,
            c2=c2,
            sigvec1=sigvec1,
            lam1=lam1,
            gam_sm=gam_sm,
            qpmeth1=qpmeth1,
            nepochs=nepochs,
            mode_sgncon=mode_sgncon,
            verbose=0,
        )

        alph1 = res["alph1"]
        sigbeta1 = res["sigbeta1"]
        ilamn1 = 1.0 / (lam1 * ntras_cv)
        yalph1 = y_tra * alph1
        sco_tst = ilamn1 * (K_xx_ekm.T @ yalph1)
        if mode_sgncon == "sc":
            sco_tst = sco_tst + K_qx_ekm.T @ sigbeta1

        accs[cv] = _percv_acc(sco_tst, y_tst)

    return accs


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ablation study with gemini/gpt/claude constraints."
    )
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument(
        "--output-root", type=Path, default=PROJECT_ROOT / "out-ablation-study"
    )
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--ntras", nargs="*", type=int, default=[5, 10])
    parser.add_argument("--ncvs-rnd", type=int, default=200)
    parser.add_argument("--master-random-seed", type=int, default=20260308)
    return parser.parse_args()


def _discover_mat_paths(data_dir: Path) -> list[Path]:
    # Backward compatibility helper used by _discover_dataset_mats.
    return sorted(data_dir.glob("*.mat"))


def _discover_dataset_mats(data_dir: Path) -> dict[str, dict[str, Path]]:
    provider_dirs = {
        "gemini": data_dir / "gemini",
        "openai": data_dir / "openai",
        "anthropic": data_dir / "anthropic",
    }
    dataset_mats: dict[str, dict[str, Path]] = {}

    has_provider_layout = any(path.is_dir() for path in provider_dirs.values())
    if has_provider_layout:
        for provider, provider_dir in provider_dirs.items():
            mats = sorted(provider_dir.glob("*/*.mat"))
            seen_stems: set[str] = set()
            for mat_path in mats:
                stem = mat_path.stem
                if stem in seen_stems:
                    raise ValueError(
                        f"Duplicate dataset '{stem}' found under {provider_dir} across multiple model folders."
                    )
                seen_stems.add(stem)
                dataset_mats.setdefault(stem, {})[provider] = mat_path
        return dataset_mats

    for mat_path in _discover_mat_paths(data_dir):
        dataset_mats[mat_path.stem] = {"gemini": mat_path}
    return dataset_mats


def _validate_required_provider_mats(dataset_mats: dict[str, dict[str, Path]]) -> None:
    required_providers = ("gemini", "openai", "anthropic")
    missing_by_dataset: dict[str, list[str]] = {}

    for dataset_name, mats in dataset_mats.items():
        missing = [
            provider for provider in required_providers if mats.get(provider) is None
        ]
        if missing:
            missing_by_dataset[dataset_name] = missing

    if missing_by_dataset:
        details = "; ".join(
            f"{dataset}: missing {', '.join(missing)}"
            for dataset, missing in sorted(missing_by_dataset.items())
        )
        raise FileNotFoundError(
            "Missing required provider .mat files for ablation study (gemini/openai/anthropic). "
            f"{details}"
        )


def main() -> None:
    args = _parse_args()
    _configure_warning_filters()
    np.random.seed(0)

    data_dir = args.data_dir
    dataset_mats = _discover_dataset_mats(data_dir)
    if args.datasets:
        wanted = set(args.datasets)
        dataset_mats = {
            name: mats for name, mats in dataset_mats.items() if name in wanted
        }
    if not dataset_mats:
        raise FileNotFoundError(f"No dataset .mat files found under: {data_dir}")
    _validate_required_provider_mats(dataset_mats)

    v_ntras: list[int] = list(args.ntras)
    datasets = sorted(dataset_mats.keys())

    scale = 10.0
    bias_factor = 10.0
    lamn = 1.0
    delta = 1.0
    eta_nlr = 0.8
    gam_rbf = 0.001
    b_prime = np.array([1.0, -1.0])
    s_prime = np.array([delta, 0.0])
    gam_sm = 0.01
    nepochs = 1000
    qpmeth1 = "cqkp"
    rng_init = 0

    results: dict[str, dict[str, dict[int, dict[str, np.ndarray]]]] = {
        kernel: {method: {ntras: {} for ntras in v_ntras} for method in METHODS}
        for kernel in KERNELS
    }
    metadata_summary: dict[str, dict[str, object]] = {}

    timestamp = datetime.now().strftime("%y%m%d%H%M%S")
    output_dir = args.output_root / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(Path(__file__), output_dir / Path(__file__).name)
    print(f"Output directory: {output_dir}")

    total_t0 = time.time()
    dataset_seed_rng = np.random.default_rng(args.master_random_seed)

    for i_db, dataset_name in enumerate(datasets, start=1):
        mats = dataset_mats[dataset_name]
        mat_path = mats.get("gemini") or mats.get("openai") or mats.get("anthropic")
        if mat_path is None:
            raise FileNotFoundError(f"No MAT file found for dataset: {dataset_name}")

        dataset_seed = int(dataset_seed_rng.integers(0, np.iinfo(np.int64).max))
        print(f"\n[{i_db:2d}/{len(datasets):2d}] {dataset_name}")

        bundle = load_preprocessed_dataset_with_constraints(
            mat_path,
            scale=scale,
            bias_factor=bias_factor,
            master_random_seed=dataset_seed,
            gpt_mat_path=mats.get("openai"),
            claude_mat_path=mats.get("anthropic"),
        )
        X_all = np.asarray(bundle["X_all"], dtype=float)
        y_all = np.asarray(bundle["y_all"], dtype=float).reshape(-1)
        constraints = bundle["constraints"]
        metadata_summary[dataset_name] = dict(bundle["metadata"])

        for ntras in v_ntras:
            np.random.seed(rng_init)
            lmat_tra = h34_gen_lmat_tra(y_all, ntras, args.ncvs_rnd)

            common_kw = dict(
                eta_nlr=eta_nlr,
                gam_rbf=gam_rbf,
                lamn=lamn,
                b_prime=b_prime,
                s_prime=s_prime,
                gam_sm=gam_sm,
                nepochs=nepochs,
                qpmeth1=qpmeth1,
            )

            for kernel in KERNELS:
                for method in METHODS:
                    cvec_sgn = np.asarray(constraints[method], dtype=float)
                    mode_sgncon = "sf" if method == "SF" else "sc"
                    t0 = time.time()
                    accs = run_kernel_percv(
                        X_all,
                        y_all,
                        cvec_sgn,
                        lmat_tra,
                        mode_sgncon,
                        kernel,
                        **common_kw,
                    )
                    elapsed = time.time() - t0
                    results[kernel][method][ntras][dataset_name] = accs
                    print(
                        f"  n={ntras:3d}  {kernel:3s}  {method:7s}  "
                        f"acc={np.nanmean(accs):.4f}±{np.nanstd(accs):.4f}  ({elapsed:.1f}s)"
                    )

    total_elapsed = time.time() - total_t0
    print(f"\nAll experiments done in {total_elapsed:.0f}s.")

    npz_path = output_dir / "percv_accs_ablation.npz"
    flat: dict[str, np.ndarray] = {}
    for kernel in KERNELS:
        for method in METHODS:
            for ntras in v_ntras:
                for dataset in datasets:
                    flat[f"{kernel}__{method}__n{ntras}__{dataset}"] = results[kernel][
                        method
                    ][ntras][dataset]
    np.savez_compressed(npz_path, **flat)
    print(f"Raw accs saved: {npz_path}")

    meta_path = output_dir / "ablation_metadata.json"
    meta_path.write_text(
        json.dumps(metadata_summary, indent=2, default=str), encoding="utf-8"
    )
    print(f"Metadata saved: {meta_path}")

    xlsx_path = write_ablation_excel(
        results=results,
        kernels=KERNELS,
        methods=METHODS,
        v_ntras=v_ntras,
        datasets=datasets,
        output_path=output_dir / "result_ablation.xlsx",
    )
    print(f"Excel saved:    {xlsx_path}")


if __name__ == "__main__":
    main()
