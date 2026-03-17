from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import loadmat


def _zscore_rows(X: np.ndarray) -> np.ndarray:
	mu = np.mean(X, axis=1, keepdims=True)
	sigma = np.std(X, axis=1, ddof=0, keepdims=True)
	sigma[sigma == 0.0] = 1.0
	return (X - mu) / sigma


def _mutable_indices_for_random_sc(cvec_sgn: np.ndarray, protect_last_bias: bool) -> np.ndarray:
	nfeas = cvec_sgn.size
	if protect_last_bias and nfeas > 0 and cvec_sgn[-1] == 0.0:
		return np.arange(nfeas - 1, dtype=int)
	return np.arange(nfeas, dtype=int)


def make_random_sign_constraints(
	cvec_sgn: np.ndarray,
	*,
	variant: str,
	seed: int,
	protect_last_bias: bool = True,
) -> tuple[np.ndarray, dict[str, np.ndarray | int | bool | str]]:
	sigma_orig = np.asarray(cvec_sgn, dtype=float).reshape(-1)
	sigma_rand = sigma_orig.copy()
	mutable_idx = _mutable_indices_for_random_sc(sigma_orig, protect_last_bias)
	rng = np.random.default_rng(seed)

	perm_full = np.arange(sigma_orig.size, dtype=np.int64)
	flip_mask = np.zeros(sigma_orig.size, dtype=bool)

	if variant == "perm":
		permuted_idx = rng.permutation(mutable_idx)
		sigma_rand[mutable_idx] = sigma_orig[permuted_idx]
		perm_full[mutable_idx] = permuted_idx
	elif variant == "flip":
		flip_local = rng.random(mutable_idx.size) < 0.5
		flip_mask[mutable_idx] = flip_local
		sigma_rand[mutable_idx] = sigma_orig[mutable_idx] * np.where(flip_local, -1.0, 1.0)
	else:
		raise ValueError(f"Unknown random sign-constraint variant: {variant}")

	if protect_last_bias and sigma_orig.size > 0 and sigma_orig[-1] == 0.0:
		sigma_rand[-1] = 0.0
		perm_full[-1] = sigma_orig.size - 1
		flip_mask[-1] = False

	meta: dict[str, np.ndarray | int | bool | str] = {
		"variant": variant,
		"seed": int(seed),
		"protect_last_bias": bool(protect_last_bias),
		"mutable_indices": mutable_idx.astype(np.int64),
		"sigma_orig": sigma_orig,
		"sigma_rand": sigma_rand,
		"perm_indices": perm_full,
		"flip_mask": flip_mask,
	}
	return sigma_rand, meta


def summarize_random_sc(meta: dict[str, np.ndarray | int | bool | str]) -> dict[str, object]:
	sigma_orig = np.asarray(meta["sigma_orig"], dtype=float)
	sigma_rand = np.asarray(meta["sigma_rand"], dtype=float)
	perm_indices = np.asarray(meta["perm_indices"], dtype=np.int64)
	flip_mask = np.asarray(meta["flip_mask"], dtype=bool)
	mutable_idx = np.asarray(meta["mutable_indices"], dtype=np.int64)

	def _counts(vec: np.ndarray) -> tuple[int, int, int]:
		n_pos = int(np.sum(vec > 0))
		n_neg = int(np.sum(vec < 0))
		n_zero = int(np.sum(vec == 0))
		return n_pos, n_neg, n_zero

	n_pos_orig, n_neg_orig, n_zero_orig = _counts(sigma_orig)
	n_pos_rand, n_neg_rand, n_zero_rand = _counts(sigma_rand)

	return {
		"variant": str(meta["variant"]),
		"seed": int(meta["seed"]),
		"protect_last_bias": bool(meta["protect_last_bias"]),
		"n_features": int(sigma_orig.size),
		"n_mutable": int(mutable_idx.size),
		"n_nonzero_orig": int(np.sum(sigma_orig != 0.0)),
		"n_pos_orig": n_pos_orig,
		"n_neg_orig": n_neg_orig,
		"n_zero_orig": n_zero_orig,
		"n_nonzero_rand": int(np.sum(sigma_rand != 0.0)),
		"n_pos_rand": n_pos_rand,
		"n_neg_rand": n_neg_rand,
		"n_zero_rand": n_zero_rand,
		"n_positions_changed": int(np.sum(perm_indices != np.arange(sigma_orig.size))),
		"n_flipped_entries": int(np.sum(flip_mask)),
	}


def _append_bias_constraint(cvec: np.ndarray) -> np.ndarray:
	return np.concatenate([np.asarray(cvec, dtype=float).reshape(-1), [0.0]])


def _should_add_bias(dataset_name: str) -> bool:
	return any(k in dataset_name for k in ["uci", "harbor", "indian"])


def load_preprocessed_dataset_with_constraints(
	mat_path: Path,
	*,
	scale: float,
	bias_factor: float,
	master_random_seed: int,
	gpt_mat_path: Path | None = None,
	claude_mat_path: Path | None = None,
) -> dict[str, object]:
	data = loadmat(mat_path)
	required_keys = ["X_dat", "y_dat", "cvec_sgn"]
	for key in required_keys:
		if key not in data:
			raise KeyError(f"Missing key '{key}' in {mat_path}")

	dataset_name = mat_path.stem
	X_dat = np.asarray(data["X_dat"], dtype=float)
	y_dat = np.asarray(data["y_dat"], dtype=float)
	gemini_raw = np.asarray(data["cvec_sgn"], dtype=float)

	X_all = X_dat.T.astype(float)
	y_all = y_dat.T.astype(float).reshape(-1)
	gemini_raw = gemini_raw.T.astype(float).reshape(-1)

	raw_feature_count = int(X_all.shape[0])
	if gemini_raw.size != raw_feature_count:
		raise ValueError(
			f"Gemini constraint length mismatch for {dataset_name}: "
			f"features={raw_feature_count}, cvec_sgn={gemini_raw.size}"
		)

	if gpt_mat_path is not None:
		gpt_data = loadmat(gpt_mat_path)
		if "cvec_sgn" not in gpt_data:
			raise KeyError(f"Missing key 'cvec_sgn' in {gpt_mat_path}")
		gpt_raw = np.asarray(gpt_data["cvec_sgn"], dtype=float).T.astype(float).reshape(-1)
		if gpt_raw.size != raw_feature_count:
			raise ValueError(
				f"GPT constraint length mismatch for {dataset_name}: "
				f"features={raw_feature_count}, cvec_sgn={gpt_raw.size}"
			)
	else:
		gpt_raw = gemini_raw.copy()

	if claude_mat_path is not None:
		claude_data = loadmat(claude_mat_path)
		if "cvec_sgn" not in claude_data:
			raise KeyError(f"Missing key 'cvec_sgn' in {claude_mat_path}")
		claude_raw = np.asarray(claude_data["cvec_sgn"], dtype=float).T.astype(float).reshape(-1)
		if claude_raw.size != raw_feature_count:
			raise ValueError(
				f"Claude constraint length mismatch for {dataset_name}: "
				f"features={raw_feature_count}, cvec_sgn={claude_raw.size}"
			)
	else:
		claude_raw = gemini_raw.copy()

	if _should_add_bias(dataset_name):
		_, npts = X_all.shape
		X_all = _zscore_rows(X_all)
		X_all = scale * X_all
		X_all = np.vstack([X_all, bias_factor * np.ones((1, npts))])

		gemini = _append_bias_constraint(gemini_raw)
		gpt = _append_bias_constraint(gpt_raw)
		claude = _append_bias_constraint(claude_raw)
	else:
		gemini = gemini_raw.copy()
		gpt = gpt_raw.copy()
		claude = claude_raw.copy()

	rng = np.random.default_rng(master_random_seed)
	perm_seed = int(rng.integers(0, np.iinfo(np.int64).max))
	flip_seed = int(rng.integers(0, np.iinfo(np.int64).max))
	perm, perm_meta = make_random_sign_constraints(gemini, variant="perm", seed=perm_seed)
	flip, flip_meta = make_random_sign_constraints(gemini, variant="flip", seed=flip_seed)
	sf = np.zeros_like(gemini)

	return {
		"dataset_name": dataset_name,
		"X_all": X_all,
		"y_all": y_all,
		"constraints": {
			"SF": sf,
			"Perm": perm,
			"Flip": flip,
			"gemini": gemini,
			"gpt": gpt,
			"claude": claude,
		},
		"metadata": {
			"raw_feature_count": raw_feature_count,
			"preprocessed_feature_count": int(X_all.shape[0]),
			"constraint_mat_paths": {
				"gemini": str(mat_path),
				"gpt": str(gpt_mat_path) if gpt_mat_path is not None else str(mat_path),
				"claude": str(claude_mat_path) if claude_mat_path is not None else str(mat_path),
			},
			"random_seeds": {
				"Perm": perm_seed,
				"Flip": flip_seed,
			},
			"random_summary": {
				"Perm": summarize_random_sc(perm_meta),
				"Flip": summarize_random_sc(flip_meta),
			},
		},
	}
