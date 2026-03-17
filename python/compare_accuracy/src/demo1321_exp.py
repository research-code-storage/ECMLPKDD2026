import numpy as np

from .dataset_io import demo1315_get_dataset


def _zscore_rows(X: np.ndarray) -> np.ndarray:
    mu = np.mean(X, axis=1, keepdims=True)
    sigma = np.std(X, axis=1, ddof=0, keepdims=True)
    sigma[sigma == 0.0] = 1.0
    return (X - mu) / sigma


def prepare_demo1321_dataset(
    param_dataset: dict,
    dbname1: str,
    scale: float,
    bias_factor: float,
) -> dict:
    X_all = np.asarray(param_dataset["X_all"], dtype=float)
    y_all = np.asarray(param_dataset["y_all"], dtype=float).reshape(-1)
    cvec_sgn = np.asarray(param_dataset["cvec_sgn"], dtype=float).reshape(-1)

    if any(k in dbname1 for k in ["uci", "harbor", "indian"]):
        _, npts = X_all.shape
        X_all = _zscore_rows(X_all)
        X_all = scale * X_all
        X_all = np.vstack([X_all, bias_factor * np.ones((1, npts))])
        cvec_sgn = np.concatenate([cvec_sgn, [0.0]])

    return {
        "X_all": X_all,
        "y_all": y_all,
        "cvec_sgn": cvec_sgn,
    }


def load_dataset_for_demo1321(
    data_dir: str, dbname1: str, verbose: bool = False
) -> dict:
    return demo1315_get_dataset(data_dir=data_dir, dbname1=dbname1, verbose=verbose)
