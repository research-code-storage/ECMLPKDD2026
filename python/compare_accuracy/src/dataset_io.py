from pathlib import Path

import numpy as np
from scipy.io import loadmat


def demo1315_get_dataset(data_dir: str, dbname1: str, verbose: bool = False) -> dict:
    file_in = Path(data_dir) / f"{dbname1}.mat"
    if not file_in.exists():
        raise FileNotFoundError(f"Dataset file not found: {file_in}")
    data = loadmat(file_in)
    required_keys = ["X_dat", "y_dat", "cvec_sgn"]
    for key in required_keys:
        if key not in data:
            raise KeyError(f"Missing key '{key}' in {file_in}")

    X_dat = data["X_dat"]
    y_dat = data["y_dat"]
    cvec_sgn = data["cvec_sgn"]

    X_all = X_dat.T.astype(float)
    y_all = y_dat.T.astype(float).reshape(-1)
    cvec_sgn = cvec_sgn.T.astype(float).reshape(-1)

    nfeas, npts = X_all.shape
    if y_all.shape[0] != npts:
        raise ValueError("y_all length mismatch")

    feanames_sel = [str(i + 1) for i in range(nfeas)]

    if verbose:
        n_pos = int(np.sum(y_all > 0))
        n_neg = int(np.sum(y_all < 0))
        n_zero = int(np.sum(y_all == 0))
        n_all = int(y_all.size)
        print(
            f"y_all balance: pos={n_pos} ({100*n_pos/n_all:.2f}%), "
            f"neg={n_neg} ({100*n_neg/n_all:.2f}%), "
            f"zero={n_zero} ({100*n_zero/n_all:.2f}%)"
        )

    return {
        "X_all": X_all,
        "y_all": y_all,
        "feanames_sel": feanames_sel,
        "cvec_sgn": cvec_sgn,
    }
