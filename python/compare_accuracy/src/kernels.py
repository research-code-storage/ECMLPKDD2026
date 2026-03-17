import numpy as np
from scipy.spatial.distance import cdist


def lin_kernel(X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
    return X1.T @ X2


def rbf_kernel(X1: np.ndarray, X2: np.ndarray, gamma: float) -> np.ndarray:
    dists = cdist(X1.T, X2.T, metric="sqeuclidean")
    return np.exp(-gamma * dists)


def get_kernel_function(typ_kern: str, eta_nlr: float, gam_rbf: float):
    def fh_kern(X_tra: np.ndarray, X_tst: np.ndarray) -> np.ndarray:
        if typ_kern == "lin":
            return lin_kernel(X_tra, X_tst)
        if typ_kern == "rbf":
            return rbf_kernel(X_tra, X_tst, gam_rbf)
        raise ValueError(f"Unknown kernel type: {typ_kern}")

    return fh_kern
