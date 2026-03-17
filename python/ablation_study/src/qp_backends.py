from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BoxQP:
    cd: float
    avec: np.ndarray
    sigvec: np.ndarray
    bound: float = 100.0


def make_box_qp(cd: float, avec: np.ndarray, sigvec: np.ndarray, bound: float = 100.0) -> BoxQP:
    avec = np.asarray(avec, dtype=np.float64).reshape(-1)
    sigvec = np.asarray(sigvec, dtype=np.float64).reshape(-1)
    if avec.shape != sigvec.shape:
        raise ValueError("avec and sigvec must have same shape")
    if cd <= 0:
        raise ValueError("cd must be positive")
    sigvec = np.where(sigvec >= 0.0, 1.0, -1.0)
    return BoxQP(cd=float(cd), avec=avec, sigvec=sigvec, bound=float(bound))
