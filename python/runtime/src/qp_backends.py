from __future__ import annotations

from dataclasses import dataclass

import numpy as np

Bound = float


@dataclass(frozen=True)
class BoxQP:
    cd: float
    avec: np.ndarray
    sigvec: np.ndarray
    bound: Bound = 100.0

    @property
    def n(self) -> int:
        return int(self.avec.shape[0])

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        sig = self.sigvec
        lb = -self.bound * (sig == -1.0)
        ub = +self.bound * (sig == +1.0)
        return lb.astype(float), ub.astype(float)


def _as_f64_1d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError(f"Expected 1D array, got shape={x.shape}")
    return x


def make_box_qp(
    cd: float, avec: np.ndarray, sigvec: np.ndarray, bound: Bound = 100.0
) -> BoxQP:
    avec = _as_f64_1d(avec)
    sigvec = _as_f64_1d(sigvec)
    if avec.shape != sigvec.shape:
        raise ValueError(
            f"avec and sigvec must have same shape: {avec.shape} vs {sigvec.shape}"
        )
    if not np.all(np.isin(np.round(sigvec), (-1.0, 1.0))):
        raise ValueError("sigvec must be +/-1")
    sigvec = np.where(sigvec >= 0, 1.0, -1.0)
    if cd <= 0:
        raise ValueError("cd must be positive for the projection QP")
    return BoxQP(cd=float(cd), avec=avec, sigvec=sigvec, bound=float(bound))
