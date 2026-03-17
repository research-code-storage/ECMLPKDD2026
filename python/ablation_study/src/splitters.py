import numpy as np


def h34_gen_lmat_tra(y_all: np.ndarray, ntras: int, ncvs_rnd: int) -> np.ndarray:
    y_all = np.asarray(y_all).copy().reshape(-1)
    if not np.all(np.abs(y_all) == 1):
        raise ValueError("y_all must be +/-1")

    y_all[y_all == -1] = 2
    npts = y_all.size
    nps_all = int(np.sum(y_all == 1))
    nns_all = int(npts - nps_all)
    ratio_tra = ntras / npts
    nps_ave = nps_all * ratio_tra
    nps_floor = int(np.floor(nps_ave))
    hasu1 = nps_ave - nps_floor

    n1mat = np.zeros((ncvs_rnd, 2), dtype=int)
    n1mat[:, 0] = (np.random.rand(ncvs_rnd) < hasu1).astype(int) + nps_floor
    n1mat[:, 1] = ntras - n1mat[:, 0]

    lmat_tra = np.zeros((ncvs_rnd, npts), dtype=bool)
    for cv_rnd in range(ncvs_rnd):
        for ct1, n2 in [(1, nps_all), (2, nns_all)]:
            r_ct = np.flatnonzero(y_all == ct1)
            n1 = int(n1mat[cv_rnd, ct1 - 1])
            selected = np.random.choice(n2, size=n1, replace=False)
            lmat_tra[cv_rnd, r_ct[selected]] = True
    return lmat_tra
