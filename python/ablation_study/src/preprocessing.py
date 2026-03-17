import numpy as np


def prepro_sckmtwo_blocks_light(
    X_tra,
    y_tra,
    X_tst,
    fh_kern,
    cvec_sgn,
    s1,
    s2,
    block_size=256,
):
    nfeas, ntras = X_tra.shape
    _, ntsts = X_tst.shape
    if X_tst.shape[0] != nfeas:
        raise ValueError("X_tra and X_tst row size mismatch")
    if y_tra.shape != (ntras,):
        raise ValueError("y_tra shape mismatch")

    l_con = cvec_sgn != 0
    sigvec1 = cvec_sgn[l_con].astype(float)
    ncons = int(np.sum(l_con))

    s1 = np.asarray(s1, dtype=float)
    s2 = np.asarray(s2, dtype=float)
    if s1.shape != s2.shape:
        raise ValueError("s1 and s2 must have same shape")
    len1 = len(s1)
    len2 = len(s2)
    active_idx = np.flatnonzero(l_con)

    def build_q_cols(indices):
        nb = len(indices)
        Q = np.zeros((nfeas, nb * len2), dtype=X_tra.dtype)
        rows = active_idx[indices]
        for pos, row in enumerate(rows):
            base = pos * len2
            for p in range(len2):
                Q[row, base + p] = s2[p]
        return Q

    if ncons > 0:
        Q_first = build_q_cols(np.array([0], dtype=int))
        q_b1 = Q_first[:, :len2]
        k_11 = fh_kern(q_b1, q_b1)
        c1 = float(s1 @ k_11 @ s1.T)
        if ncons >= 2:
            Q_second = build_q_cols(np.array([1], dtype=int))
            q_b2 = Q_second[:, :len2]
            k_12 = fh_kern(q_b1, q_b2)
            c2 = float(s1 @ k_12 @ s1.T)
        else:
            c2 = 0.0
    else:
        c1 = 0.0
        c2 = 0.0

    K_xx = fh_kern(X_tra, X_tra)
    K_xx_ekm = fh_kern(X_tra, X_tst)

    K_qx = np.zeros((ncons, ntras), dtype=K_xx.dtype)
    K_qx_ekm = np.zeros((ncons, ntsts), dtype=K_xx.dtype)

    for i0 in range(0, ncons, block_size):
        i1 = min(i0 + block_size, ncons)
        bi = i1 - i0
        idx_i = np.arange(i0, i1)

        Q_i = build_q_cols(idx_i)

        K_i_x = fh_kern(Q_i, X_tra).reshape(bi, len1, ntras)
        K_qx_i = np.einsum("p,ipj->ij", s1, K_i_x)
        K_qx[i0:i1, :] = K_qx_i

        K_i_xt = fh_kern(Q_i, X_tst).reshape(bi, len1, ntsts)
        K_qx_ekm[i0:i1, :] = np.einsum("p,ipj->ij", s1, K_i_xt)

    K_xx = (y_tra[:, None] * K_xx) * y_tra[None, :]
    K_qx = K_qx * y_tra[None, :]

    return K_xx, K_qx, c1, c2, sigvec1, K_qx_ekm, K_xx_ekm
