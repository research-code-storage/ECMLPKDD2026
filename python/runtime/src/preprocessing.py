import numpy as np


def prepro_sckmtwo_blocks_light(
    X_tra, y_tra, X_tst, fh_kern, cvec_sgn, s1, s2, block_size=256
):
    nfeas, ntras = X_tra.shape
    _, ntsts = X_tst.shape
    assert X_tst.shape[0] == nfeas
    assert y_tra.shape == (ntras,)
    assert np.all(np.abs(y_tra) == 1)

    l_con = cvec_sgn != 0
    sigvec1 = cvec_sgn[l_con]
    ncons = int(np.sum(l_con))

    s1 = np.asarray(s1)
    s2 = np.asarray(s2)
    assert s1.shape == s2.shape
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


def prepro_sckmtwo_blocks(
    X_tra, y_tra, X_tst, fh_kern, cvec_sgn, s1, s2, block_size=256
):
    nfeas, ntras = X_tra.shape
    _, ntsts = X_tst.shape
    assert X_tst.shape[0] == nfeas
    assert y_tra.shape == (ntras,)
    assert np.all(np.abs(y_tra) == 1)

    l_con = cvec_sgn != 0
    sigvec1 = cvec_sgn[l_con]
    ncons = np.sum(l_con)

    assert s1.shape == s2.shape

    active_idx = np.flatnonzero(l_con)
    k = len(s1)

    def build_q_cols(indices):
        nb = len(indices)
        Q = np.zeros((nfeas, nb * k), dtype=X_tra.dtype)
        rows = active_idx[indices]
        for pos, row in enumerate(rows):
            base = pos * k
            for p in range(k):
                Q[row, base + p] = s2[p]
        return Q

    K_xx = fh_kern(X_tra, X_tra)
    K_xx_ekm = fh_kern(X_tra, X_tst)

    K_qx = np.zeros((ncons, ntras), dtype=K_xx.dtype)
    K_qq = np.zeros((ncons, ncons), dtype=K_xx.dtype)
    K_qx_ekm = np.zeros((ncons, ntsts), dtype=K_xx.dtype)

    for i0 in range(0, ncons, block_size):
        i1 = min(i0 + block_size, ncons)
        bi = i1 - i0
        idx_i = np.arange(i0, i1)

        Q_i = build_q_cols(idx_i)

        K_i_x = fh_kern(Q_i, X_tra).reshape(bi, k, ntras)
        K_qx_i = np.einsum("p,ipj->ij", s1, K_i_x)
        K_qx[i0:i1, :] = K_qx_i

        K_i_xt = fh_kern(Q_i, X_tst).reshape(bi, k, ntsts)
        K_qx_ekm[i0:i1, :] = np.einsum("p,ipj->ij", s1, K_i_xt)

        for j0 in range(i0, ncons, block_size):
            j1 = min(j0 + block_size, ncons)
            bj = j1 - j0
            idx_j = np.arange(j0, j1)

            Q_j = build_q_cols(idx_j)
            K_ij = fh_kern(Q_i, Q_j).reshape(bi, k, bj, k)
            K_qq_ij = np.einsum("p,q,ipjq->ij", s1, s1, K_ij)

            K_qq[i0:i1, j0:j1] = K_qq_ij
            if j0 != i0:
                K_qq[j0:j1, i0:i1] = K_qq_ij.T

    K_xx = (y_tra[:, None] * K_xx) * y_tra[None, :]
    K_qx = K_qx * y_tra[None, :]

    return K_xx, K_qx, K_qq, sigvec1, K_qx_ekm, K_xx_ekm


def prepro_sckmtwo(X_tra, y_tra, X_tst, fh_kern, cvec_sgn, s1, s2, block_size=256):
    K_xx, K_qx, K_qq, sigvec1, K_qx_ekm, K_xx_ekm = prepro_sckmtwo_blocks(
        X_tra,
        y_tra,
        X_tst,
        fh_kern,
        cvec_sgn,
        s1,
        s2,
        block_size=block_size,
    )

    ntras = K_xx.shape[0]
    ncons = K_qx.shape[0]
    n_all = ntras + ncons

    kernmat1_big = np.zeros((n_all, n_all), dtype=K_xx.dtype)
    kernmat1_big[:ntras, :ntras] = K_xx
    kernmat1_big[ntras:, :ntras] = K_qx
    kernmat1_big[:ntras, ntras:] = K_qx.T
    kernmat1_big[ntras:, ntras:] = K_qq

    return kernmat1_big, sigvec1, K_qx_ekm, K_xx_ekm
