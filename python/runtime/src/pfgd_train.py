import time

import numpy as np

import losses

try:
    import pfgd_solvers
except ImportError:
    pfgd_solvers = None

from qp_backends import make_box_qp


def _parse_kernel_input(kern_input, sigvec1):
    ncons = len(sigvec1)

    if isinstance(kern_input, dict):
        K_xx = kern_input["K_xx"]
        K_qx = kern_input["K_qx"]
        if "K_qq" in kern_input:
            K_qq = kern_input["K_qq"]
            c1 = float(K_qq[0, 0]) if len(sigvec1) >= 1 else 0.0
            c2 = float(K_qq[0, 1]) if len(sigvec1) >= 2 else 0.0
            return K_xx, K_qx, K_qq, c1, c2
        c1 = float(kern_input["c1"])
        c2 = float(kern_input["c2"])
        return K_xx, K_qx, None, c1, c2

    kernmat1_big = kern_input
    ntras = kernmat1_big.shape[0] - ncons
    r_tra = slice(0, ntras)
    r_q = slice(ntras, ntras + ncons)
    K_xx = kernmat1_big[r_tra, r_tra]
    K_qx = kernmat1_big[r_q, r_tra]
    K_qq = kernmat1_big[r_q, r_q]
    c1 = float(K_qq[0, 0]) if ncons >= 1 else 0.0
    c2 = float(K_qq[0, 1]) if ncons >= 2 else 0.0
    return K_xx, K_qx, K_qq, c1, c2


def _apply_kqq(v, c1, c2):
    return (c1 - c2) * v + c2 * np.sum(v)


def demo1334_proj(
    alph1_half,
    beta1_half,
    lam1,
    sigvec1,
    kernmat1_qx,
    qpmeth1,
    c1=None,
    c2=None,
    kernmat1_qq=None,
):
    ncons, ntras = kernmat1_qx.shape
    lamn1 = lam1 * ntras
    ilamn1 = 1.0 / lamn1

    if ncons == 0:
        return np.array([])

    if c1 is None or c2 is None:
        if kernmat1_qq is None:
            raise ValueError("Either (c1,c2) or kernmat1_qq must be provided.")
        c2 = float(kernmat1_qq[0, 1]) if ncons >= 2 else 0.0
        c1 = float(kernmat1_qq[0, 0]) if ncons >= 1 else 0.0

    if abs(c2) != 0.0:
        cd = c2 / (c1 - c2)
        kqq_v = _apply_kqq(sigvec1 * beta1_half, c1, c2)
        avec1 = -(ilamn1 * kernmat1_qx @ alph1_half + kqq_v) / (c1 - c2)
        r_srt = np.argsort(avec1)
        avec1_srt = avec1[r_srt]
        sigvec1_srt = sigvec1[r_srt]

        qp = make_box_qp(cd, avec1_srt, sigvec1_srt)
        qpm = str(qpmeth1).lower()
        if qpm in {"cqkp", "proxqp", "cvxopt"}:
            if pfgd_solvers is None:
                raise ImportError(
                    "Cython module 'pfgd_solvers' is not available. "
                    "Build it with: (cd demo1332_01_python/src && python setup.py build_ext --inplace)"
                )
            if qpm == "cqkp":
                xvec1_star_srt = pfgd_solvers.solve_cqkp(cd, qp.avec, qp.sigvec)
            elif qpm == "proxqp":
                xvec1_star_srt = pfgd_solvers.solve_proxqp(cd, qp.avec, qp.sigvec)
            elif qpm == "cvxopt":
                xvec1_star_srt = pfgd_solvers.solve_cvxopt(cd, qp.avec, qp.sigvec)
        else:
            raise ValueError("Unknown qpmeth1. Supported: cqkp, proxqp, cvxopt")
        sigbeta1 = np.zeros(ncons)
        sigbeta1[r_srt] = np.asarray(xvec1_star_srt, dtype=np.float64).reshape(-1)

    else:
        ic1 = 1.0 / c1
        avec1 = -ic1 * ilamn1 * (sigvec1 * (kernmat1_qx @ alph1_half))
        beta1 = np.maximum(avec1, beta1_half)
        sigbeta1 = sigvec1 * (beta1 - beta1_half)

    return sigbeta1


def demo1334_get_obj(
    alph1,
    sigbeta1,
    K_xx,
    K_qx,
    K_qq,
    sigvec1,
    lam1,
    fh_loss_p,
    fh_grad_p,
    fh_loss_d,
    qpmeth1_proj="cqkp",
    c1=None,
    c2=None,
):
    ncons = len(sigvec1)
    ntras = K_xx.shape[0]
    lamn1 = lam1 * ntras
    ilamn1 = 1.0 / lamn1
    if (c1 is None or c2 is None) and K_qq is not None:
        c1 = float(K_qq[0, 0]) if ncons >= 1 else 0.0
        c2 = float(K_qq[0, 1]) if ncons >= 2 else 0.0
    if c1 is None or c2 is None:
        raise ValueError("Either K_qq or (c1,c2) must be provided.")

    sco2s_tra = ilamn1 * (K_xx @ alph1) + K_qx.T @ sigbeta1
    if K_qq is None:
        sco2s_q = ilamn1 * (K_qx @ alph1) + _apply_kqq(sigbeta1, c1, c2)
    else:
        sco2s_q = ilamn1 * (K_qx @ alph1) + K_qq @ sigbeta1
    ualph1 = -fh_grad_p(sco2s_tra)
    usigbeta1 = demo1334_proj(
        ualph1,
        np.zeros(ncons),
        lam1,
        sigvec1,
        K_qx,
        qpmeth1_proj,
        c1=c1,
        c2=c2,
        kernmat1_qq=K_qq,
    )

    sco1s_tra = ilamn1 * (K_xx @ ualph1) + K_qx.T @ usigbeta1
    if K_qq is None:
        sco1s_q = ilamn1 * (K_qx @ ualph1) + _apply_kqq(usigbeta1, c1, c2)
    else:
        sco1s_q = ilamn1 * (K_qx @ ualph1) + K_qq @ usigbeta1
    reg_p = (
        0.5 * lam1 * (np.dot(ilamn1 * ualph1, sco1s_tra) + np.dot(usigbeta1, sco1s_q))
    )

    loss_p = np.mean(fh_loss_p(sco1s_tra))
    loss_d = -np.mean(fh_loss_d(-ualph1))

    if K_qq is None:
        usig_kqq_usig = np.dot(usigbeta1, _apply_kqq(usigbeta1, c1, c2))
    else:
        usig_kqq_usig = np.dot(usigbeta1, K_qq @ usigbeta1)

    reg_d = -0.5 * lam1 * (np.dot(ualph1, K_xx @ ualph1) * ilamn1**2 - usig_kqq_usig)

    obj_p = reg_p + loss_p
    obj_d = reg_d + loss_d

    return obj_p, obj_d, sco2s_q


def train_pfgd(
    kern_input,
    sigvec1,
    lam1,
    gam_sm,
    qpmeth1,
    nepochs=100000,
    thres_gap=1e-4,
    verbose=0,
):
    ncons = len(sigvec1)
    K_xx, K_qx, K_qq, c1, c2 = _parse_kernel_input(kern_input, sigvec1)
    ntras = K_xx.shape[0]
    lamn1 = lam1 * ntras
    ilamn1 = 1.0 / lamn1
    K_xx_diag = np.diag(K_xx)
    R_mx_square = np.max(K_xx_diag)
    eta_L = (2.0 * gam_sm) / (2.0 * lam1 * gam_sm + R_mx_square)
    alph1_new = np.zeros(ntras)
    beta1_new = np.zeros(ncons)
    iters_rec = np.unique(np.ceil(np.logspace(0, np.log10(nepochs), 100)).astype(int))
    fh_loss_p = losses.logiloss
    fh_grad_p = losses.logiloss_grad
    fh_loss_d = losses.logiloss_ast

    niters_pfgd = nepochs
    i_iter_rec = 0
    obj1s_p = np.zeros(niters_pfgd)
    tms = np.zeros(len(iters_rec))

    start_time = time.time()

    for tea1 in range(1, niters_pfgd + 1):
        eta1 = eta_L
        alph1 = alph1_new.copy()
        beta1 = beta1_new.copy()
        sco3s_tra = ilamn1 * (K_xx @ alph1) + K_qx.T @ (sigvec1 * beta1)
        sigbeta1_cur = sigvec1 * beta1
        if K_qq is None:
            sco3s_q = ilamn1 * (K_qx @ alph1) + _apply_kqq(sigbeta1_cur, c1, c2)
        else:
            sco3s_q = ilamn1 * (K_qx @ alph1) + K_qq.T @ sigbeta1_cur
        pnw3 = ilamn1 * np.dot(alph1, sco3s_tra) + np.dot(sigvec1 * beta1, sco3s_q)
        reg3_p = 0.5 * lam1 * pnw3
        loss_p_each = fh_loss_p(sco3s_tra)
        loss_p = np.mean(loss_p_each)
        obj_p = reg3_p + loss_p
        if (i_iter_rec < len(iters_rec)) and (tea1 == iters_rec[i_iter_rec]):
            obj1s_p[i_iter_rec] = obj_p
            sigbeta1 = beta1 * sigvec1
            obj2_p, obj2_d, _ = demo1334_get_obj(
                alph1,
                sigbeta1,
                K_xx,
                K_qx,
                K_qq,
                sigvec1,
                lam1,
                fh_loss_p,
                fh_grad_p,
                fh_loss_d,
                qpmeth1,
                c1=c1,
                c2=c2,
            )
            gap1 = obj2_p - obj2_d
            tm1 = time.time() - start_time
            tms[i_iter_rec] = tm1

            if verbose > 0:
                print(
                    f"{tea1}: obj_p={obj_p:.6g}, obj2_p={obj2_p:.6g}, "
                    f"obj2_d={obj2_d:.6g}, gap={gap1:.6g}"
                )

            i_iter_rec += 1

            if gap1 < thres_gap:
                break
        loss_p_each_grad_sco = fh_grad_p(sco3s_tra)
        alph1_nabla = (alph1 + loss_p_each_grad_sco) * lam1
        beta1_nabla = beta1 * lam1
        alph1_half = alph1 - eta1 * alph1_nabla
        beta1_half = beta1 - eta1 * beta1_nabla
        sigbeta6_delta = demo1334_proj(
            alph1_half,
            beta1_half,
            lam1,
            sigvec1,
            K_qx,
            qpmeth1,
            c1=c1,
            c2=c2,
            kernmat1_qq=K_qq,
        )
        beta6_delta = sigbeta6_delta * sigvec1
        alph1_new = alph1_half
        beta1_new = beta1_half + beta6_delta
    actual_rec = i_iter_rec
    res1 = {
        "obj1s_p": obj1s_p[:actual_rec],
        "iters_rec": iters_rec[:actual_rec],
        "tms": tms[:actual_rec],
        "alph1": alph1,
        "sigbeta1": beta1 * sigvec1,
    }
    return res1
