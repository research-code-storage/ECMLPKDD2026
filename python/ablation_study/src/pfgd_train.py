import time

import numpy as np

from . import losses
from .qp_backends import make_box_qp

try:
    from . import pfgd_solvers
except ImportError:
    pfgd_solvers = None


def _apply_kqq(v, c1, c2):
    return (c1 - c2) * v + c2 * np.sum(v)


def _solve_cqkp_python(cd: float, avec1: np.ndarray, sigvec1: np.ndarray) -> np.ndarray:
    ncons = avec1.shape[0]
    advec1 = np.empty(ncons + 1, dtype=float)
    advec1[0] = avec1[0] - 1.0
    advec1[-1] = avec1[-1] + 1.0
    advec1[1:-1] = 0.5 * (avec1[:-1] + avec1[1:])

    grad = np.empty(ncons, dtype=float)
    for i, tea in enumerate(avec1):
        val = (avec1 - tea) * sigvec1
        x = np.where(val > 0.0, val * sigvec1, 0.0)
        grad[i] = np.sum(x) - tea / cd

    idx = np.where(grad <= 0.0)[0]
    j1 = int(idx[0]) if idx.size > 0 else ncons
    threshold = advec1[j1]
    l1 = ((avec1 - threshold) * sigvec1 >= 0.0).astype(float)
    tea_star = float(np.dot(avec1, l1) / (np.sum(l1) + 1.0 / cd))
    val = (avec1 - tea_star) * sigvec1
    x_star = np.where(val > 0.0, val * sigvec1, 0.0)
    return x_star


def demo1334_proj(
    alph1_half,
    beta1_half,
    lam1,
    sigvec1,
    kernmat1_qx,
    qpmeth1,
    c1=None,
    c2=None,
):
    ncons, ntras = kernmat1_qx.shape
    lamn1 = lam1 * ntras
    ilamn1 = 1.0 / lamn1

    if ncons == 0:
        return np.array([])

    if c1 is None or c2 is None:
        raise ValueError("c1 and c2 are required")

    if abs(c2) != 0.0:
        cd = c2 / (c1 - c2)
        kqq_v = _apply_kqq(sigvec1 * beta1_half, c1, c2)
        avec1 = -(ilamn1 * kernmat1_qx @ alph1_half + kqq_v) / (c1 - c2)

        r_srt = np.argsort(avec1)
        avec1_srt = avec1[r_srt]
        sigvec1_srt = sigvec1[r_srt]
        qp = make_box_qp(cd, avec1_srt, sigvec1_srt)

        qpm = str(qpmeth1).lower()
        if qpm == "cqkp":
            if pfgd_solvers is not None:
                xvec1_star_srt = pfgd_solvers.solve_cqkp(cd, qp.avec, qp.sigvec)
            else:
                xvec1_star_srt = _solve_cqkp_python(cd, qp.avec, qp.sigvec)
        elif qpm in {"proxqp", "cvxopt"}:
            if pfgd_solvers is None:
                raise ImportError("Cython module pfgd_solvers is required for proxqp/cvxopt")
            if qpm == "proxqp":
                xvec1_star_srt = pfgd_solvers.solve_proxqp(cd, qp.avec, qp.sigvec)
            else:
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


def train_pfgd(
    K_xx,
    K_qx,
    c1,
    c2,
    sigvec1,
    lam1,
    gam_sm,
    qpmeth1,
    nepochs=1000,
    mode_sgncon="sc",
    verbose=0,
):
    ncons = len(sigvec1)
    ntras = K_xx.shape[0]
    lamn1 = lam1 * ntras
    ilamn1 = 1.0 / lamn1

    fh_loss_p = losses.logiloss
    fh_grad_p = losses.logiloss_grad

    if mode_sgncon not in {"sc", "sf"}:
        raise ValueError("mode_sgncon must be sc or sf")

    eta_L = (2.0 * gam_sm) / (2.0 * lam1 * gam_sm + np.max(np.diag(K_xx)))

    alph1_new = np.zeros(ntras)
    beta1_new = np.zeros(ncons)
    alph1 = alph1_new.copy()
    beta1 = beta1_new.copy()

    iters_rec = np.unique(np.ceil(np.logspace(0, np.log10(nepochs), 100)).astype(int))
    obj1s_p = np.zeros(len(iters_rec))
    tms = np.zeros(len(iters_rec))
    i_iter_rec = 0
    start_time = time.time()

    for tea1 in range(1, nepochs + 1):
        alph1 = alph1_new.copy()
        beta1 = beta1_new.copy()

        if mode_sgncon == "sc":
            sco3s_tra = ilamn1 * (K_xx @ alph1) + K_qx.T @ (sigvec1 * beta1)
            sco3s_q = ilamn1 * (K_qx @ alph1) + _apply_kqq(sigvec1 * beta1, c1, c2)
            pnw3 = ilamn1 * np.dot(alph1, sco3s_tra) + np.dot(sigvec1 * beta1, sco3s_q)
        else:
            sco3s_tra = ilamn1 * (K_xx @ alph1)
            pnw3 = ilamn1 * np.dot(alph1, sco3s_tra)

        reg3_p = 0.5 * lam1 * pnw3
        loss_p = np.mean(fh_loss_p(sco3s_tra))
        obj_p = reg3_p + loss_p

        if i_iter_rec < len(iters_rec) and tea1 == iters_rec[i_iter_rec]:
            obj1s_p[i_iter_rec] = obj_p
            tms[i_iter_rec] = time.time() - start_time
            if verbose > 0:
                print(f"{tea1}: obj_p={obj_p:.6g}")
            i_iter_rec += 1

        loss_p_each_grad_sco = fh_grad_p(sco3s_tra)
        alph1_nabla = (alph1 + loss_p_each_grad_sco) * lam1
        alph1_half = alph1 - eta_L * alph1_nabla

        if mode_sgncon == "sc":
            beta1_nabla = beta1 * lam1
            beta1_half = beta1 - eta_L * beta1_nabla
            sigbeta6_delta = demo1334_proj(
                alph1_half,
                beta1_half,
                lam1,
                sigvec1,
                K_qx,
                qpmeth1,
                c1=c1,
                c2=c2,
            )
            beta6_delta = sigbeta6_delta * sigvec1
            beta1_new = beta1_half + beta6_delta

        alph1_new = alph1_half

    out = {
        "obj1s_p": obj1s_p[:i_iter_rec],
        "iters_rec": iters_rec[:i_iter_rec],
        "tms": tms[:i_iter_rec],
        "alph1": alph1,
    }
    if mode_sgncon == "sc":
        out["sigbeta1"] = beta1 * sigvec1
    else:
        out["sigbeta1"] = np.zeros(ncons)
    return out
