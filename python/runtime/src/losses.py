import numpy as np


def logiloss(z):
    ret = np.zeros_like(z, dtype=float)
    l_p = z >= 0
    l_n = ~l_p
    z_p = z[l_p]
    z_n = z[l_n]
    ret[l_p] = np.log(1.0 + np.exp(-z_p))
    ret[l_n] = -z_n + np.log(1.0 + np.exp(z_n))
    return ret


def logiloss_grad(z):
    ret = np.zeros_like(z, dtype=float)
    l_p = z >= 0
    l_n = ~l_p
    z_p = z[l_p]
    z_n = z[l_n]
    tmp1 = np.exp(-z_p)
    ret[l_p] = -tmp1 / (tmp1 + 1.0)
    ret[l_n] = -1.0 / (np.exp(z_n) + 1.0)
    return ret


def logiloss_ast(a1):
    ret = np.zeros_like(a1, dtype=float)
    l_lt = a1 < -1.0
    l_lp = a1 == -1.0
    l_in = (a1 > -1.0) & (a1 < 0.0)
    l_rp = a1 == 0.0
    l_rr = a1 > 0.0

    a1_in = a1[l_in]
    term1 = -a1_in * np.log(-a1_in)
    term2 = (1.0 + a1_in) * np.log(1.0 + a1_in)
    ret[l_in] = term1 + term2
    ret[l_lt | l_rr] = np.inf
    ret[l_lp | l_rp] = 0.0
    return ret
