import numpy as np


def logiloss(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    ret = np.zeros_like(z)
    l_p = z >= 0
    l_n = ~l_p
    z_p = z[l_p]
    z_n = z[l_n]
    ret[l_p] = np.log(1.0 + np.exp(-z_p))
    ret[l_n] = -z_n + np.log(1.0 + np.exp(z_n))
    return ret


def logiloss_grad(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    ret = np.zeros_like(z)
    l_p = z >= 0
    l_n = ~l_p
    z_p = z[l_p]
    z_n = z[l_n]
    tmp1 = np.exp(-z_p)
    ret[l_p] = -tmp1 / (tmp1 + 1.0)
    ret[l_n] = -1.0 / (np.exp(z_n) + 1.0)
    return ret


def logiloss_ast(a1: np.ndarray) -> np.ndarray:
    a1 = np.asarray(a1, dtype=float)
    ret = np.zeros_like(a1)
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


def qhingeloss(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    return 0.5 * np.maximum(0.0, 1.0 - z) ** 2


def qhingeloss_grad(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    return -(1.0 - z) * (z <= 1.0)


def qhingeloss_ast(a1: np.ndarray) -> np.ndarray:
    a1 = np.asarray(a1, dtype=float)
    ret = np.full_like(a1, np.inf)
    mask = (a1 <= 0.0) & (a1 >= -1e100)
    ret[mask] = 0.5 * a1[mask] ** 2 + a1[mask]
    return ret


def smhingeloss(z: np.ndarray, m: float) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    ret = np.zeros_like(z)
    l1 = z <= (1.0 - m)
    l2 = (z > (1.0 - m)) & (z < 1.0)
    ret[l1] = 1.0 - z[l1] - 0.5 * m
    ret[l2] = ((1.0 - z[l2]) ** 2) / (2.0 * m)
    return ret


def smhingeloss_grad(z: np.ndarray, m: float) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    ret = np.zeros_like(z)
    l1 = z <= (1.0 - m)
    l2 = (z > (1.0 - m)) & (z < 1.0)
    ret[l1] = -1.0
    ret[l2] = -(1.0 - z[l2]) / m
    return ret


def smhingeloss_ast(a1: np.ndarray, m: float) -> np.ndarray:
    a1 = np.asarray(a1, dtype=float)
    ret = np.full_like(a1, np.inf)
    mask = (a1 >= -1.0) & (a1 <= 0.0)
    ret[mask] = 0.5 * m * a1[mask] ** 2 + a1[mask]
    return ret
