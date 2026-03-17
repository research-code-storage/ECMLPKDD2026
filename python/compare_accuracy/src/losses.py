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
