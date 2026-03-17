import numpy as np
from scipy.spatial.distance import cdist


def lin_kernel(X1, X2):
    return X1.T @ X2


def rbf_kernel(X1, X2, gamma):
    dists = cdist(X1.T, X2.T, metric="sqeuclidean")
    return np.exp(-gamma * dists)
