import numpy as np


def init_WH(V, k):
    m, n = np.shape(V)
    avg = np.sqrt(V.mean()/k)
    W = avg * np.random.randn(m, k)
    H = avg * np.random.randn(n, k)
    np.abs(W, W)
    np.abs(H, H)
    return W, H

