import numpy as np

"""
Euclidean 2-norm (length) and normalization
"""
def length(v):
    return np.sqrt(np.vdot(v, v))

def normalize(v):
    return v / length(v)

"""
Logistic function
"""
def logistic(x, k):
    return 1/(1+ np.exp(-k * x))

"""
Exponential function that avoids numerical errors
"""
def expn(x, avoid_num_errors=True):
    """
    Exponentiates scalar x, but if x too negative, outputs 1.

    Input:
        - x scalar.
    """
    if avoid_num_errors and x < -35:
        x_eff = 0
    else:
        x_eff = x
    return np.exp(x_eff)

expn = np.vectorize(expn)
