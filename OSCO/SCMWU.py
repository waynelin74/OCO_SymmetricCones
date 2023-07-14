"""
Version 2023-07-04.
"""

import numpy as np
from copy import deepcopy

import SC

"""
SCMWU
"""

def opt_stepsize_for_SCMWU(dim, T):
    """
    Returns the fixed stepsize to get minimal regret bound
    """
    r = SC.rk_from_dim(dim)
    return np.sqrt(np.log(r)/T)

def iter(stepsize, sum_loss_vectors):
    """
    Returns the next SCMWU update with stepsize stepsize,
    when the sum of losses is sum_losses.

    Inputs:
        - stepsize (scalar)
        - sum_losses ((d+1)-dim ndarray)

    Output:
        - next iterate ((d+1)-dim ndarray)
    """
    exponent = SC.scalar_mult(sum_loss_vectors, -stepsize)
    unnormalized_iter = SC.exp(exponent)
    return unnormalized_iter / tr(unnormalized_iter)

def SCMWU(dim, stepsize, T, m_hist):
    # Initialization
    p1 = SC.e(dim)
    # print("dim =", dim, "\n")
    # print("(unnormalized) p1 =", p1, "\n")
    p1 = SC.scalar_mult(p1, 1/SC.tr(p1))
    p_hist = [p1]
    # print("p1 =", p1, "\n")
    # First Loss
    m1 = deepcopy(m_hist[0])
    # print("m1 =", m1, "\n")
    sum_m = m1
    accum_loss = SC.innerprod(m1, p1)
    regret = accum_loss - min(SC.eigvs(sum_m))
    regret_hist = [regret]
    # Loop
    for t in range(2, min(T, len(m_hist)) + 1):
        # Update
        exponent = SC.scalar_mult(sum_m, -stepsize)
        p = SC.exp(exponent)
        # print("p =", p, "\n")
        p = SC.scalar_mult(p,  1/SC.tr(p))
        p_hist.append(p)
        # Loss
        m = deepcopy(m_hist[t-1])
        sum_m = SC.add(sum_m, m)
        # regret
        accum_loss += SC.innerprod(m, p)
        regret = accum_loss - min(SC.eigvs(sum_m))
        regret_hist.append(regret)
        # scaled_regret_hist = [x * 4 for x in regret_hist]
    return (p_hist, regret_hist)

def SCMWU_optimized(dim, T, m_hist):
    """
    SCMWU with optimized stepsize for fixed time horizon T.
    """
    stepsize = opt_stepsize_for_SCMWU(dim, T)
    return SCMWU(dim, stepsize, T, m_hist)

def SCMWU_doubling(dim, T, m_hist):
    p_hist, regret_hist = [], []
    time_left = T
    epoch = 0
    sum_m = SC.zeros(dim)
    accum_loss = 0
    while time_left > 0:
        time_add = min(time_left, 2**epoch)
        time_left -= 2**epoch
        m_hist_for_epoch = deepcopy(m_hist[2**epoch - 1 : 2**epoch + time_add - 1])
        p_hist_new = SCMWU_optimized(dim, 2**epoch, m_hist_for_epoch)[0]
        p_hist += p_hist_new
        for t in range(time_add):
            p = p_hist_new[t]
            m = m_hist_for_epoch[t]
            accum_loss += SC.innerprod(p, m)
            sum_m = SC.add(sum_m, m)
            regret = accum_loss - min(SC.eigvs(sum_m))
            regret_hist.append(regret)
        epoch += 1
    return (p_hist, regret_hist)

def regbound(dim, T):
    r = SC.rk_from_dim(dim)
    return 2 * np.sqrt(T * np.log(r))

def regbound_doubling(dim, T):
    constant = np.sqrt(2)/(np.sqrt(2)-1)
    return constant * regbound(dim, T)
