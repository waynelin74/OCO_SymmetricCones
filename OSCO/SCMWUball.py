"""
Packages
"""
import numpy as np
from copy import deepcopy
from helpers import length


"""
SCMWUball
"""

def init(d):
    return np.zeros(d)

def norm_exp(x):
    x_2norm = length(x)
    x_normalized = x / x_2norm
    if x_2norm < 36:
        a = np.exp(x_2norm)
        denom = a + 1/a
        return (a/denom)*(x_normalized/2) + ((1/a)/denom)*(-x_normalized/2)
    else: # if x_2norm will overflow np.exp
        return x_normalized/2

def SCMWU_iter(stepsize, sum_loss_vectors):
    """
    Returns the next SCMWU_ball update with stepsize stepsize,
    when the sum of losses is sum_losses.

    Inputs:
        - stepsize (scalar)
        - sum_losses (d-dim ndarray)

    Output:
        -iterate (d-dim ndarray)
    """
    return 2 * norm_exp(-stepsize * sum_loss_vectors)

# def SCMWU_iter_w_last(stepsize, sum_loss_vectors, last_iter):
#     return SCMWU_iter(stepsize, sum_loss_vectors)

def opt_stepsize_for_SCMWUball(T):
    """
    Returns the fixed stepsize to get minimal regret bound
    """
    return np.sqrt(np.log(2)/T)

def SCMWUball(d, stepsize, T, m_hist):
    # Initialization
    p1 = init(d)
    p_hist = [p1]
    # First Loss
    m1 = deepcopy(m_hist[0])
    # print("m1 =", m1, "\n")
    sum_m = m1
    accum_loss = np.vdot(m1, p1)
    regret = accum_loss - (- length(sum_m))
    regret_hist = [regret]
    # Loop
    for t in range(2, min(T, len(m_hist)) + 1):
        # Update
        exponent = -stepsize * sum_m
        p = SCMWU_iter(stepsize, sum_m)
        p_hist.append(p)
        # Loss
        m = deepcopy(m_hist[t-1])
        sum_m += m
        # regret
        accum_loss += np.vdot(m, p)
        regret = accum_loss - (- length(sum_m))
        regret_hist.append(regret)
        # scaled_regret_hist = [x * 4 for x in regret_hist]
    return (p_hist, regret_hist)

def SCMWUball_optimized(d, T, m_hist):
    """
    SCMWU with optimized stepsize for fixed time horizon T.
    """
    stepsize = opt_stepsize_for_SCMWUball(T)
    return SCMWUball(d, stepsize, T, m_hist)

def SCMWUball_doubling(d, T, m_hist):
    p_hist, regret_hist = [], []
    time_left = T
    epoch = 0
    sum_m = np.zeros(d)
    accum_loss = 0
    while time_left > 0:
        time_add = min(time_left, 2**epoch)
        time_left -= 2**epoch
        m_hist_for_epoch = deepcopy(m_hist[2**epoch - 1 : 2**epoch + time_add - 1])
        p_hist_new = SCMWUball_optimized(d, 2**epoch, m_hist_for_epoch)[0]
        p_hist += p_hist_new
        for t in range(time_add):
            p = p_hist_new[t]
            m = m_hist_for_epoch[t]
            accum_loss += np.vdot(p, m)
            sum_m += m
            regret = accum_loss - (- length(sum_m))
            regret_hist.append(regret)
        epoch += 1
    return (p_hist, regret_hist)

def regbound(T):
    return np.sqrt(8 * np.log(2)) * np.sqrt(T)

def regbound_doubling(T):
    constant = np.sqrt(2)/(np.sqrt(2)-1)
    return constant * regbound(T)

"""
OGD (over ball)
"""
def opt_stepsize_OGD(T):
    """
    Returns the fixed stepsize to get minimal regret bound
    """
    return 1 / np.sqrt(T)

def OGD_iter(last_iter, stepsize, new_loss_vector):
    new_iter = last_iter - stepsize * new_loss_vector
    iter_norm = length(new_iter)
    if iter_norm > 1:
        new_iter /= iter_norm
    return new_iter

def OGD(d, T, m_hist, stepsize):
    # Iterate Initialization
    p = np.zeros(d)
    p_hist_OGD = [p]
    # Regret initialization
    accum_loss = np.vdot(m_hist[0], p_hist_OGD[0])
    m = m_hist[0].copy()
    sum_m = m.copy()
    regret = accum_loss - (-length(sum_m))
    regret_hist_OGD = [regret]
    for t in range(2, T+1):
        # Update
        p = OGD_iter(p, stepsize, m)
        p_hist_OGD.append(p)
        # Loss
        m = m_hist[t-1].copy()
        sum_m += m
        accum_loss += np.vdot(m, p)
        # regret
        regret = accum_loss - (- length(sum_m))
        regret_hist_OGD.append(regret)
    return (p_hist_OGD, regret_hist_OGD)

def OGD_optimized(d, T, m_hist):
    opt_stepsize = opt_stepsize_OGD(T)
    return OGD(d, T, m_hist, opt_stepsize)

"""
OGD_lazy (over ball)
"""

def OGD_lazy(d, T, m_hist, stepsize):
    # Iterate Initialization
    p1 = np.zeros(d)
    p_hist_OGD = [p1]
    # Regret initialization
    accum_loss = np.vdot(m_hist[0], p_hist_OGD[0])
    sum_m = m_hist[0].copy()
    regret = accum_loss - (-length(sum_m))
    regret_hist_OGD = [regret]
    for t in range(2, T+1):
        # Update
        dir = - stepsize * sum_m[:-1]
        dir_norm = length(dir)
        if dir_norm > 1:
            dir = dir / dir_norm
        p = np.append(dir, np.array([0.5]))
        p_hist_OGD.append(p)
        # Loss
        m = m_hist[t-1].copy()
        sum_m += m
        accum_loss += np.vdot(m, p)
        # regret
        regret = accum_loss - (- length(sum_m))
        regret_hist_OGD.append(regret)
    return (p_hist_OGD, regret_hist_OGD)

def OGD_lazy_optimized(d, T, m_hist):
    """
    TODO: verify that this is indeed the opt stepsize for OGD_lazy.
    """
    opt_stepsize = opt_stepsize_OGD(T)
    return OGD_lazy(d, T, m_hist, opt_stepsize)
