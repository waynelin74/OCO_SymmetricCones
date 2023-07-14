import numpy as np
from copy import deepcopy
from scipy.stats import unitary_group
from helpers import length

"""
Non-negative Orthant
"""
def gen_loss_random_NO(m):
    """
    Generates a loss in R^m with entries (eigenvalues) in [-1, 1].
    """
    return -1 + 2 * np.random.rand(m)

"""
SOC
"""

def sample_spherical(d=2):
    vec = np.random.randn(d)
    vec /= np.linalg.norm(vec)
    return vec

def gen_loss_random_SOC(d):
    """
    Generates a loss in R^(d+1) with eigenvalues in [-1, 1].
    height t, radius ||\bar{x}||_2 = r.
    """
    x0 = -1 + 2 * np.random.rand()
    r = (1 - abs(x0)) * np.random.rand()
    # print("d=", d)
    x_bar = sample_spherical(d=d)
    # print(x_bar)
    return np.append(r * x_bar, x0)


"""
PSD
"""

def gen_loss_random_PSD(n):
    """
    Generates a loss in S_+^n with eigenvalues in [-1, 1].
    Pseudocode:
        Generate random basis according to scipy.stats.unitary_group.
        Generate eigenvalues uniform in [-1, 1].
    """
    U = unitary_group.rvs(n)
    eigs = -1 + 2 * np.random.rand(n)
    D = np.diag(eigs)
    return U@D@U.T

"""
General symmetric cone
"""

def gen_loss_random_SC(dim):
    """
    Generates a loss in EJA with direct-sum structure dim.
    """
    (m, n, d) = deepcopy(dim)
    m = gen_loss_random_NO(m)
    for i in range(len(n)):
        n[i] = gen_loss_random_PSD(n[i])
    for i in range(len(d)):
        d[i] = gen_loss_random_SOC(d[i])
    return (m, n, d)


def gen_loss_hist_random_SC(dim, T):
    loss_hist = []
    for i in range(T):
        loss_hist.append(gen_loss_random_SC(dim))
    return loss_hist

"""
ball
"""

def gen_rand_point(d, distr='ball'):
    """
    Generate a point uniform randomly from the space 'distr'.

    Parameters:
        -distr:
            -cube: uniformly from the box [-1, 1]^d.
            -ball: uniformly from the 2-norm-ball B^{d-1} = int(S^{d-1}).
    """
    point = np.array([])
    if distr == 'cube':
        for j in range(d):
            point = np.append(point, -1 + 2 * np.random.rand())
    elif distr == 'ball':
        point = gen_rand_point(d, distr='cube')
        while length(point) > 1:
            point = gen_rand_point(d, distr='cube')
    return point

def gen_loss_hist_random_ball(d, T):
    """
    Generates sequence of losses in unit ball. (Uniform distribution.)
    """
    m_hist = []
    for i in range(T):
        m_hist.append(gen_rand_point(d, distr='ball'))
    return m_hist
