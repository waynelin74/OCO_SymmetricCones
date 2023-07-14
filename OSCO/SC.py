import numpy as np

import SC_NO as NO
import SC_PSD as PSD
import SC_SOC as SOC

from copy import deepcopy

"""
General symmetric cone of
Dimension, dim = (m, [n1, n2, ..., nk], [d1, d2, ..., dl]).
symmetric cone element x will be stored as
(
    ndarray of len m,
    [list of k matrices of size ni * ni each],
    [list of l ndarrays of size di+1 each]
).

Vector null entries:
    -If x does not have R^m component, x[0] should be np.array([]).
    -If x does not have PSD components, x[1] should be [].
    -If x does not have SOC components, x[2] should be [].

dim null entries:
    -If x does not have R^m component, dim(x)[0] should be 0.
    -If x does not have PSD components, dim(x)[1] should be [].
    -If x does not have SOC components, dim(x)[2] should be [].
"""

def dim(x):
    """
    Returns dimension of x.
    """
    # print("dim call: \t x =", x)
    m = len(x[0])
    k = len(x[1])
    n = []
    for i in range(k):
        n.append(len(x[1][i]))
    l = len(x[2])
    d = []
    for i in range(l):
        d.append(len(x[2][i])-1)
    return (m, n, d)

def rk_from_dim(dim):
    return dim[0] + sum(dim[1]) + 2 * len(dim[2])

def rk(x):
    """
    Returns rank of x.
    """
    # (m, n, d) = dim(x)
    # if m:
    #     return len(m) + len(n) + len(d)
    # else:
    #     return len(n) + len(d)
    return rk_from_dim(dim(x))

def e(dim):
    """
    Returns identity element in symmetric cone of dimension d.
    """
    (m, n, d) = deepcopy(dim)
    m = NO.e(m)
    if n:
        # n = n.copy()
        n = list(map(PSD.e, n))
    if d:
        # d = d.copy()
        d = list(map(SOC.e, d))
    return (m, n, d)

def zeros(dim):
    """
    Returns zero element in symmetric cone.
    """
    (m, n, d) = deepcopy(dim)
    m = NO.zeros(m)
    if n:
        n = list(map(PSD.zeros, n))
    if d:
        d = list(map(SOC.zeros, d))
    return (m, n, d)

def jordanprod(x, y):
    """
    Input: x, y in general symmetric cone
    Output: x \circ y
    """
    (m, n, d) = deepcopy(dim(x))
    m = NO.jordanprod(x[0], y[0])
    if n:
        k = len(n)
        n = []
        for i in range(k):
            n.append(PSD.jordanprod(x[1][i], y[1][i]))
    if d:
        l = len(d)
        d = []
        for i in range(l):
            d.append(SOC.jordanprod(x[2][i], y[2][i]))
    return (m, n, d)

def innerprod(x, y):
    """
    EJA inner product.
    Input: x, y in general symmetric cone
    Output: < x, y > = x^T y
    """
    (m, n, d) = deepcopy(dim(x))
    # print("(m, n, d) = (", m, ",", n, ",", d, "). \n")
    sum = 0
    if m:
        sum += NO.innerprod(x[0], y[0])
    k = len(n)
    for i in range(k):
        sum += PSD.innerprod(x[1][i], y[1][i])
    l = len(d)
    # print("x[2] =", x[2])
    # print("y[2] =", y[2])
    for i in range(l):
        sum += SOC.innerprod(x[2][i], y[2][i])
    return sum

def EJAnorm(x):
    return np.sqrt(innerprod(x,x))

# def Eucnorm(x):
#     return np.sqrt(x.T @ x)

def eigvs(x):
    (m, n, d) = deepcopy(dim(x))
    list_eigs = []
    if m:
        list_eigs += NO.eigvs(x[0])
    if n:
        k = len(n)
        for i in range(k):
            list_eigs += PSD.eigvs(x[1][i])
    if d:
        l = len(d)
        for i in range(l):
            list_eigs += SOC.eigvs(x[2][i])
    return list_eigs

# def tr(x):
#     return sum(eigvs(x))

def tr(x):
    (m, n, d) = deepcopy(dim(x))
    sum = 0
    if m:
        sum += NO.tr(x[0])
    if n:
        k = len(n)
        for i in range(k):
            sum += PSD.tr(x[1][i])
    if d:
        l = len(d)
        for i in range(l):
            sum += SOC.tr(x[2][i])
    return sum

def exp(x):
    """
    Input: x in general symm cone
    Output: exp(x)
    """
    (m, n, d) = deepcopy(dim(x))
    m = NO.exp(x[0])
    if n:
        k = len(n)
        n = []
        for i in range(k):
            n.append(PSD.exp(x[1][i]))
    if d:
        l = len(d)
        d = []
        for i in range(l):
            d.append(SOC.exp(x[2][i]))
    return (m, n, d)

def ln(x):
    (m, n, d) = deepcopy(dim(x))
    m = NO.ln(x[0])
    if n:
        k = len(n)
        n = []
        for i in range(k):
            n.append(PSD.ln(x[1][i]))
    if d:
        l = len(d)
        d = []
        for i in range(l):
            d.append(SOC.ln(x[2][i]))
    return (m, n, d)

"""
Basic operations
"""

def scalar_mult(x, c):
    """
    Scales EJA vector x by scalar c.
    """
    (m, n, d) = deepcopy(dim(x))
    m = c * x[0]
    if n:
        k = len(n)
        n = []
        for i in range(k):
            n.append(c * x[1][i])
    if d:
        l = len(d)
        d = []
        for i in range(l):
            d.append(c * x[2][i])
    return (m, n, d)

def add(x,y):
    """
    Adds two EJA vectors x and y.
    """
    (m, n, d) = deepcopy(dim(x))
    m = x[0] + y[0]
    if n:
        k = len(n)
        n = []
        for i in range(k):
            n.append(x[1][i] + y[1][i])
    if d:
        l = len(d)
        d = []
        for i in range(l):
            d.append(x[2][i] + y[2][i])
    return (m, n, d)
