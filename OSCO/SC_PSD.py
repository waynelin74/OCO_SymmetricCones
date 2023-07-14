import numpy as np
import scipy.linalg as scilin

# e = np.array([0, 0, 1]);

def e(n):
    """
    Returns identity element in S_+^n.
    """
    return np.eye(n)

def zeros(n):
    """
    Returns zero element in S_+^n.
    """
    return np.zeros((n, n))

def tr(x):
    """
    Input: x in S_+^n
    Output: tr(x)
    """
    return np.real(np.trace(x))

def jordanprod(x, y):
    """
    Input: x, y in S_+^n
    Output: x \circ y
    """
    return (x@y + y@x)/2

def innerprod(x, y):
    """
    EJA inner product.
    Input: x, y in S_+^n
    Output: < x, y >
    """
    return tr(jordanprod(x,y))

def EJAnorm(x):
    return np.sqrt(innerprod(x,x))

def Eucnorm(x):
    return np.sqrt(np.trace(x.T @ x))

def eigvs(x):
    return list(np.real(np.linalg.eigvals(x)))

def exp(x):
    """
    Input: x in S_+^n
    Output: exp(x)
    """
    return np.real(scilin.expm(x))

def ln(x):
    return np.real(scilin.logm(x))
