import numpy as np

# e = np.array([0, 0, 1]);

def e(d):
    """
    Returns identity element in SOC^d = R^(d+1).
    """
    l = [0.0] * d
    l.append(1.0)
    return np.array(l)

def zeros(d):
    """
    Returns zero element in R^(d+1).
    """
    l = [0.0] * d
    l.append(0.0)
    return np.array(l)

def bar(x):
    """
    Input: x in R^{d+1}
    Output: [x0, x1]
    """
    return x[:-1]

def tr(x):
    """
    Input: x in R^{d+1}
    Output: tr(x)
    """
    return 2*x[-1]

def jordanprod(x, y):
    """
    Input: x, y in R^{d+1}
    Output: x \circ y
    """
    out = x[-1]*bar(y) + y[-1]*bar(x)
    out = np.append(np.array(out), np.transpose(x)@np.transpose(y))
    return out

def innerprod(x, y):
    """
    EJA inner product.
    Input: x, y in R^{d+1}
    Output: < x, y > = 2 x^T y
    """
    return 2*(np.transpose(x)@y)

def EJAnorm(x):
    return np.sqrt(innerprod(x,x))

def Eucnorm(x):
    return np.sqrt(x.T@x)

def eigvs(x):
    xbar = bar(x)
    normbar = np.sqrt(xbar.T@xbar)
    return [x[-1] - normbar, x[-1] + normbar]

def exp(x):
    """
    Input: x in R^3
    Output: exp(x) in int(second-order cone.)
    """
    xbar = bar(x)
    normbar = np.sqrt(xbar.T@xbar)
    if normbar != 0:
        normedbar = xbar/normbar
        # eigenvalues
        l1, l2 = x[-1] - normbar, x[-1] + normbar
        e1, e2 = np.exp(l1), np.exp(l2)
        # eigenvectors
        v1, v2 = 0.5*np.append(-normedbar, 1), 0.5*np.append(normedbar, 1)
        # Output
        return e1*v1 + e2*v2
    else:
        return np.exp(x[-1]) * e

def ln(x):
    xbar = bar(x)
    normbar = np.sqrt(xbar.T@xbar)
    if normbar != 0:
        normedbar = xbar/normbar
        # eigenvalues
        l1, l2 = x[-1] - normbar, x[-1] + normbar
        e1, e2 = np.log(l1), np.log(l2)
        # eigenvectors
        v1, v2 = 0.5*np.append(-normedbar, 1), 0.5*np.append(normedbar, 1)
        # Output
        return e1*v1 + e2*v2
    else:
        return np.log(x[-1]) * e
