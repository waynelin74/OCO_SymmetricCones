import numpy as np

# e = np.array([0, 0, 1]);

def e(m):
    """
    Returns identity element in R_+^m.
    """
    l = [1.0] * m
    return np.array(l)

def zeros(m):
    """
    Returns zero element in R_+^m.
    """
    l = [0.0] * m
    return np.array(l)

def tr(x):
    """
    Input: x in R^m
    Output: tr(x)
    """
    return sum(x)

def jordanprod(x, y):
    """
    Input: x, y in R^m
    Output: x \circ y
    """
    out = []
    for i in range(len(x)):
        out.append(x[i] * y[i])
    return np.array(out)

def innerprod(x, y):
    """
    EJA inner product.
    Input: x, y in R^m
    Output: < x, y > = x^T y
    """
    return np.transpose(x) @ y

def EJAnorm(x):
    return np.sqrt(innerprod(x,x))

def Eucnorm(x):
    return np.sqrt(x.T @ x)

def eigvs(x):
    return list(x)

def exp(x):
    """
    Input: x in R^m
    Output: exp(x)
    """
    lx = list(x)
    out = np.array(list(map(np.exp, lx)))
    return out

def ln(x):
    lx = list(x)
    out = np.array(list(map(np.log, lx)))
    return out
