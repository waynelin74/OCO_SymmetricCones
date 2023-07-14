import numpy as np
from helpers import expn # Exponential function that avoids numerical error

def init(d):
    return (1/d) * np.ones(d)

def iter(stepsize, sum_loss_vectors):
    """
    Returns the next MWU update with stepsize stepsize,
    when the sum of losses is sum_loss_vectors.

    Inputs:
        - stepsize (scalar)
        - sum_losses (d-dim ndarray)

    Output:
        -iterate (d-dim ndarray)
    """
    # unnormalized_iter = np.exp(-stepsize * sum_loss_vectors)
    unnormalized_iter = expn(-stepsize * sum_loss_vectors)
    return unnormalized_iter / np.sum(unnormalized_iter)


def update(stepsize, last_iter, loss_vector):
    """
    Returns the next MWU update with stepsize stepsize,
    when the last iterate is last_iter,
    and the new loss_vector is loss_vector.

    Inputs:
        - stepsize (scalar)
        - last_iter (d-dim ndarray)
        - loss_vector (d-dim ndarray)

    Output:
        - next iterate (d-dim ndarray)
    """
    # unnormalized_update = last_iter * np.exp(-stepsize * loss_vector)
    unnormalized_update = last_iter * expn(-stepsize * loss_vector)
    return unnormalized_update / np.sum(unnormalized_update)
