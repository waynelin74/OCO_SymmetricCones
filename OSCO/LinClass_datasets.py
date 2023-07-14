"""
Version 2023-07-04.
"""

import numpy as np
from helpers import normalize, logistic
import SC_losses

def gen_rand_point(d, distr='ball'):
    """
    Generate a point uniform randomly from the space 'distr'.

    Parameters:
        -distr:
            -cube: uniformly from the box [-1, 1]^d.
            -ball: uniformly from the ball B^{d-1} = int(S^{d-1}).
    """
    point = np.array([])
    if distr == 'cube':
        for j in range(d):
            point = np.append(point, -1 + 2 * np.random.rand())
    elif distr == 'ball':
        dir = SC_losses.sample_spherical(d)
        r = (np.random.rand())**(1/d)
        point = r * dir
    elif distr == 'sphere':
        point = SC_losses.sample_spherical(d)
    return point

def gen_rand_dir(d, distr='sphere'):
    """
    Generates a point on the sphere S^{d-1}.
    Note:
        - Distribution depends on distr: use 'ball' to be uniform on the sphere.
    """
    if distr == 'cube':
        point = gen_rand_point(d, distr)
        point = normalize(point)
    elif distr == 'sphere':
        point = SC_losses.sample_spherical(d=d)
    return point

def update_dir(dir, perturbation, rate_of_change):
    return normalize(dir + rate_of_change * perturbation)

def gen_dataset(d, N,
    dir=None,
    halfwidth=1,
    margin=0,
    dir_perturb_rate=0,
    soft_halfwidth=0, soft_model='random', soft_param=0,
    flip_rate=0):
    """
    Samples a random direction z in R^d.
    Generates N uniformly generated points (x, y) with
        x in [- halfwidth, halfwidth]^d.
        For each x,
            label y = 1 if <z, d> >= 0,
            else label y = -1.
            Label wrongly with probability error_rate.
    Returns (X, Y), where
        X is list of datapoints x (ndarrays),
        Y is list of labels y (scalars).

    Parameters:
        -Dataset scaling
            -halfwidth: determines box ([-halfwidth, halfwidth]^d) that all points lie in.
        -Margin with no points:
            -margin: determines min(abs(<x, dir>)).
        -Soft clustering layer
            -soft_halfwidth: determines layer that has
            -soft_model: 'linear', 'logistic', or 'random'
            -soft_param: relative slope (in [0,1]), logistic param, or random error rate
        -Random errors
            -flip_rate: chance that a point is flipped to opposite classification.
    """
    # Initialization
    if not dir.any():
        dir = gen_rand_dir(d)
    X, Y = [], []
    # Generate dataset
    for n in range(N):
        # Generate data point
        x = halfwidth * gen_rand_point(d)
        shadow = np.vdot(dir, x)
        # Re-generate data point if data point lies in forbidden margin
        while np.abs(shadow) < margin:
            x = halfwidth * gen_rand_point(d)
            shadow = np.vdot(dir, x)
        X.append(x)
        ## Generate label
        y = np.sign(shadow)
        # Soft-clustered margin
        soft_rand = np.random.rand()
        if soft_halfwidth:
            if np.abs(shadow) < soft_halfwidth:
                if soft_model == 'random':
                    if soft_rand < soft_param:
                        y = -y
                elif soft_model == 'logistic':
                    if soft_rand > logistic((np.abs(shadow) / soft_halfwidth), soft_param):
                        y = -y
                elif soft_model == 'linear':
                    if soft_rand > 0.5 + 0.5 * soft_param * (np.abs(shadow) / soft_halfwidth):
                        y = -y
        # Random flips
        flip_rand = np.random.rand()
        if flip_rate:
            if flip_rand < flip_rate:
                y = -y
        # Final label
        Y.append(y)
        # Update dir
        perturbation = gen_rand_point(d)
        if dir_perturb_rate:
            dir = update_dir(dir, perturbation, dir_perturb_rate)
    return (X, Y)
