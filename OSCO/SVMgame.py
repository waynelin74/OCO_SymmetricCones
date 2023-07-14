import os
import numpy as np
import MWU
import SCMWUball as SCMWU_ball
from helpers import length, normalize # Euclidean 2-norm and normalization
# Display
import matplotlib.pyplot as plt
# User-friendliness
from tqdm import tqdm


"""
Data pre-processing functions
"""

def col_vec(x):
    """
    Input:
        x 1D ndarray (row vector).

    Output:
        2D ndarray (column vector).
    """
    x_col = x[:, np.newaxis]
    return x_col

def preprocess(X, Y):
    """
    Input:
        X list of 1D ndarrays,
        Y list of ints (-1s or 1s).

    Output:
        A 2D ndarray, where i-th row is X[i]*Y[i].
    """
    # Make Y a column vector
    Yc = np.array(Y)
    Yc = col_vec(Yc)
    # Multiply X[i] by Y[i]
    A = np.array(X)*Yc
    return A

"""
Game and dynamics
"""

def time_needed(margin_error, n):
    """
    returns time horizon needed to guarantee
    margin to within margin_error of (additive) error,
    for n datapoints.
    """
    # return int(np.ceil(16 * np.log(n) * margin_error**(-2)))
    return int(np.ceil(4 * (np.sqrt(np.log(n)) + np.sqrt(2 * np.log(2)))**2 * margin_error**(-2)))

def dynamics(A, T, ball_update="SCMWU", show_progress=True):
    """
    Runs game max_x min_p p^T A x, where x in ball, p in simplex.
    x-player uses SCMWU_ball, p-player uses MWU.
    """
    # Parameters
    n, d = A.shape
    MWU_stepsize = 2 * np.sqrt(np.log(n) / T)
    if ball_update == "SCMWU":
        ball_stepsize = SCMWU_ball.opt_stepsize_for_SCMWUball(T)
    else:
        ball_stepsize = SCMWU_ball.opt_stepsize_OGD(T)
    # Intialization
    p, x = MWU.init(n), SCMWU_ball.init(d)
    # print("A", A.shape)
    # print("p", p.shape)
    # print("x", x.shape)
    p_hist, x_hist = [p], [x]
    A_T = A.T
    p_sum_loss_vectors, x_sum_loss_vectors = np.zeros(n), np.zeros(d)
    # Run
    if T >= 10**4 and show_progress:
        time_series = tqdm(range(2, T+1))
    else:
        time_series = range(2, T+1)
    for t in time_series:
        p_loss_vector = A @ x
        x_loss_vector = - A_T @ p
        # p_sum_loss_vectors += p_loss_vector
        x_sum_loss_vectors += x_loss_vector
        # p = MWU.iter(MWU_stepsize, p_sum_loss_vectors)
        p = MWU.update(MWU_stepsize, p, p_loss_vector)
        if ball_update == "SCMWU":
            x = SCMWU_ball.SCMWU_iter(ball_stepsize, x_sum_loss_vectors)
        else:
            x = SCMWU_ball.OGD_iter(x, ball_stepsize, x_loss_vector)
        p_hist.append(p)
        x_hist.append(x)
    return (p_hist, x_hist)

def compute_margin(A, w):
    """
    Given (hypothesis) w, computes min_p p^T A w = min_i (Aw)_i for i in [n].
    """
    return min(A @ normalize(w))

def visualize(X, w, w_hat,
            Y=0, datapoint_color_scheme=['lightgreen', 'violet'],
            show_plot=True,
            save_plot=False,
            filepath="",
            run=0,
            classifier_color_scheme=['orange', 'blue']
            ):
    """
    NOTE: Only works for 2D visualization (d = 2).
    Inputs:
        - X: dataset, where rows are d-dim datapoints. (n, d)-ndarray.
        - w: actual normal vector used to separate dataset. (d,)-ndarray, 2norm = 1.
        - w_hat: predicted normal vector. (d,)-ndarray, 2norm <= 1.
    """
    d = len(w)
    if d != 2:
        return None
    # Plot data points
    x, y = X.T
    if Y and datapoint_color_scheme:
        colors = [datapoint_color_scheme[int((label+1)/2)] for label in Y]
        plt.scatter(x,y, s=1, c=colors)
    else:
        plt.scatter(x,y, s=1, c='lightgray')
    # Plot w, w_hat
    plt.quiver([0, 0], [0, 0], [w[0], w_hat[0]], [w[1], w_hat[1]], color=classifier_color_scheme, angles='uv', scale_units='x', scale=1)
    # Plot lines
    plt.axline([0, 0], slope=(-w[0]/w[1]), color=classifier_color_scheme[0], linestyle=(0, (5,5))) # w separator
    plt.axline([0, 0], slope=(-w_hat[0]/w_hat[1]), color=classifier_color_scheme[1], linestyle=(0, (5,5))) # w separator
    # Plot unit circle
    density = 10**2
    x_axis = np.linspace(-0.5, 0.5, density)
    y_axis = np.linspace(-0.5, 0.5, density)
    theta = np.linspace(0, 2*np.pi, 10**4)
    x0 = np.cos(theta)
    x1 = np.sin(theta)
    plt.plot(x0, x1, linewidth=1, color='gray')
    # Display plot
    plt.axis('scaled')
    if save_plot:
        filename = save_plot + "_" + str(run) + ".png"
        plt.savefig(filepath + filename, dpi=300)
    if show_plot:
        plt.show()
    plt.clf()
