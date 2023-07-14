"""
Packages
"""
import numpy as np
import LinClass_datasets as LinClass_datasets
import SVMgame
# User-friendliness
from tqdm import tqdm


"""--------------------------------------------------------------------------"""
"""
PARAMETERS
"""
# Dataset parameters
n = 10**3 # Number of data points per run
d = 2 # Dimension of data points
margin = 0.1 # Linear classification margin (guaranteed for dataset)

# Run parameters
compute_time_horizon = False #If False, use T. If True, use margin_error to compute time_needed.
if compute_time_horizon:
    margin_error = 0.05 # Margin to guarantee
    T = SVMgame.time_needed(margin_error, n)
else:
    T = 10**3 # Time Horizon (Integer)
print("T =", T)

# Number of runs
runs = 10**1

# Randomization/seed parameters
randseed = 1234 # (1234 was used for data, unless otherwise stated)

"""--------------------------------------------------------------------------"""
"""
Display and FileSave Parameters
"""
# Display parameters
plot_every = False # Plot every run? If False, only plot last run.
show_progress = True # If true, use tqdm package for sufficiently large runs.

# File save parameters
save_plots = "SVM_game" + "_d" + str(int(d)) + "_n" + str(int(n)) + "_T" + str(T) # Plot filename. If 0 or False, won't save plots.
filepath = "figs/"

"""--------------------------------------------------------------------------"""
"""
Main
"""
## Dataset generation
print("Generating datasets...")
X_data = []
Y_data = []
A_data = []
w_data = []
for run in tqdm(range(runs)):
    np.random.seed(randseed + run)
    w = LinClass_datasets.gen_rand_dir(d)
    (X, Y) = LinClass_datasets.gen_dataset(d, n,
                                    halfwidth = 1,
                                    dir = w,
                                    margin = margin)
    A = SVMgame.preprocess(X, Y)
    X = np.array(X)
    X_data.append(X)
    Y_data.append(Y)
    A_data.append(A)
    w_data.append(w)

## SVM game
# np.random.seed(randseed)
print("Number of data points, n =", n)
print("Dimension, d =", d)
print("margin =", margin)
print("time horizon, T =", T)
print("margin_SCMWU", "\t \t", "margin_OGD", "\t \t", "generated margin")
for run in range(runs):
    np.random.seed(randseed + run)
    A = A_data[run]
    p_hist, x_hist = SVMgame.dynamics(A, T, ball_update="SCMWU", show_progress=show_progress)
    p_hist_OGD, x_hist_OGD = SVMgame.dynamics(A, T, ball_update="OGD", show_progress=show_progress)
    p_ave = (1/len(p_hist)) * sum(p_hist)
    x_ave = (1/len(x_hist)) * sum(x_hist)
    p_ave_OGD = (1/len(p_hist_OGD)) * sum(p_hist_OGD)
    x_ave_OGD = (1/len(x_hist_OGD)) * sum(x_hist_OGD)
    margin_SCMWU = SVMgame.compute_margin(A, x_ave)
    margin_OGD = SVMgame.compute_margin(A, x_ave_OGD)
    margin_generated = SVMgame.compute_margin(A, w_data[run])
    print(margin_SCMWU, "\t", margin_OGD, "\t", margin_generated)
    show_plot = d == 2 and (plot_every or run==(runs - 1))
    SVMgame.visualize(np.array(X_data[run]), w_data[run], x_ave, Y=Y_data[run],
        save_plot=save_plots,
        filepath=filepath,
        show_plot=show_plot,
        run=(run+1)
        )
