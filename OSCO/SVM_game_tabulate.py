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
ds = [
    2, 3,
    5, 10
]# Dimension of data points
margin = 0.1 # Linear classification margin (guaranteed for dataset)

# Run parameters
compute_time_horizon = False #If False, use T. If True, use margin_error to compute time_needed.
if compute_time_horizon:
    margin_error = 0.05 # Margin to guarantee
    T = SVMgame.time_needed(margin_error, n)
else:
    Ts = [
    10**2, 10**3
    ] # Time Horizon (Integer)

# Number of runs
runs = 10**2

# Randomization/seed parameters
randseed = 1234 # (1234 was used for data, unless otherwise stated)

"""--------------------------------------------------------------------------"""
"""
Display/User-friendliness Parameters
"""
show_progress = True # If true, use tqdm package for sufficiently large runs.

"""--------------------------------------------------------------------------"""
"""
Main
"""
## Dataset generation
print("Generating datasets...")
X_data = {}
Y_data = {}
A_data = {}
w_data = {}
for d in ds:
    X_data[d] = []
    Y_data[d] = []
    A_data[d] = []
    w_data[d] = []
    for run in tqdm(range(runs)):
        np.random.seed(randseed + run)
        w = LinClass_datasets.gen_rand_dir(d)
        (X, Y) = LinClass_datasets.gen_dataset(d, n,
                                        halfwidth = 1,
                                        dir = w,
                                        margin = margin)
        A = SVMgame.preprocess(X, Y)
        X = np.array(X)
        X_data[d].append(X)
        Y_data[d].append(Y)
        A_data[d].append(A)
        w_data[d].append(w)

## SVM game
print("Number of data points, n =", n)
print("Dimension, ds =", ds)
print("margin =", margin)
print("time horizons, Ts =", Ts)
print("number of runs =", runs)
table_ratio = {} # mean
worst_ratio = {} # worst-case
table_error = {} # mean
worst_error = {} # worst-case
for d in ds:
    table_ratio[d] = []
    table_error[d] = []
    worst_ratio[d] = []
    worst_error[d] = []
    for T in Ts:
        table_ratio[d].append(0)
        table_error[d].append(0)
        worst_ratio[d].append(1)
        worst_error[d].append(0)
        for run in range(runs):
            np.random.seed(randseed + run)
            A = A_data[d][run]
            p_hist, x_hist = SVMgame.dynamics(A, T, ball_update="SCMWU", show_progress=show_progress)
            p_ave = (1/len(p_hist)) * sum(p_hist)
            x_ave = (1/len(x_hist)) * sum(x_hist)
            margin_SCMWU = SVMgame.compute_margin(A, x_ave)
            margin_generated = SVMgame.compute_margin(A, w_data[d][run])
            ratio = margin_SCMWU / margin_generated
            error = margin_generated - margin_SCMWU
            table_ratio[d][-1] += ratio
            table_error[d][-1] += error
            worst_ratio[d][-1] = min(worst_ratio[d][-1], ratio)
            worst_error[d][-1] = max(worst_error[d][-1], error)
        table_ratio[d][-1] /= runs
        table_error[d][-1] /= runs

print("Table of Mean Ratios")
print("\t", Ts)
for d in ds:
    print(d, "\t", table_ratio[d])

print("Table of Worst-Case Ratios")
print("\t", Ts)
for d in ds:
    print(d, "\t", worst_ratio[d])

print("Table of Mean Errors")
print("\t", Ts)
for d in ds:
    print(d, "\t", table_error[d])

print("Table of Worst-Case Errors")
print("\t", Ts)
for d in ds:
    print(d, "\t", worst_error[d])
