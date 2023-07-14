"""
PARAMETERS
"""

"""Run parameters"""
dims = [
    (0, [], [2, 3, 4, 5, 6]),
    (5, [], [5]),
    (0, [3], [5]),
    (3, [2,3], [2,3])
]  # Dimension (3-tuple): (m, n, d), where
    # m int: dimension of R^m direct-sum component;
    # n list;
    # d list.
Ts = [
    10**5
] # Time Horizon (Integer)
doubling = True # Set to TRUE to use doubling trick, FALSE for fixed stepsize (Bool)
plot_bound = True # (Bool)

"""Multiplicity parameters"""
many_SCMWU = 100 # Number of SCMWU trajectories per plot (Integer >= 1)
runs = 1 # Number of runs, i.e., number of plots.

"""Randomization parameters"""
seed = 1234

"""--------------------------------------------------------------------------"""

"""
Display and FileSave Parameters
"""

"""Display parameters"""
legend = True # Set to TRUE to display legend (Bool)
plot_every = False # Plot every run? If False, only plot last run.
show_plot = False

"""File save parameters"""
save_plots = "SCMWU" # if "", don't save plots
filepath = "figs/"


"""--------------------------------------------------------------------------"""

"""
Packages
"""
import numpy as np
import matplotlib.pyplot as plt
# from SOC import *
from SCMWU import *
import SC_losses

# User-friendliness
from tqdm import tqdm


"""
Regret Plot functions
"""
def plot(dim, T):
    x_axis = range(1, T+1)
    for run in range(runs):
        """File save"""
        save_plots_full = save_plots + "_dim=" + str(dim) + "_T=" + str(T)
        print(save_plots_full)

        """Get losses and compute SCMWU iterates"""
        m_hists, p_hists, regret_hists = [], [], []
        for trajectory in tqdm(range(many_SCMWU)):
            # Get losses
            np.random.seed(seed + (10**3 * run) + trajectory)
            m_hist = SC_losses.gen_loss_hist_random_SC(dim, T)
            m_hists.append(m_hist)
            # Get SCMWU iterates
            if doubling:
                (p_hist, regret_hist) = SCMWU_doubling(dim, T, m_hist)
            else:
                (p_hist, regret_hist) = SCMWU_optimized(dim, T, m_hist)
            p_hists.append(p_hist)
            regret_hists.append(regret_hist)

        """Plot regret"""
        # Plot SCMWU regret bound
        if plot_bound:
            if doubling:
                bound = list(map(lambda x: regbound_doubling(dim, x),  x_axis))
            else:
                bound = list(map(lambda x: regbound(dim, T), x_axis))
            plt.plot(x_axis, bound, label='bound', color='blue', linestyle=':', linewidth=0.75)
        # Plot runs
        for traj in range(many_SCMWU):
            m_hist = m_hists[traj]
            # Plot SCMWU iterates
            p_hist = p_hists[traj]
            regret_hist = regret_hists[traj]
            if traj == 0:
                plt.plot(x_axis, regret_hist, label='SCMWU', color='blue')
            else:
                plt.plot(x_axis, regret_hist, color='blue')
        # Plot figure
        if legend:
            plt.legend(prop={'size': 16})
        plt.xlabel('Time')
        plt.ylabel('Regret')
        if save_plots:
            filename = save_plots_full + "_" + str(run+1) + ".png"
            plt.savefig(filepath + filename, dpi=300)
        show_plot_internal = show_plot and (plot_every or run==(runs - 1))
        if show_plot_internal:
            plt.show()
        plt.clf()

"""
Regret Plot
"""
for T in Ts:
    for dim in dims:
        plot(dim, T)
