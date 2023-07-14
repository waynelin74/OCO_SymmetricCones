"""
PARAMETERS
"""

"""Run parameters"""
ds = [
    10
] # list of dimensions of ball to learn over.
Ts = [
    10**4
] # Time Horizon (Integer)
doubling = False # Set to TRUE to use doubling trick, FALSE for fixed stepsize (Bool)
OGD_compare = True # Set to TRUE if want to plot OGD (Bool)
plot_bound = False # (Bool)

"""Multiplicity parameters"""
many_SCMWU = 1 # Number of SCMWU trajectories per plot (Integer >= 1)
runs = 10 # Number of runs, i.e., number of plots.

"""Randomization parameters"""
seed = 1234

"""--------------------------------------------------------------------------"""
"""
Display and FileSave Parameters
"""

"""Display parameters"""
legend = True # Set to TRUE to display legend (Bool)
plot_time_averaged_regret = False
plot_every = False # Plot every run? If False, only plot last run.
show_plot = True

"""File save parameters"""
save_plots = "SCMWUball" # if "", don't save plots
filepath = "figs/"

"""--------------------------------------------------------------------------"""

"""
Packages
"""
import numpy as np
import matplotlib.pyplot as plt
# from SOC import *
from SCMWUball import *
import SC_losses

# User-friendliness
from tqdm import tqdm


"""
Regret Plot functions
"""
def plot(d, T):
    x_axis = range(1, T+1)
    for run in range(runs):
        """File save"""
        save_plots_full = save_plots + "_d=" + str(d) + "_T=" + str(T)
        print(save_plots_full)

        """Get losses and compute SCMWU iterates"""
        m_hists, p_hists, regret_hists = [], [], []
        for trajectory in tqdm(range(many_SCMWU)):
            # Get losses
            np.random.seed(seed + (10**3 * run) + trajectory)
            m_hist = SC_losses.gen_loss_hist_random_ball(d, T)
            m_hists.append(m_hist)
            # Get SCMWU iterates
            if doubling:
                (p_hist, regret_hist) = SCMWUball_doubling(d, T, m_hist)
            else:
                (p_hist, regret_hist) = SCMWUball_optimized(d, T, m_hist)
            p_hists.append(p_hist)
            regret_hists.append(regret_hist)

        """Plot regret"""
        # Plot SCMWU regret bound
        if plot_bound:
            if doubling:
                bound = list(map(lambda x: regbound_doubling(x),  x_axis))
            else:
                bound = list(map(lambda x: regbound(T), x_axis))
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
            # Plot OGD
            if OGD_compare:
                (p_hist_OGD, regret_hist_OGD) = OGD_optimized(d, T, m_hist)
                plt.plot(x_axis, regret_hist_OGD, label='OGD', color='green')
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

        """Plot time-averaged regret"""
        if doubling and plot_time_averaged_regret:
            # Plot SCMWU regret bound
            if plot_bound:
                bound = list(map(lambda x: np.log(regbound_doubling(dim, x) / x),  x_axis))
                plt.plot(x_axis, bound, label='bound', color='blue', linestyle=':', linewidth=0.75)
            # Plot runs
            for traj in range(many_SCMWU):
                m_hist = m_hists[traj]
                # Plot SCMWU iterates
                p_hist = p_hists[traj]
                regret_hist = regret_hists[traj]
                log_time_ave_regret_hist = []
                for t in x_axis:
                    log_time_ave_regret_hist.append(np.log(regret_hist[t-1] / t))
                if traj == 0:
                    plt.plot(x_axis, log_time_ave_regret_hist, label='SCMWU', color='blue')
                else:
                    plt.plot(x_axis, log_time_ave_regret_hist, color='blue')
                # Plot OGD
                if OGD_compare:
                    (p_hist_OGD, regret_hist_OGD) = OGD_optimized(d, T, m_hist)
                    log_time_ave_regret_hist_OGD = []
                    for t in x_axis:
                        log_time_ave_regret_hist_OGD.append(np.log(regret_hist_OGD / t))
                    plt.plot(x_axis, log_time_ave_regret_hist_OGD, label='OGD', color='green')
            # Plot figure
            if legend:
                plt.legend(prop={'size': 16})
            plt.xlabel('Time')
            plt.ylabel('log (time-averaged regret)')
            if save_plots:
                filename = save_plots_full + "_TimeAve_" + str(run+1) + ".png"
                plt.savefig(filepath + filename, dpi=300)
            show_plot_internal = show_plot and (plot_every or run==(runs - 1))
            if show_plot_internal:
                plt.show()
            plt.clf()

"""
Regret Plot
"""
if OGD_compare:
    save_plots += "vOGD"

for T in Ts:
    for d in ds:
        plot(d, T)
