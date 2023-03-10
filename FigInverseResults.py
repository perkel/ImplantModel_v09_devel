#  fig_inverse_results.py
#  This script takes the latest (from common_params.py) run and plots the results

import matplotlib.pyplot as plt
from common_params import *  # import common values across all models
import subject_data



def fig_inverse_results():
    # Key constants to set
    use_fwd_model = True
    if use_fwd_model:
        figscenarios = ['Gradual80R00', 'RampRposS80', 'RampRposSGradual80']

    # All on one plot
    figrows = 3
    figcols = 3
    ncols = 3
    fig_consol, axs = plt.subplots(figrows, figcols)
    fig_consol.set_figheight(8)
    fig_consol.set_figwidth(11)
    xvals = np.arange(0, NELEC) + 1
    l_e = NELEC - 1  # last electrode to plot

    # Loop on datafiles
    for i, scenario in enumerate(figscenarios):

        ax = axs.flat[i]
        # Open file and load data
        data_filename = INVOUTPUTDIR + scenario + '_fitResults_' + 'combined.npy'

        data = np.load(data_filename, allow_pickle=True)
        if use_fwd_model:
            [sigmaVals, rposvals, survvals, thrsim, thrtargs, initvec, [fitrposvals, fitsurvvals], rposerrs,
             rpos_err_metric, survivalerr] = data
        else:
            [sigmaVals, rposvals, survvals, thrsim, thrtargs, initvec, [fitrposvals, fitsurvvals], rposerrs,
             rpos_err_metric, survivalerrs, ct_vals] = data


        # Make plots
        ax.plot(xvals[1:l_e] - 0.2, thrsim[1][0][1:l_e], marker='^', color='purple', label='TP fit')
        ax.plot(xvals[1:l_e] + 0.2, thrtargs[1][0][1:l_e], marker='^', color='orange', label='TP actual')
        ax.plot(xvals - 0.2, thrsim[0][0], marker='o', color='purple', label='MP fit')
        ax.plot(xvals + 0.2, thrtargs[0][0], marker='o', color='orange', label='MP actual')
        # title_text = scenario
        # ax.set_title(title_text)
        ax.spines.right.set_visible(False)
        ax.spines.top.set_visible(False)
        xlim = ax.get_xlim()
        ax.set_xticks([2, 4, 6, 8, 10, 12, 14])
        ax.set_ylim(20, 70)
        if i == 0:
            ax.set_ylabel('Threshold (dB)', font='Helvetica')

        if i == 2:
            ax.legend()

        ax = axs.flat[i + ncols]
        # Note converting rpos to distance from inner wall, calling it "Electrode distance"
        ax.plot(xvals[1:l_e] + 0.2, 1-rposvals[1:l_e], marker='o', color='orange', label='actual')
        ax.plot(xvals[1:l_e] - 0.2, 1-fitrposvals[1:l_e], marker='o', color='purple', label='estimated')

        ax.spines.right.set_visible(False)
        ax.spines.top.set_visible(False)
        ax.set_xlim(xlim)
        ax.set_xticks([2, 4, 6, 8, 10, 12, 14])
        ax.set_ylim(0.2, 1.7)
        if i == 0:
            ax.set(ylabel='Electrode distance (mm)')
            ax.legend()

        ax = axs.flat[i + 2*ncols]
        ax.plot(xvals[1:l_e] + 0.2, survvals[1:l_e], marker='o', color='orange', label='actual')
        ax.plot(xvals[1:l_e] - 0.2, fitsurvvals[1:l_e], marker='o', color='purple', label='estimated')
        ax.set(xlabel='Electrode number')
        ax.spines.right.set_visible(False)
        ax.spines.top.set_visible(False)
        ax.set_xlim(xlim)
        ax.set_xticks([2, 4, 6, 8, 10, 12, 14])
        ax.set_ylim(0.25, 0.95)
        if i == 0:
            ax.set(ylabel='Fractional neuronal density')
            # ax.legend()
        if i == ncols - 1:
            ax.set(xlabel='Electrode number')

        fig_consol.tight_layout()

    # -- could add plots of error (difference between desired/measured and fitted values)

    plt.show()


if __name__ == '__main__':
    fig_inverse_results()
