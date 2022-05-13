#  plot_inverse_results.py
#  This script takes the latest (from common_params.py) run and plots the results

import matplotlib.pyplot as plt
from common_params import *  # import common values across all models


def plot_inverse_results():
    # Key constants to set
    use_fwd_model = True
    subject = 'S43'
    ct_vals = []
    scenario = scenarios[0]

    # Open file and load data
    data_filename = INVOUTPUTDIR + scenario + '_fitResults_' + 'combined.npy'
    [sigmaVals, rposvals, survvals, thrsim, thrtargs, initvec, [fitrposvals, fitsurvvals],
     rposerrs, survivalerrs] = np.load(data_filename, allow_pickle=True)

    # Make plots
    xvals = np.arange(0, NELEC) + 1
    l_e = NELEC - 1  # last electrode to plot

    # All on one plot
    figrows = 3
    figcols = 1
    fig_consol, axs = plt.subplots(figrows, figcols)
    fig_consol.set_figheight(9)
    fig_consol.set_figwidth(7.5)
    axs[0].plot(xvals, thrsim[0][0], marker='o', color='blue', label='fit')
    axs[0].plot(xvals, thrtargs[0][0], marker='o', color='black', label='desired')
    axs[0].plot(xvals[1:l_e], thrsim[1][0][1:l_e], marker='s', color='blue')
    axs[0].plot(xvals[1:l_e], thrtargs[1][0][1:l_e], marker='s', color='black')
    yl = 'Threshold (dB)'
    if use_fwd_model:
        title_text = 'Known scenario thresholds: ' + scenario
    else:
        title_text = 'Subject thresholds: ' + subject
    axs[0].set(xlabel='Electrode number', ylabel=yl, title=title_text)
    axs[0].legend()

    title_text = 'Fit and actual positions'
    axs[1].plot(xvals, fitrposvals, marker='o', color='red', label='fit')
    if use_fwd_model:
        axs[1].plot(xvals, rposvals, marker='o', color='green', label='desired')
    else:
        axs[1].plot(xvals, ct_vals, marker='o', color='green', label='CT')

    axs[1].plot(xvals, initvec[:NELEC], marker='o', color='blue', label='start')

    axs[1].set(xlabel='Electrode number', ylabel='Electrode position (mm)', title=title_text)
    axs[1].legend()

    title_text = 'Fit survival values'
    axs[2].plot(xvals, fitsurvvals, marker='o', color='red', label='fit')
    axs[2].plot(xvals, initvec[NELEC:], marker='o', color='blue', label='start')
    axs[2].plot(xvals, survvals, marker='o', color='green', label='desired')
    axs[2].set(xlabel='Electrode number', ylabel='Fit survival value', title=title_text)
    axs[2].legend()
    fig_consol.tight_layout()

    # -- could add plots of error (difference between desired/measured and fitted values)

    plt.show()


if __name__ == '__main__':
    plot_inverse_results()
