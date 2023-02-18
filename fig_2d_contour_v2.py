# Script to make figure 3 (3D plot) for basic implant model paper on forward and inverse models

# Import required packages
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import csv
from scipy import interpolate
import intersection as intsec
from common_params import *


def fig_2D_contour():
    # Set default figure values
    # mpl.rcParams['font.family'] = 'Helvetica'
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.linewidth'] = 2

    # Declare variables
    surv_vals = np.arange(0.04, 0.97, 0.02)
    rpos_vals = np.arange(-0.95, 0.96, 0.02)

    # Load monopolar data
    datafile = FWDOUTPUTDIR + "Monopolar_2D_" + STD_TEXT + ".csv"
    file = open(datafile)
    numlines = len(file.readlines())
    file.close()

    with open(datafile, newline='') as csvfile:
        datareader = csv.reader(csvfile, delimiter=',')
        ncol = len(next(datareader))
        csvfile.seek(0)
        mono_thr = np.empty([numlines, ncol])
        for i, row in enumerate(datareader):
            # Do the parsing
            mono_thr[i, :] = row

    print("average monopolar threshold: ", np.nanmean(mono_thr))

    # Load tripolar data
    datafile = FWDOUTPUTDIR + "Tripolar_09_2D_" + STD_TEXT + ".csv"
    file = open(datafile)
    numlines = len(file.readlines())
    file.close()

    with open(datafile, newline='') as csvfile:
        datareader = csv.reader(csvfile, delimiter=',')
        ncol = len(next(datareader))
        csvfile.seek(0)
        tripol_thr = np.empty([numlines, ncol])
        for i, row in enumerate(datareader):
            # Do the parsing
            tripol_thr[i, :] = row

    print('Data retrieved from directory: ', FWDOUTPUTDIR)

    # Measure min/max/mean differences between monopolar and tripolar
    thr_diff = tripol_thr - mono_thr
    mean_diff = np.mean(thr_diff[:])
    min_diff = np.min(thr_diff[:])
    max_diff = np.max(thr_diff[:])
    print('Min/max/mean differences: ', min_diff, ' , ', max_diff, ' , ', mean_diff)

    # # set up 2D interpolation
    rp_curt = rpos_vals[0:-2]
    xnew = np.linspace(rpos_vals[1], rpos_vals[-1], 50)
    ynew = np.linspace(surv_vals[1], surv_vals[-1], 50)
    np.meshgrid(xnew, ynew)

    f_interp = interpolate.interp2d(rp_curt, surv_vals, mono_thr[:, 0:-2])
    znew_mp = f_interp(xnew, ynew)
    m_min = np.min(znew_mp)
    m_max = np.max(znew_mp)

    f_interp = interpolate.interp2d(rp_curt, surv_vals, tripol_thr[:, 0:-2])
    znew_tp = f_interp(xnew, ynew)
    t_min = np.min(znew_tp)
    t_max = np.max(znew_tp)

    all_min = np.min([t_min, m_min])
    all_max = np.max([t_max, m_max])

    # rounding manually
    all_min = 20.0
    all_max = 75.0
    n_levels = 100
    labels = ['P1', 'P2', 'P3', 'P4']

    fig, axs = plt.subplots(1, 3, figsize=(8, 8), gridspec_kw={'width_ratios': [8, 8, 1]})
    fig.tight_layout(pad=3, w_pad=2, h_pad=2.0)
    #     CS3 = axs[0].contourf(XN, YN, znew_mp, np.arange(all_min, all_max, (all_max-all_min)/n_levels))
    # without interpolation

    CS3 = axs[0].contourf(1 - rpos_vals, surv_vals, mono_thr,
                             np.arange(all_min, all_max, (all_max - all_min) / n_levels),
                             cmap='viridis', extend='both')
    # CS3 = axs[0].contourf(rpos_vals, surv_vals, mono_thr, cmap='hot')
    low_rpos_val = -0.5
    high_rpos_val = 0.5
    low_surv_val = 0.4
    high_surv_val = 0.8
    axs[0].set_xlabel('Electrode distance (mm)')
    axs[0].set_ylabel('Fractional neuronal density')
    axs[0].set_title('Monopolar', fontsize=12)
    lab_shift = 0.025
    axs[0].text(1 - high_rpos_val + lab_shift, high_surv_val, labels[0], horizontalalignment='left',
                   verticalalignment='bottom')
    axs[0].text(1 - low_rpos_val + lab_shift, high_surv_val, labels[1], horizontalalignment='left',
                   verticalalignment='bottom')
    axs[0].text(1 - high_rpos_val + lab_shift, low_surv_val, labels[2], horizontalalignment='left',
                   verticalalignment='bottom')
    axs[0].text(1 - low_rpos_val + lab_shift, low_surv_val, labels[3], horizontalalignment='left',
                   verticalalignment='bottom')
    # axs[0, 0].text(-0.5, 1.0, 'A', fontsize=20)
    axs[0].plot([np.min(1 - rpos_vals), np.max(1 - rpos_vals)], [high_surv_val, high_surv_val], color='blue')
    axs[0].plot([np.min(1 - rpos_vals), np.max(1 - rpos_vals)], [low_surv_val, low_surv_val], color='blue',
                   linestyle='dashed')
    axs[0].plot([1 - low_rpos_val, 1 - low_rpos_val], [np.min(surv_vals), np.max(surv_vals)], color='black',
                   linestyle='dashed')
    axs[0].plot([1 - high_rpos_val, 1 - high_rpos_val], [np.min(surv_vals), np.max(surv_vals)], color='black')
    axs[0].set_xticks([0.4, 0.8, 1.2, 1.6])

    #    axs[0].plot([0.5, 0.5, 1.5, 1.5], [0.5, 0.8, 0.8, 0.5], 'ok', mfc='none')
    # cax, kw = mpl.colorbar.make_axes([ax for ax in axs.flat])
    # plt.colorbar(CS3, cax=cax, **kw)
    # cbar3 = plt.colorbar(CS3)
    # cax.set_ylabel('Threshold (dB)')
    # # Add the contour line levels to the colorbar
    # cbar3.add_lines(CS3)
    # fig4, ax4 = plt.subplots()
    #     CS4 = axs[1].contourf(XN, YN, znew_tp, np.arange(all_min, all_max, (all_max-all_min)/n_levels))
    cs4 = axs[1].contourf(1 - rpos_vals, surv_vals, tripol_thr,
                             np.arange(all_min, all_max, (all_max - all_min) / n_levels),
                             cmap='viridis', extend='both')
    axs[1].set_title('Tripolar', fontsize=12)
    axs[1].set_xlabel('Electrode distance (mm)')
    axs[1].text(1 - high_rpos_val + lab_shift, high_surv_val, labels[0], horizontalalignment='left',
                   verticalalignment='bottom')
    axs[1].text(1 - low_rpos_val + lab_shift, high_surv_val, labels[1], horizontalalignment='left',
                   verticalalignment='bottom')
    axs[1].text(1 - high_rpos_val + lab_shift, low_surv_val, labels[2], horizontalalignment='left',
                   verticalalignment='bottom')
    axs[1].text(1 - low_rpos_val + lab_shift, low_surv_val, labels[3], horizontalalignment='left',
                   verticalalignment='bottom')
    axs[1].plot([np.min(1 - rpos_vals), np.max(1 - rpos_vals)], [high_surv_val, high_surv_val], color='red')
    axs[1].plot([np.min(1 - rpos_vals), np.max(1 - rpos_vals)], [low_surv_val, low_surv_val], color='red',
                   linestyle='dashed')
    axs[1].plot([1 - low_rpos_val, 1 - low_rpos_val], [np.min(surv_vals), np.max(surv_vals)], color='gray',
                   linestyle='dashed')
    axs[1].plot([1 - high_rpos_val, 1 - high_rpos_val], [np.min(surv_vals), np.max(surv_vals)], color='gray')
    axs[1].set_xticks([0.4, 0.8, 1.2, 1.6])

    # Make colorbar
    cbar = fig.colorbar(cs4, ax=axs[2], ticks=range(25, 80, 25))

    # ax4.set_ylabel('Survival fraction')
    # cbar4 = fig4.colorbar(CS4)
    # cbar4.ax.set_ylabel('Threshold (dB)')

    # fig2, axs2 = plt.subplots(1, 1, figsize=(10, 4))
    # cax1, kw1 = mpl.colorbar.make_axes(axs[1])
    # cbar = plt.colorbar(CS4, ax = axs[1])
    # cbar.ax.set_yticks(np.arange(40, 121, 40))

    #     CS5 = axs[2].contourf(XN, YN, znew_tp - znew_mp, np.arange(-20, 30, 1.5))

    #  calculate  indices from target survival and rpos values
    low_rpos_idx = np.argmin(np.abs(rpos_vals - low_rpos_val))
    high_rpos_idx = np.argmin(np.abs(rpos_vals - high_rpos_val))
    low_surv_idx = np.argmin(np.abs(surv_vals - low_surv_val))
    high_surv_idx = np.argmin(np.abs(surv_vals - high_surv_val))

    # Lower panels or perhaps new figure
    # axs[1, 0].plot(1 - rpos_vals, mono_thr[high_surv_idx, :], color='blue', linestyle='solid')
    # axs[1, 0].plot(1 - rpos_vals, tripol_thr[high_surv_idx, :], color='red', linestyle='solid')
    # axs[1, 0].plot(1 - rpos_vals, mono_thr[low_surv_idx, :], color='blue', linestyle='dashed')
    # axs[1, 0].plot(1 - rpos_vals, tripol_thr[low_surv_idx, :], color='red', linestyle='dashed')
    # axs[1, 0].axes.set_xlabel('Electrode distance (mm)')
    # axs[1, 0].axes.set_ylabel('Threshold (dB)')
    # axs[1, 0].axes.set_xlim([0.1, 1.9])
    # axs[1, 0].axes.set_ylim([10, 75])
    # axs[1, 0].set_xticks([0.4, 0.8, 1.2, 1.6])
    #
    # axs[1, 1].plot(surv_vals, mono_thr[:, high_rpos_idx], color='black', linestyle='solid')
    # axs[1, 1].plot(surv_vals, tripol_thr[:, high_rpos_idx], color='gray', linestyle='solid')
    # axs[1, 1].plot(surv_vals, mono_thr[:, low_rpos_idx], color='black', linestyle='dashed')
    # axs[1, 1].plot(surv_vals, tripol_thr[:, low_rpos_idx], color='gray', linestyle='dashed')
    # axs[1, 1].axes.set_xlabel('Fractional neuronal density')
    # axs[1, 1].axes.set_xlim([0.1, 0.9])
    # axs[1, 1].axes.set_ylim([10, 75])

    # axs[2].plot(labels, [mono_thr[high_surv_idx, high_rpos_idx], mono_thr[high_surv_idx, low_rpos_idx],
    #                                    mono_thr[low_surv_idx, high_rpos_idx], mono_thr[low_surv_idx, low_rpos_idx]],
    #             'ok', mfc='none', label='monopolar')
    # axs[2].plot(labels, [tripol_thr[high_surv_idx, high_rpos_idx],
    #                                    tripol_thr[high_surv_idx, low_rpos_idx],
    #                                    tripol_thr[low_surv_idx, high_rpos_idx],
    #                                    tripol_thr[low_surv_idx, low_rpos_idx]],
    #             '^k', mfc='none', label='tripolar')
    # axs[2].set_xlabel('Parameters')
    # axs[2].set_ylabel('Threshold (dB)')
    # axs[2].legend(loc='upper left', bbox_to_anchor=(0, 0.99), fontsize='x-small')
    # axs[2].set_ylim(28, 72)
    # axs[2].text(-0.5, 75, 'B', fontsize=20)
    # fig.tight_layout(pad=1, w_pad=2, h_pad=4.0)

    plt.savefig('Fig_2D_contour.pdf', format='pdf')

    # Now make next figure with example contours
    figrows = 2
    figcols = 2
    fig2, axs2 = plt.subplots(figrows, figcols)
    idx_surv = 0
    idx_rpos = 0
    the_ax = 0
    for i in range(4):
        if i == 0:
            idx_surv = high_surv_idx
            idx_rpos = high_rpos_idx
            the_ax = axs2[0, 0]
        elif i == 1:
            idx_surv = high_surv_idx
            idx_rpos = low_rpos_idx
            the_ax = axs2[0, 1]
        elif i == 2:
            idx_surv = low_surv_idx
            idx_rpos = high_rpos_idx
            the_ax = axs2[1, 0]
        elif i == 3:
            idx_surv = low_surv_idx
            idx_rpos = low_rpos_idx
            the_ax = axs2[1, 1]

        this_mp_thr = [mono_thr[idx_surv, idx_rpos]]
        cont_mp = the_ax.contour(1 - rpos_vals, surv_vals, mono_thr, this_mp_thr, colors='blue')
        if i == 2 or i == 3:
            the_ax.axes.set_xlabel('Electrode distance (mm)')

        if i == 0:
            the_ax.axes.set_ylabel('Fractional neuronal density')
            the_ax.yaxis.set_label_coords(-0.2, -0.08)
        this_tp_thr = [tripol_thr[idx_surv, idx_rpos]]
        cont_tp = the_ax.contour(1 - rpos_vals, surv_vals, tripol_thr, this_tp_thr, colors='red')
        mpcontour = cont_mp.allsegs[0]
        tpcontour = cont_tp.allsegs[0]
        nmp = len(mpcontour[0])
        ntp = len(tpcontour[0])
        mpx = np.zeros(nmp)
        mpy = np.zeros(nmp)
        tpx = np.zeros(ntp)
        tpy = np.zeros(ntp)

        for j in range(0, nmp):  # Should be able to do this without for loops
            mpx[j] = mpcontour[0][j][0]
            mpy[j] = mpcontour[0][j][1]

        for j in range(0, ntp):
            tpx[j] = tpcontour[0][j][0]
            tpy[j] = tpcontour[0][j][1]

        x, y = intsec.intersection(mpx, mpy, tpx, tpy)  # find intersection(s)
        the_ax.plot(x[-1], y[-1], 'x', color='black', markersize='10', mew=2.5)
        if i > 0:
            the_ax.plot(x[0], y[0], 'x', color='gray', markersize='8')
        the_ax.set_xlim([0, 1.9])
        the_ax.text(0.1, 0.8, labels[i], fontsize=16)

    # fig3, ax4 = plt.subplots()
    # ax4 = plt.contour(1-rpos_vals, surv_vals, mono_thr, [mono_thr[low_surv_idx, high_rpos_idx]],
    #                   colors='green')
    # ax4.axes.set_xlabel('Electrode distance (mm)')
    # ax4.axes.set_ylabel('Fractional neuronal density')
    # ax5 = plt.contour(1-rpos_vals, surv_vals, tripol_thr, [tripol_thr[low_surv_idx, high_rpos_idx]],
    #                   colors='red')
    # mpcontour = ax4.allsegs[0]
    # tpcontour = ax5.allsegs[0]
    # nmp = len(mpcontour[0])
    # ntp = len(tpcontour[0])
    # mpx = np.zeros(nmp)
    # mpy = np.zeros(nmp)
    # tpx = np.zeros(ntp)
    # tpy = np.zeros(ntp)
    #
    # for j in range(0, nmp):  # Should be able to do this without for loops
    #     mpx[j] = mpcontour[0][j][0]
    #     mpy[j] = mpcontour[0][j][1]
    #
    # for j in range(0, ntp):
    #     tpx[j] = tpcontour[0][j][0]
    #     tpy[j] = tpcontour[0][j][1]
    #
    # x, y = intsec.intersection(mpx, mpy, tpx, tpy)  # find intersection(s)
    # plt.plot(x[1], y[1], 'x', color='black', markersize='12')

    plt.show()


if __name__ == '__main__':
    fig_2D_contour()
