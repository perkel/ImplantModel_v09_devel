# Script to make figure 3 (3D plot) for basic implant model paper on forward and inverse models

# Import required packages
import matplotlib as mpl
import matplotlib.pyplot as plt
import csv
from pylab import cm
from scipy import interpolate
from common_params import *

def Fig_2D_contour():

    # Set default figure values
    # mpl.rcParams['font.family'] = 'Helvetica'
    plt.rcParams['font.size'] = 16
    plt.rcParams['axes.linewidth'] = 2
    # Generate 2 colors from the 'tab10' colormap
    colors = cm.get_cmap('tab10', 2)

    # Declare variables
    survVals = np.arange(0.05, 0.96, 0.05)
    rposVals = np.arange(-0.95, 0.96, 0.05)

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
    rp_curt = rposVals[0:-2]
    xnew = np.linspace(rposVals[1], rposVals[-1], 50)
    ynew = np.linspace(survVals[1], survVals[-1], 50)
    XN, YN = np.meshgrid(xnew, ynew)

    f_interp = interpolate.interp2d(rp_curt, survVals, mono_thr[:, 0:-2])
    znew_mp = f_interp(xnew, ynew)
    m_min = np.min(znew_mp)
    m_max = np.max(znew_mp)

    f_interp = interpolate.interp2d(rp_curt, survVals, tripol_thr[:, 0:-2])
    znew_tp = f_interp(xnew, ynew)
    t_min = np.min(znew_tp)
    t_max = np.max(znew_tp)

    # all_min = np.min([t_min, m_min])
    # all_max = np.max([t_max, m_max])

    # rounding manually
    all_min = 20.0
    all_max = 200.0
    n_levels = 40

    fig, axs = plt.subplots(1, 3, figsize=(10, 4))
    fig.tight_layout(pad=1, w_pad=1, h_pad=3.0)
#     CS3 = axs[0].contourf(XN, YN, znew_mp, np.arange(all_min, all_max, (all_max-all_min)/n_levels))
    # without interpolation

    CS3 = axs[0].contourf(rposVals, survVals, mono_thr, np.arange(all_min, all_max, (all_max-all_min)/n_levels))

    axs[0].set_xlabel('Electrode position (mm)')
    axs[0].set_ylabel('Survival fraction')
    axs[0].set_title('Monopolar')
    #cax, kw = mpl.colorbar.make_axes([ax for ax in axs.flat])
    #plt.colorbar(CS3, cax=cax, **kw)
    # cbar3 = plt.colorbar(CS3)
    #cax.set_ylabel('Threshold (dB)')
    # # Add the contour line levels to the colorbar
    # cbar3.add_lines(CS3)
    # fig4, ax4 = plt.subplots()
#     CS4 = axs[1].contourf(XN, YN, znew_tp, np.arange(all_min, all_max, (all_max-all_min)/n_levels))
    CS4 = axs[1].contourf(rposVals, survVals, tripol_thr, np.arange(all_min, all_max, (all_max-all_min)/n_levels))
    axs[1].set_title('Tripolar')
    axs[1].set_xlabel('Electrode position (mm)')
    # ax4.set_ylabel('Survival fraction')
    # cbar4 = fig4.colorbar(CS4)
    # cbar4.ax.set_ylabel('Threshold (dB)')

    # fig2, axs2 = plt.subplots(1, 1, figsize=(10, 4))
    fig.tight_layout(pad=1, w_pad=1, h_pad=7.0)
    cax1, kw1 = mpl.colorbar.make_axes(axs[1])
    plt.colorbar(CS4, cax=cax1, **kw1)

#     CS5 = axs[2].contourf(XN, YN, znew_tp - znew_mp, np.arange(-20, 30, 1.5))
    CS5 = axs[2].contourf(rposVals, survVals, thr_diff, np.arange(-20, 50, 1.5))

    axs[2].set_title('Difference')
    axs[2].set_xlabel('Electrode position (mm)')
    cax2, kw2 = mpl.colorbar.make_axes(axs[2])
    plt.colorbar(CS5, cax=cax2, **kw2)


    plt.savefig('Fig_2D_contour.pdf', format='pdf')
    plt.show()

if __name__ == '__main__':
    Fig_2D_contour()
