# FigContourExamples.py

import intersection as intsec
from common_params import *  # import common values across all models
import set_scenario as s_scen
import os
import csv
import load_fwd_csv_data as lcsv
import matplotlib.pyplot as plt
from scipy import interpolate

if not os.path.isdir(INVOUTPUTDIR):
    os.mkdir(INVOUTPUTDIR)

# First make sure that the 2D forward model has been run and load the data
# It would be ideal to double check that it's the correct 2D data with identical parameters
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

# Now hold these data until about to fit a particular combination of monopolar and tripolar threshold values
# to see if there is more than one solution
surv_grid_vals = np.arange(0.04, 0.97, 0.02)
rpos_grid_vals = np.arange(-0.95, 0.96, 0.02)

scenario = scenarios[0]
[survvals, rposvals] = s_scen.set_scenario(scenario, NELEC)
csv_file = FWDOUTPUTDIR + 'FwdModelOutput_' + scenario + '.csv'
[thr_data, ct_data] = lcsv.load_fwd_csv_data(csv_file)

fig, axs = plt.subplots(1, 4)
fig.set_figheight(3)
fig.set_figwidth(12)

axs[0].axes.set_ylabel('Fractional neuronal density')

for i, electrode in enumerate([2, 6, 10, 14]):  # Fit params for each electrode (except ends, where there is no tripol value)
    mptarg = thr_data['thrmp_db'][electrode]
    tptarg = thr_data['thrtp_db'][electrode]
    # Get contours and find intersection to find initial guess for overall fitting
    rp_curt = rpos_grid_vals[0:-2]  # curtailed rpos values
    f_interp_mp = interpolate.interp2d(rp_curt, surv_grid_vals, mono_thr[:, 0:-2])
    f_interp_tp = interpolate.interp2d(rp_curt, surv_grid_vals, tripol_thr[:, 0:-2])
    xnew = np.linspace(rpos_grid_vals[0], rpos_grid_vals[-2], 50)
    ynew = np.linspace(surv_grid_vals[0], surv_grid_vals[-2], 50)
    xn, yn = np.meshgrid(xnew, ynew)
    znew_mp = f_interp_mp(xnew, ynew)
    znew_tp = f_interp_tp(xnew, ynew)

    # Note converting rpos to distance from inner wall, calling it "Electrode distance"
    cs_mono = axs[i].contour(1-xn, yn, znew_mp, [mptarg], colors='blue', label='monopolar')
    cs_tri = axs[i].contour(1-xn, yn, znew_tp, [tptarg], colors='red', label='tripolar')
    # axs[i].legend(['monopolar', 'tripolar'])
    axs[i].axes.set_xlabel('Electrode distance (mm)')
    axs[i].set_xticks([0.4, 0.8, 1.2, 1.6])
    titletext = "Electrode " + str(electrode+1)
    axs[i].title.set_text(titletext)
    axs[i].spines.right.set_visible(False)
    axs[i].spines.top.set_visible(False)
    mpcontour = cs_mono.allsegs[0]  # Contour points in rpos x survival space that give this threshold
    tpcontour = cs_tri.allsegs[0]
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
    print("one solution: ", x, y)
    rp_guess = x
    sv_guess = y
    axs[i].plot(rp_guess, sv_guess, 'ko', mfc='none', markersize=12)

plt.show()
