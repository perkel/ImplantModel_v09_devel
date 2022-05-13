# InverseModelCombined.py
# Script to fit TP and MP threshold data to the Goldwyn, Bierer, and
# Bierer cochlear activation model. The code takes as a starting point the
# CIAP 2013 version of the model, as initially worked on in the "2014 Model Reboot" directory.
# This version was intended for the 2015 CIAP meeting. Threshold and CT
# data are now brought in from the EFI ladder network batch file.

# 2016.05.13: This version trying to break up electrode and neural fitting into
# different parts. Start with exact CT-estimated fits, update neural survival
# separately for MP and TP; then fit both simultaneously while allowing both
# electrode and survival to vary, with initial conditions derived from first step.

# Translated to python 3 and adapted to fit both survival and rpos values by David Perkel December 2020

# Modified 6 August 2021 by DJP to start by fitting a single electrode at a time. This gives initial conditions
# for each electrode. Then there's a holistic fitting process using all electrode parameters

import set_scenario as s_scen
import survFull
import getThresholds as gT
import csv
import os
import scipy.signal as sig
import matplotlib.pyplot as plt
import pickle
import load_fwd_csv_data as lcsv
from scipy import interpolate
from lmfit import Minimizer, Parameters, report_fit
import intersection as intsec
from common_params import *  # import common values across all models
import subject_data
import PlotInverseResults

# Which variable(s) to fit?
fit_mode = 'combined'  # alternatives are 'combined', 'rpos' or 'survival'

ifPlot = True  # Whether to plot output at end
use_fwd_model = True  # Whether to use output from the forward model or, alternatively, from subject
SUBJECT = 'S43'


# For optimizing fit to thresholds need e_field, sim_params, sigvals
# This is for a single electrode
def objectivefunc_lmfit(p, sigvals, sim_params, f_par, e_field, thr_goals, this_elec):
    nel = len(sim_params['electrodes']['zpos'])
    vals = p.valuesdict()
    show_retval = False

    sim_params['electrodes']['rpos'] = vals['rpos_val']
    tempsurv = np.zeros(nel)
    tempsurv[:] = vals['surv_val']
    print('tempsurv = ', tempsurv)

    sim_params['neurons']['nsurvival'] = survFull.survFull(sim_params['electrodes']['zpos'],
                                                           tempsurv, simParams['grid']['z'])

    # Call for monopolar then tripolar
    sim_params['channel']['sigma'] = sigvals[0]
    thresh_mp = gT.getThresholds(e_field, f_par, sim_params)
    sim_params['channel']['sigma'] = sigvals[1]
    thresh_tp = gT.getThresholds(e_field, f_par, sim_params)
    mp_err = np.abs(thresh_mp[0] - thr_goals['thrmp_db'][this_elec])
    tp_err = np.abs(thresh_tp[0] - thr_goals['thrtp_db'][this_elec])
    if np.isnan(tp_err):
        tp_err = 0.0
    mean_error = (mp_err + tp_err) / 2.0
    if show_retval:
        print('Mean error (dB) = ', mean_error)

    retval = [mp_err, tp_err]
    return retval


# For optimizing fit to thresholds need e_field, sim_params, sigvals
# This is for all electrodes at once
def objectivefunc_lmfit_all(par, sigvals, sim_params, f_par, e_field, thr_goals):
    # Repack parameters into arrays
    nel = len(sim_params['electrodes']['zpos'])
    vals = par.valuesdict()
    show_retval = True

    sim_params['electrodes']['rpos'] = np.zeros(nel)
    for i in range(0, nel):
        varname = 'v_%i' % i
        myvalue = vals[varname]
        sim_params['electrodes']['rpos'][i] = myvalue

    tempsurv = np.zeros(nel)
    for i, loopval in enumerate(range(nel, 2 * nel)):
        varname = 'v_%i' % (i + nel)
        myvalue = vals[varname]
        tempsurv[i] = myvalue

    sim_params['neurons']['nsurvival'] = survFull.survFull(sim_params['electrodes']['zpos'],
                                                           tempsurv, simParams['grid']['z'])

    # Call for monopolar then tripolar
    sim_params['channel']['sigma'] = sigvals[0]
    thresh_mp = gT.getThresholds(e_field, f_par, sim_params)
    sim_params['channel']['sigma'] = sigvals[1]
    thresh_tp = gT.getThresholds(e_field, f_par, sim_params)
    # Calculate errors
    mp_err = np.nanmean(np.abs(np.subtract(thresh_mp[0], thr_goals['thrmp_db'])))
    tp_err = np.nanmean(np.abs(np.subtract(thresh_tp[0][1:nel - 1], thr_goals['thrtp_db'][1:nel - 1])))
    mean_error = (mp_err + tp_err) / 2.0
    if show_retval:
        print('tempsurv[4] = ', tempsurv[4], ' rpos[4]= ', sim_params['electrodes']['rpos'][4],
              ' Mean error (dB) = ', mean_error)
    # retval = np.append(np.subtract(thresh_mp[0], thr_goals['thrmp_db']), 0)
    mp_diff = np.subtract(thresh_mp[0], thr_goals['thrmp_db'])
    tp_diff = np.subtract(thresh_tp[0][1:nel - 1], thr_goals['thrtp_db'][1:nel - 1])
    tempzero = np.zeros(1)
    retval = np.concatenate((mp_diff, tempzero, tp_diff, tempzero))
    # Returns a vector of errors, with the first and last of the tripolar errors set to zero
    # because they can't be calculated
    return retval


def find_closest(x1, y1, x2, y2):  # returns indices of the point on each curve that are closest
    # Brute force (pretty ugly, but hopefully a rare case)
    n1 = len(x1)
    n2 = len(x2)

    min_dist = 5.0
    min_idx = [0, 0]
    for ii in range(n1):
        for jj in range(n2):
            dist = np.sqrt((((x1[ii] - x2[jj]) / 2) ** 2) + ((y1[ii] - y2[jj]) ** 2))
            if dist < min_dist:
                min_idx = [ii, jj]
                min_dist = dist

    return min_idx[0]


def inverse_model_combined_se():  # Start this script
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
    surv_grid_vals = np.arange(0.05, 0.96, 0.05)
    rpos_grid_vals = np.arange(-0.95, 0.96, 0.05)

    # Open field data and load data
    if "fieldTable" not in locals():
        with open(FIELDTABLE, 'rb') as combined_data:
            data = pickle.load(combined_data)
            combined_data.close()

            #  load(FIELDTABLE) # get model voltage/activating tables, if not already loaded
            # (output is 'fieldTable' and 'fieldParams')
            fp = data[0]
            # Temp fixup
            fp['zEval'] = np.array(fp['zEval'])
            act_vals = data[2]  # the data[1] has voltage values, which we are ignoring here
            simParams['grid']['table'] = act_vals

    for scen in range(0, len(scenarios)):
        scenario = scenarios[scen]
        if use_fwd_model:
            [survvals, rposvals] = s_scen.set_scenario(scenario, NELEC)
            csv_file = FWDOUTPUTDIR + 'FwdModelOutput_' + scenario + '.csv'
            [thr_data, ct_data] = lcsv.load_fwd_csv_data(csv_file)
        else:  # use threshold data from a subject
            ELECTRODES['rpos'] = np.zeros(NELEC)
            rposvals = ELECTRODES['rpos']
            thr_data = {'thrmp_db': (subject_data.subj_thr_data(SUBJECT))[0], 'thrmp': [],
                        'thrtp_db': (subject_data.subj_thr_data(SUBJECT))[1], 'thrtp': [], 'thrtp_sigma': 0.9}
            thr_data['thrtp_db'] = np.insert(thr_data['thrtp_db'], 0, np.NaN)
            thr_data['thrtp_db'] = np.append(thr_data['thrtp_db'], np.NaN)

            mp_offset_db = np.nanmean(thr_data['thrmp_db']) - np.nanmean(mono_thr)
            tp_offset_db = np.nanmean(thr_data['thrtp_db']) - np.nanmean(tripol_thr)
            overall_offset_db = np.mean([mp_offset_db, tp_offset_db])
            thr_data['thrmp_db'] -= overall_offset_db
            thr_data['thrtp_db'] -= overall_offset_db

            survvals = np.empty(NELEC)
            survvals[:] = np.nan
            ct_data = {'stdiameter': [], 'scala': [], 'elecdist': [], 'espace': 1.1, 'type': [], 'insrt_base': [],
                       'insert_apex': []}
            radius = 1.0
            ct_data['stdiameter'] = radius * 2.0 * (np.zeros(NELEC) + 1.0)

        #  rposvals = ELECTRODES['rpos']  # save this for later
        saverposvals = rposvals

        cochlea_radius = ct_data['stdiameter'] / 2.0
        if np.isnan(thr_data['thrtp_sigma']) or thr_data['thrtp_sigma'] < 0.75 or thr_data['thrtp_sigma'] > 1:
            print('The sigma value for the TP configuration is invalid.')
            exit()

        fradvec = (ct_data['stdiameter'] / 2)  # smooth the radius data!!
        fradvec = sig.filtfilt(np.hanning(5) / sum(np.hanning(5)), 1, fradvec)
        simParams['cochlea']['radius'] = fradvec
        avec = np.arange(0, 1.005, 0.01)  # create the neuron count to neuron spikes transformation
        rlvec = NEURONS['coef'] * (np.power(avec, 2.0)) + (1 - NEURONS['coef']) * avec
        rlvec = 100 * np.power(rlvec, NEURONS['power'])
        rlvltable = np.stack((avec, rlvec))
        simParams['neurons']['rlvl'] = rlvltable

        # Construct the simParams structure
        simParams['cochlea'] = COCHLEA
        simParams['electrodes'] = ELECTRODES
        simParams['channel'] = CHANNEL
        simParams['channel']['sigma'] = 0.0
        thresholds = np.empty(NELEC)  # Array to hold threshold data for different simulation values
        thresholds[:] = np.nan
        fitrposvals = np.zeros(NELEC)
        fitsurvvals = np.zeros(NELEC)
        par = Parameters()
        initvec = []

        if fit_mode == 'combined':  # Optimize survival and rpos to match MP and TP thresholds
            # Loop on electrodes, fitting rpos and survival fraction at each location
            nsols = np.zeros(NELEC)  # number of solutions found from the 2D maps

            for i in range(1, NELEC - 1):  # Fit params for each electrode (except ends, where there is no tripol value)
                mptarg = thr_data['thrmp_db'][i]
                tptarg = thr_data['thrtp_db'][i]

                # Get contours and find intersection to find initial guess for overall fitting
                fig3, ax3 = plt.subplots()
                rp_curt = rpos_grid_vals[0:-2]  # curtailed rpos values
                f_interp_mp = interpolate.interp2d(rp_curt, surv_grid_vals, mono_thr[:, 0:-2])
                f_interp_tp = interpolate.interp2d(rp_curt, surv_grid_vals, tripol_thr[:, 0:-2])
                xnew = np.linspace(rpos_grid_vals[0], rpos_grid_vals[-2], 50)
                ynew = np.linspace(surv_grid_vals[0], surv_grid_vals[-2], 50)
                xn, yn = np.meshgrid(xnew, ynew)
                znew_mp = f_interp_mp(xnew, ynew)
                znew_tp = f_interp_tp(xnew, ynew)
                ax3 = plt.contour(xn, yn, znew_mp, [mptarg], colors='green')
                ax3.axes.set_xlabel('Rpos (mm)')
                ax3.axes.set_ylabel('Survival fraction')
                ax4 = plt.contour(xn, yn, znew_tp, [tptarg], colors='red')
                ax4.axes.set_xlabel('Rpos (mm)')
                ax4.axes.set_ylabel('Survival fraction')
                mpcontour = ax3.allsegs[0]  # Contour points in rpos x survival space that give this threshold
                tpcontour = ax4.allsegs[0]

                fig4, ax5 = plt.subplots()
                ax5 = plt.contour(rpos_grid_vals, surv_grid_vals[2:], mono_thr[2:, :], [mptarg], colors='green')
                ax5.axes.set_xlabel('Rpos (mm)')
                ax5.axes.set_ylabel('Survival fraction')
                ax6 = plt.contour(rpos_grid_vals, surv_grid_vals[2:], tripol_thr[2:, :], [tptarg], colors='red')
                ax6.axes.set_xlabel('Rpos (mm)')
                ax6.axes.set_ylabel('Survival fraction')
                ax5.axes.xaxis.set_label('Rpos (mm)')
                ax5.axes.yaxis.set_label('Survival fraction')
                mpcontour = ax5.allsegs[0]
                tpcontour = ax6.allsegs[0]
                # plt.show()

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

                # fig_altcontour, ax_alt = plt.subplots()
                # ax_alt = plt.plot(mpx, mpy, '.', color='green')
                # ax_alt = plt.plot(tpx, tpy, '.', color='red')

                x, y = intsec.intersection(mpx, mpy, tpx, tpy)  # find intersection(s)

                # How many intersections? 0, 1 or more?  If single intersection use those values
                nsols[i] = len(x)
                print("electrode: ", i)
                if nsols[i] == 0:
                    # no solution. This shouldn't happen with known scenarios,
                    # since the forward model calculated threshold, but it does happen in some cases.
                    # Maybe there's some error in this process.
                    # plt.show()
                    retval = find_closest(mpx, mpy, tpx, tpy)

                    rp_guess = mpx[retval]
                    sv_guess = mpy[retval]
                    ax_guess = plt.plot(rp_guess, sv_guess, 'x')
                    print("no solutions. Closest: ", retval, ' , leading to guesses of (position, survival): ',
                          rp_guess, sv_guess)

                elif nsols[i] == 1:  # unique solution
                    print("one solution")
                    rp_guess = x
                    sv_guess = y
                    ax_guess = plt.plot(rp_guess, sv_guess, 'x')

                else:  # multiple solutions
                    print(nsols[i], " solutions: ", x, ' and: ', y)
                    which_sols = np.zeros((4, int(nsols[i])))  # array for solutions and best fit
                    for sol in range(int(nsols[i])):  # Try all potential solutions; keep best
                        rp_guess = x[sol]
                        sv_guess = y[sol]

                        # Put variables into Parameters
                        # par.add('rpos_val', value=rp_guess, min=-0.8, max=0.8)
                        # par.add('surv_val', value=sv_guess, min=0.2, max=0.9)

                        # do fit, here with the default leastsq algorithm
                        # minner = Minimizer(objectivefunc_lmfit, par, fcn_args=(sigmaVals, simParams,
                        # fp, act_vals, thr_data, i))
                        # result = minner.minimize()
                        # which_sols[0, sol] = result.params['rpos_val']
                        # which_sols[1, sol] = result.params['surv_val']
                        # which_sols[2, sol] = np.mean(result.residual)
                        rposweight = 2
                        survweight = 0.0
                        if i > 1:  # calculate distance from previous coords
                            which_sols[3, sol] = np.sqrt(rposweight * (rp_guess - fitrposvals[i - 1]) ** 2 +
                                                         survweight * (sv_guess - fitsurvvals[i - 1]) ** 2)

                    # figure out which is best
                    print("which is best?")
                    # First attempt: pick rpos and surv that are closest to previous electrode (if there is one)
                    min_val = np.amin(which_sols[3, :])
                    closest_idx = np.where(which_sols[3, :] == min_val)[0]
                    rp_guess = x[closest_idx]
                    sv_guess = y[closest_idx]
                    print('Closest is # ', closest_idx, ' , and guesses are: ', rp_guess, sv_guess)
                    # print('And residual is: ', which_sols[2, closest_idx])

                    ax_guess = plt.plot(rp_guess, sv_guess, 'x')

                fitrposvals[i] = rp_guess
                fitsurvvals[i] = sv_guess

            # fix values for first and last electrodes as identical to their neighbors
            fitrposvals[0] = fitrposvals[1]
            fitrposvals[-1] = fitrposvals[-2]
            fitsurvvals[0] = fitsurvvals[1]
            fitsurvvals[-1] = fitsurvvals[-2]

            initvec = np.append(fitrposvals, fitsurvvals)

            for i, val in enumerate(initvec):  # place values in to the par object
                if i < 16:
                    lb = -0.85  # lower and upper bounds
                    ub = 0.85
                else:
                    lb = 0.1
                    ub = 0.9

                par.add('v_%i' % i, value=initvec[i], min=lb, max=ub)

            start_pos = fitsurvvals  # prior to fitting

            # end block for single electrode fits during the combined fit

        elif fit_mode == 'rpos':  # fit rpos only; hold survival fixed as the values loaded from the scenario
            initvec = np.append(np.zeros(NELEC), survvals)
            for i, val in enumerate(initvec):
                if i < 16:
                    lb = -0.85  # lower and upper bounds
                    ub = 0.85
                    par.add('v_%i' % i, value=initvec[i], min=lb, max=ub)
                else:
                    par.add('v_%i' % i, value=initvec[i], vary=False)

            #  start_pos = initvec[NELEC:]
        elif fit_mode == 'survival':
            initvec = np.append(rposvals, (np.ones(NELEC) * 0.5))
            for i, val in enumerate(initvec):
                if i < 16:
                    lb = rposvals[i] - 0.01  # lower and upper bounds
                    ub = rposvals[i] + 0.01
                else:
                    lb = 0.1
                    ub = 0.9
                par.add('v_%i' % i, value=initvec[i], min=lb, max=ub)

            start_pos = initvec[NELEC:]

        # Now do the main fitting for all electrodes at once
        minner = Minimizer(objectivefunc_lmfit_all, par, diff_step=0.02, nan_policy='omit',
                           fcn_args=(sigmaVals, simParams, fp, act_vals, thr_data))
        if use_fwd_model:
            result = minner.minimize(method='least_squares', diff_step=0.02)
        else:  # use CT data
            result = minner.minimize(method='least_squares', ftol=1e-1, xtol=1e-9, max_nfev=2000)

        for i in range(NELEC):
            vname = 'v_%i' % i
            fitrposvals[i] = result.params[vname]

        simParams['electrodes']['rpos'] = fitrposvals

        for i in range(NELEC):
            vname = 'v_%i' % (i + NELEC)
            fitsurvvals[i] = result.params[vname]

        # report last fit
        report_fit(result)
        result.params.pretty_print()
        simParams['electrodes']['rpos'] = fitrposvals
        simParams['neurons']['nsurvival'] = survFull.survFull(simParams['electrodes']['zpos'], fitsurvvals,
                                                              simParams['grid']['z'])
        simParams['channel']['sigma'] = sigmaVals[0]
        thrsimmp = gT.getThresholds(act_vals, fp, simParams)
        simParams['channel']['sigma'] = sigmaVals[1]
        thrsimtp = gT.getThresholds(act_vals, fp, simParams)

        errvals = [np.subtract(thrsimmp[0], thr_data['thrmp_db']), np.subtract(thrsimtp[0][1:NELEC - 1],
                                                                               thr_data['thrtp_db'][1:NELEC - 1])]
        thrsim = [[thrsimmp[0]], [thrsimtp[0]]]
        thrtargs = [[thr_data['thrmp_db']], [thr_data['thrtp_db']]]

        if use_fwd_model:
            [survvals, rposvals] = s_scen.set_scenario(scenario, NELEC)
            rposerrs = np.subtract(rposvals, fitrposvals)
            survivalerrs = np.subtract(survvals, fitsurvvals)
            # Save values in CSV format
            save_file_name = INVOUTPUTDIR + scenario + '_fitResults_' + 'combined.csv'
        else:
            ct_vals = subject_data.subj_ct_data(SUBJECT)

            rposerrs = np.subtract(rposvals, ct_vals)
            survivalerrs = np.empty(NELEC)
            ##
            # Save values in CSV format
            save_file_name = INVOUTPUTDIR + SUBJECT + '_fitResults_' + 'combined.csv'
            # [survvals, rposvals] = s_scen.set_scenario(scenario, NELEC)

        # Save the data
        with open(save_file_name, mode='w') as data_file:
            data_writer = csv.writer(data_file, delimiter=',', quotechar='"')
            header = ['Row', 'Rposition', 'Survival', 'ThreshMP', 'ThreshTP',
                      'RpositionFit', 'SurvivalFit', 'RposError', 'SurvError']
            data_writer.writerow(header)
            for row in range(0, NELEC):
                t1 = row
                t2 = rposvals[row]
                t3 = survvals[row]
                t4a = thrsim[0][0]
                t4 = t4a[row]
                t5a = thrsim[1][0]
                t5 = t5a[row]
                # t6 = opt_result.x[row]
                # t7 = opt_result.x[NELEC + row]
                t6 = fitrposvals[row]
                t7 = fitsurvvals[row]
                t8 = rposerrs[row]
                t9 = survivalerrs[row]
                data_writer.writerow([t1, t2, t3, t4, t5, t6, t7, t8, t9])
        data_file.close()

        # Save values in npy format
        save_file_name = INVOUTPUTDIR + scenario + '_fitResults_' + 'combined'
        np.save(save_file_name, [sigmaVals, rposvals, survvals, thrsim, thrtargs, initvec, [fitrposvals, fitsurvvals],
                                 rposerrs, survivalerrs])

        # Make plots
        if ifPlot:
            PlotInverseResults.plot_inverse_results()


if __name__ == '__main__':
    inverse_model_combined_se()
