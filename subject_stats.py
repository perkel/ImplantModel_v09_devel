# subject_stats

# calculate statistics on subject CT data
# David Perkel 1 Dec 2022

import numpy as np
import subject_data
from common_params import *

scenarios = ['S22', 'S27', 'S29', 'S38', 'S40', 'S41', 'S42', 'S43', 'S46', 'S47', 'S49R', 'S50', 'S53', 'S54', 'S55',
             'S56', 'S57']

nelec = 16
ndiffs = nelec - 1
n_scen = len(scenarios)
grand_diffs = np.zeros([n_scen, ndiffs])
diffs = np.zeros(n_scen)
means = np.zeros(n_scen)
std_devs = np.zeros(n_scen)

fit_grand_diffs = np.zeros([n_scen, ndiffs])
fit_diffs = np.zeros(n_scen)
fit_means = np.zeros(n_scen)
fit_std_devs = np.zeros(n_scen)
print('Mean and maximum absolute differences in radial position between adjacent electrodes')

for i, scen in enumerate(scenarios):
    ct_data = subject_data.subj_ct_data(scen)
    thr_data = subject_data.subj_thr_data(scen)
    max_mp_thr = np.max(thr_data[0])
    min_mp_thr = np.min(thr_data[0])
    max_tp_thr = np.max(thr_data[1])
    min_tp_thr = np.min(thr_data[1])
    thr_diff = thr_data[1] - thr_data[0][1:-1]
    min_thr_diff = np.min(thr_diff)
    max_thr_diff = np.max(thr_diff)

    abs_diffs = np.abs(np.diff(ct_data))
    grand_diffs[i, :] = abs_diffs
    max_abs_diff = np.max(abs_diffs)
    mean_abs_diffs = np.mean(abs_diffs)
    std_dev_abs_diffs = np.std(abs_diffs)
    diffs[i] = max_abs_diff
    means[i] = mean_abs_diffs
    std_devs[i] = std_dev_abs_diffs
    print('subject: ', scen, ' max: ', '%.3f' % max_abs_diff, '; mean: ', '%.3f' % mean_abs_diffs,
          ' stdev: ', '%.3f' % std_dev_abs_diffs)
    print('subject: ', scen, ' max_mp_thr: ', '%.3f' % max_mp_thr, ' min_mp_thr: ', '%.3f' % min_mp_thr,
          ' max_tp_thr: ', '%.3f' % max_tp_thr, ' min_tp_thr: ', '%.3f' % min_tp_thr, ' min_diff: ',
          '%.3f' % min_thr_diff, ' max_diff: ', '%.3f' % max_thr_diff)

    # find any outliers (> 2 std from mean)
    outliers = np.argwhere((np.abs(abs_diffs) - mean_abs_diffs) > (2 * std_dev_abs_diffs))
    print('outliers are gaps #s: ', outliers)

    # Read in fit data
    # data_filename = INVOUTPUTDIR + scen + '_fitResults_' + 'combined.npy'
    # [sigmaVals, rposvals, survvals, thrsim, thrtargs, initvec, [fitrposvals, fitsurvvals],
    #  rposerrs, rpos_err_metric, survivalerrs, ct_vals] = np.load(data_filename, allow_pickle=True)
    # fit_abs_diffs = np.abs(np.diff(fitrposvals))
    # fit_grand_diffs[i, :] = fit_abs_diffs
    # fit_max_abs_diff = np.max(fit_abs_diffs)
    # fit_mean_abs_diffs = np.mean(fit_abs_diffs)
    # fit_std_dev_abs_diffs = np.std(fit_abs_diffs)
    # fit_diffs[i] = fit_max_abs_diff
    # fit_means[i] = fit_mean_abs_diffs
    # fit_std_devs[i] = fit_std_dev_abs_diffs
    # print('subject: ', scen, ' fit_max: ', '%.3f' % fit_max_abs_diff, '; fit_mean: ', '%.3f' % fit_mean_abs_diffs,
    #       ' fit_stdev: ', '%.3f' % fit_std_dev_abs_diffs)
    #
    # # find any outliers (> 2 std from mean)
    # outliers = np.argwhere((np.abs(abs_diffs) - mean_abs_diffs) > (2 * std_dev_abs_diffs))
    # print('outliers are gaps #s: ', outliers)





grand_mean = np.mean(grand_diffs.flatten())
grand_std = np.std(grand_diffs.flatten())
print('grand mean: ', '%.3f' % grand_mean, ' stdev: ', '%.3f' % grand_std)
