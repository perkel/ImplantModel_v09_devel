# calc_ecap.py -- calculates a proxy for electrical compound action potential
# David J. Perkel 26 November 2022

import numpy as np
from common_params import *
import getThresholds as gT
import cylinder3d_makeprofile as c3dm
import thrFunction

n_stim = 100  # stim levels in the ECAP

def db_to_cur(val):
    return np.power(10, val/20.0)

def cur_to_db(val):
    return np.log10(val)*20.0

def calc_ecap(act_vals, field_params, sim_params, thresh, thr_sim_db):

    # Calculate ecap from threshold up to activation of half of all neurons. We have threshold.
    # Next get stimulus intensity to activate half of all neurons
    nz = len(sim_params['grid']['z'])
    if isinstance(sim_params['electrodes']['rpos'], list):
        nelec = len(sim_params['electrodes']['rpos'])
    elif not isinstance(sim_params['electrodes']['rpos'], np.ndarray):  # special case for 2D version of fwd model
        nelec = 1
    else:
        nelec = len(sim_params['electrodes']['rpos'])
    half_act_thresh = np.zeros(nelec)  # Array to hold half_activation thresholds


# low end should be 5dB below threshold
    # high end should be 1000 neurons (given by calling function)
    # Normal dynamic range is 20 dB
    orig_targ = sim_params['neurons']['thrTarg']  # save old value
    sim_params['neurons']['thrTarg'] = thresh
    high_thresh, nact = gT.getThresholds(act_vals, field_params, sim_params)
    sim_params['neurons']['thrTarg'] = orig_targ  # restore old value

    ecap = np.zeros([nelec, n_stim, 2])

    # Loop on electrodes
    for k in range(nelec):
        sim_params['channel']['number'] = k
        aprofile = c3dm.cylinder3d_makeprofile(act_vals, field_params, sim_params)
        e_field = abs(aprofile)  # uV^2/mm^2
        currents = np.linspace(db_to_cur(thr_sim_db[k]), db_to_cur(high_thresh[k]), num=n_stim)
        # Now we have threshold and activation profile. Next calculate ECAP by calling thrFunction
        # with a range of current arguments from the treshold to a maximum value
        a, b = thrFunction.thrFunction(db_to_cur(22), e_field, sim_params)[0:2]
        for i, cur in enumerate(currents):
            ecap[k, i, 0] = cur_to_db(cur)
            ecap[k, i, 1] = thrFunction.thrFunction(cur, e_field, sim_params)[0]

    return ecap
