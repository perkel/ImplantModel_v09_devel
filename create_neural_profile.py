# CIMODEL_CREATENEURALPROFILE.M
# function [neural_out,nsurvival] = cimodel_createneuralprofile(active_profile,simParams)
# Create one or more neural profiles, i.e. the final neural "contribution"
# across an array of neural clusters, based on unscaled activation profiles. Called by
# CIMODEL_THRFUNCTION, but can also be evoked separately.
# 'active_profile' is of size nZ (#z-axis points) x nStim (#of stimuli to simulate).
# It should be already scaled by the applied current, but not transformed by the sidelobe
# ratio or to an absolute value. The activation sensitivity value, contained
# in 'simParams.neurons.act_center' is a scalar and assumed to hold for all of the profiles.
# Note that ONLY the '.neurons' substructure of 'simParams' is used in this function. Sigma,
# current level, channel, etc, are already incorporated in 'active_profile'.

import numpy as np
from scipy import special


def sigmoid_func(actr, stdrel, x):
    std = actr * stdrel
    retval = 0.5 * (1 + special.erf((x - actr) / (np.sqrt(2) * std)))
    return retval


def new_sigmoid(actr, stdrel, x):  # scaled to subtract the y-intercept and still asyptote at 1.0
    y_int = sigmoid_func(actr, stdrel, 0)
    scale_factor = 1.0 - y_int
    yval = (sigmoid_func(actr, stdrel, x) - y_int) / scale_factor
    return yval


def create_neural_profile(active_profile, simParams):
    nZ = active_profile.shape

    # Extract local values from simParams structure
    act_ctr = simParams['neurons']['act_ctr']
    act_std = act_ctr * simParams['neurons']['act_stdrel']
    nsurvival = simParams['neurons']['nsurvival']
    rlvltable = simParams['neurons']['rlvl']

    # Create neural excitation profile for each input activation profile #
    neural_out = np.empty(nZ)
    neural_out[:] = np.nan

    # scale sidelobes before computing absolute value
    active_profile[active_profile < 0] = simParams['neurons']['sidelobe'] * active_profile[active_profile < 0]

    if any(np.isreal(active_profile == False)):

        # treat non-viable neurons depending on the desired algorithm
        if simParams['neurons']['rule'] == 'proportional':
            try:

                #  The lines commented out here are the original sigmoid activation function
                #  The problem is that for input of exactly zero, the output is positive, which can cause
                #  major problems in some cases.
                #  tempval1 = active_profile[:]-act_ctr
                #  tempval2 = tempval1/(np.sqrt(2)*act_std)
                #  atemp = 0.5 * (1 + sp.special.erf((tempval2 - 5)))
                atemp = new_sigmoid(act_ctr, act_std, active_profile[:])
            except BaseException:
                print('create_neural_profile: failed to calculate atemp using error function')

            ntemp = np.interp(atemp, rlvltable[0], rlvltable[1])
            if len(ntemp) == len(nsurvival):
                neural_out[:] = np.multiply(ntemp, nsurvival)
            else:
                neural_out[:] = np.multiply(ntemp, [np.nan, nsurvival, np.nan])  # if nsurvival is only 14 elements long

        else:
            raise 'Neural excitation model not recognized.'

    neural_out = neural_out  # used to round to the nearest integer
    return neural_out
