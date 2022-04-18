# THRFUNCTION.py
# function [ncount,NeuralProfile] = cimodel_thrfunction(currentvec,AFieldUS,simParams)
# Function for least squares minimization (combined with a target, like 100 neurons)
# to obtain a final threshold current level from the CI Cylinder Model. Currently called by
# MODELFITTHRESHOLD_CYLINDER.M. The argument 'currentvec' is in (linear) uA and should be
# a scalar if the function is to be used as a least-squares iteration function.
# 'currentvec','chnum', and 'sigma' set/override values in the 'simParams.channel' substructure.
# Input matrix 'ActiveUS' should be an unscaled version of the activating field. It is of
# size # of z-axis points x # of stimuli and can be generated by a prior call to the function
# CIMODEL_CREATEACTIVEPROFILE. There should be one entry of 'currentvec' for every
# column of 'ActiveUS'. 'simParams' is only utilized for its '.neuron' substructure, for
# a call to CIMODEL_CREATENEURALPROFILE.
# Note that 'ncount' will only be meaningful for least-squares fitting for scalar values of
# 'currentvec' (i.e. one stimulus). However, 'NeuralProfile' WILL be a valid matrix if
# the length of current vec matches the number of columns of 'ActiveuS' and CIMODEL_THRFUNCTION
# is called directly (i.e not as a least-squares function argument).

import numpy as np
import create_neural_profile as cnp


def thrFunction(current, active_us, sim_params):  # simplified for scalar current

    active_profile = current * active_us  # Scale by current

    # Now call nonlinear sigmoidal activation function
    neural_profile = cnp.create_neural_profile(active_profile, sim_params)
    ncount = np.nansum(neural_profile)

    return [ncount, neural_profile, active_profile]
