# set_scenario returns a vector of survival values and a vector of radial positions
import numpy as np


def set_scenario(this_scen, n_elec):
    surv_vals = np.zeros(n_elec)
    rpos_vals = np.zeros(n_elec)
    # If uniform, then calculate values, otherwise use switch and specify manually
    if this_scen[0:7] == 'Uniform':
        # parse scenario name
        surv_vals = np.ones(n_elec) * int(this_scen[7:9]) * 0.01
        # Find 'R'
        the_idx = this_scen.find('R')
        rpos_vals = np.zeros(n_elec) + int(this_scen[the_idx + 1:]) * 0.01
    else:
        if this_scen == 'Gradual80R00':
            surv_vals = [0.8, 0.8, 0.6, 0.4, 0.4, 0.6, 0.8, 0.8, 0.8, 0.6, 0.4, 0.4, 0.6, 0.8, 0.8, 0.8]
            rpos_vals = np.zeros(n_elec)
        elif this_scen == 'Gradual80R75':
            surv_vals = [0.8, 0.8, 0.6, 0.4, 0.4, 0.6, 0.8, 0.8, 0.8, 0.6, 0.4, 0.4, 0.6, 0.8, 0.8, 0.8]
            rpos_vals = np.zeros(n_elec) + 0.75
        elif this_scen == 'Gradual50R00':
            surv_vals = [0.5, 0.5, 0.35, 0.2, 0.2, 0.35, 0.5, 0.5, 0.5, 0.35, 0.2, 0.2, 0.35, 0.5, 0.5, 0.5]
            rpos_vals = np.zeros(n_elec)
        elif this_scen == 'Sudden80R-05':
            surv_vals = [0.8, 0.8, 0.8, 0.4, 0.4, 0.8, 0.8, 0.8, 0.8, 0.8, 0.4, 0.4, 0.8, 0.8, 0.8, 0.8]
            rpos_vals = np.zeros(n_elec) - 0.5
        elif this_scen == 'Sudden80R00':
            surv_vals = [0.8, 0.8, 0.8, 0.4, 0.4, 0.8, 0.8, 0.8, 0.8, 0.8, 0.4, 0.4, 0.8, 0.8, 0.8, 0.8]
            rpos_vals = np.zeros(n_elec)
        elif this_scen == 'Sudden80R05':
            surv_vals = [0.8, 0.8, 0.8, 0.4, 0.4, 0.8, 0.8, 0.8, 0.8, 0.8, 0.4, 0.4, 0.8, 0.8, 0.8, 0.8]
            rpos_vals = np.zeros(n_elec) + 0.5
        elif this_scen == 'Uniform80R05':
            surv_vals = 0.8 * np.ones(n_elec)
            rpos_vals = np.zeros(1, n_elec) + 0.5
        elif this_scen == 'Ramp80Rvariable1':
            surv_vals = 0.8 * np.ones(n_elec)
            rpos_vals = np.arange(-0.8, 0.75, 0.1)
        elif this_scen == 'Ramp80Rvariable2':
            surv_vals = 0.8 * np.ones(n_elec)
            rpos_vals = np.arange(-0.6, 0.95, 0.1)
        elif this_scen == 'RampSurvR00':
            surv_vals = np.arange(0.1, 0.86, 0.05)
            rpos_vals = np.zeros(n_elec)
        elif this_scen == 'RampSurvR03':
            surv_vals = np.arange(0.2, 0.65, 0.03)
            rpos_vals = np.zeros(n_elec) + 0.3
        elif this_scen == 'RampSurvR07':
            surv_vals = np.arange(0.1, 0.86, 0.05)
            rpos_vals = np.zeros(n_elec) + 0.7
        elif this_scen == 'RampRposS05':
            surv_vals = 0.5 * np.ones(n_elec)
            rpos_vals = np.arange(-0.6, 0.95, 0.1)
        elif this_scen == 'RampRposSGradual80':
            surv_vals = [0.8, 0.8, 0.6, 0.4, 0.4, 0.6, 0.8, 0.8, 0.8, 0.6, 0.4, 0.4, 0.6, 0.8, 0.8, 0.8]
            rpos_vals = np.arange(0.8, -0.8, -0.1)
        elif this_scen == 'RampRpos_revSGradual80':
            surv_vals = [0.8, 0.8, 0.6, 0.4, 0.4, 0.6, 0.8, 0.8, 0.8, 0.6, 0.4, 0.4, 0.6, 0.8, 0.8, 0.8]
            rpos_vals = np.arange(-0.8, 0.8, 0.1)
        elif this_scen == 'Rpos-03S0_4':
            surv_vals = np.zeros(n_elec) + 0.4
            rpos_vals = np.zeros(n_elec) - 0.3
        elif this_scen == 'TestLookingForAnomalies':
            surv_vals = [0.35, 0.35, 0.35, 0.36, 0.36, 0.36, 0.375, 0.375, 0.375, 0.38, 0.38, 0.38, 0.39, 0.39, 0.4,
                         0.41]
            rpos_vals = [0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125,
                         0.125, 0.125, 0.125]

    return [surv_vals, rpos_vals]
