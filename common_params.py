# Common parameters for the implant model

import numpy as np

# Basic parameters
NELEC = 16
ELEC_BASALPOS = 30
ESPACE = 1.1  # in mm; 'ELECTRODE' parameters must be vectors

# Neural activation parameters
THRTARG = 1000.0  # number of active neurons = threshold
TARG_TEXT = '_TARG1000/'
ACTR = 500.0
ACTR_TEXT = 'ACTR500_'
ACT_STDREL = 0.75
STD_TEXT = 'STDR0_75'

# File locations
RES1 = 70.0  # internal resistivity
RES2 = 250.0  # external resistivity

sigmaVals = [0, 0.9]  # Always explore monopolar stimulation and one value of sigma for triploar

COCHLEA = {'source': 'manual', 'res1': RES1*np.ones(NELEC), 'res2': RES2*np.ones(NELEC),
           'timestamp': [], 'radius': []}
ELECTRODES = {'source': 'manual', 'timestamp': [], 'zpos': ELEC_BASALPOS - np.arange(NELEC - 1, -1, -1) * ESPACE,
              'rpos': []}
NEURONS = {'act_ctr': ACTR, 'act_stdrel': ACT_STDREL, 'nsurvival': [], 'sidelobe': 1.0, 'rlvl': [],
           'rule': 'proportional', 'coef': 0.0, 'power': 1.0, 'thrTarg': THRTARG}
# For COEF convex: <0 | 0.4, 0.9  linear: 0 | 1; concave: >0 | 1.0, 1.8
CHANNEL = {'source': 'manual', 'number': range(0, NELEC), 'config': 'pTP', 'sigma': 0.9, 'alpha': 0.5,
           'current': 10000000000.0}
GRID = {'r': 0.1, 'th': 0.0, 'z': np.arange(0, 33, 0.1)}
simParams = {'cochlea': COCHLEA, 'electrodes': ELECTRODES, 'channel': CHANNEL, 'grid': GRID, 'neurons': NEURONS}

nZ = len(GRID['z'])
NSURVINIT = 1.0

# Set specific scenarios to study. Not used by the 2D exploration tool
# scenarios = ['Gradual80R75']
# scenarios = ['Uniform80R05']
# scenarios = ['Ramp80Rvariable1']
scenarios = ['RampRpos_revSGradual80']
# scenarios = ['Rpos-03S0_4']
#  scenarios = ['Ramp80Rvariable1']

# File locations
FWDOUTPUTDIR = 'FWD_OUTPUT/' + ACTR_TEXT + STD_TEXT + TARG_TEXT
INVOUTPUTDIR = 'INV_OUTPUT/'

FIELDTABLE = '4Jan2022HighResolution_RInt70_nonans.dat'
# FIELDTABLE = '19Aug2021HighResolution.dat'
# FIELDTABLE = '1Oct2021MediumResolution_Rint100.dat'
