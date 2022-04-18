#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  8 13:46:54 2020

@author: perkel
"""

# implant voltage calculations from Goldwyn, denovo.

import cProfile
import io
import pickle
import pstats
from datetime import datetime
from pstats import SortKey

import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as integ
import scipy.special as spec


def goldwyn_beta(eps, k, rs, re, n):
    krs = k * rs
    denom = ((eps * spec.kvp(n, krs) * spec.iv(n, krs)) - (spec.kn(n, krs) * spec.ivp(n, krs)))
    return spec.iv(n, k * re) / (denom * krs)


def goldwyn_phi(eps, k, rs, re, reval, n):
    phi = (goldwyn_beta(eps, k, rs, re, n) * spec.kn(n, k * reval))
    return phi


def integ_func(x, m_max, pratio, radius, reval, z, theta, relec):  # This is the Bessel function along z axis
    sum_contents = 0.0
    increments = np.zeros(m_max)
    rel_incrs = np.zeros(m_max)
    for i in range(m_max):
        if i == 0:
            gamma = 1.0
        else:
            gamma = 2.0

        increments[i] = gamma * np.cos(i * theta) * goldwyn_phi(pratio, x, radius, relec, reval, i)
        sum_contents += increments[i]
        rel_incrs[i] = np.abs(increments[i]) / sum_contents

    return np.cos(x * z) * sum_contents


# See Rubinstein dissertation, 1988, Ch 6.

pr = cProfile.Profile()

# Field parameters. zEval can be higher (more precise, slower) or lower resolution (less precise, faster)
fp = {'model': 'cylinder_3d', 'fieldtype': 'Voltage_Activation', 'evaltype': 'SG', 'cylRadius': 1.0,
      'relec': np.arange(-0.95, 0.951, 0.05), 'resInt': 70.0, 'resExt': 250.0, 'rspace': 1.1, 'theta': 0.0,
      # 'zEval': (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3,
      #              1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9,
      #              3.0, 3.2, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4.0, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7,
      #              4.8, 4.9, 5.0, 5.2, 5.4, 5.6, 5.8, 6.0, 6.2, 6.4, 6.6, 6.8, 7.0, 7.2, 7.4, 7.6, 7.8,
      #              8.0, 8.2, 8.4, 8.6, 8.8, 9.0, 9.2, 9.4, 9.6, 9.8, 10.0, 10.5, 11.0, 11.5, 12.0, 12.5,
      #              13.0, 13.5, 14.0, 14.5, 15.0, 15.5, 16.0, 16.5, 17.0, 17.5, 18.0, 18.5, 19.0, 19.5, 20.0,
      #              22.0, 24.0, 26.0, 28.0, 30.0, 32.0, 34.0, 36.0, 38.0, 40.0),
      'zEval': (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2,
                1.4, 1.6, 1.8, 2.0, 2.4, 2.8, 3.2, 3.6, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0,
                8.5, 9.0, 9.5, 10.0, 12.0, 14.0, 16.0, 20.0, 30.0, 40.0),
      'mMax': 47, 'intStart': 1e-12, 'intEnd': 600.0, 'reval': 1.1,
      'ITOL': 1e-6, 'runDate': 'rundate', 'runOutFile': 'savefile'}

now = datetime.now()
date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
fp['runDate'] = date_time
fp['runOutFile'] = '20Jan2022MedResolution_RInt70.dat'

# Derived values
resRatio = fp['resInt'] / fp['resExt']
rPrime = fp['reval']
nZ = len(fp['zEval'])  # this may be a problem if zEval is just a scalar
nYpos = 7  # # of values to calculate across the y dimension

rElecRange = fp['relec']
nRElec = len(rElecRange)
voltageVals = np.empty((nRElec, nYpos, nZ))
activationVals = np.empty((nRElec, nZ))
if_plot = False

pr.enable()  # Start the profiler

# loop on electrode radial positions
for i, rElec in enumerate(rElecRange):
    # retval = integ_func(1e-12, mMax,resRatio,cylRadius,rPrime,zEval,thisTheta, rElec)
    print('starting electrode position ', i, ' of ', len(rElecRange))
    # print("vals: ", retval)
    # print("old vals: ", bessel_func(1e-12, 'V2e', mMax,resRatio,cylRadius,rPrime,zEval,thisTheta, rElec))

    # Loop on Z position; but in this streamlined version, keep reval and theta constant
    for k, thisZ in enumerate(fp['zEval']):
        print('# ', k, ' of ', nZ, ' z values.')
        # loop on y positions to get 2nd spatial derivative
        nYEval = nYpos // 2  # floor division
        yInc = 0.01  # 10 microns
        yVals = np.arange(-nYEval * yInc, nYEval * (yInc * 1.01), yInc)
        transVoltage = np.empty(nYpos)
        for j, yVal in enumerate(yVals[0:nYEval + 1]):
            thisTheta = np.arctan(yVal / fp['reval'])
            rPrime = np.sqrt((yVal ** 2) + (fp['reval'] ** 2))
            [itemp, error] = integ.quad(integ_func, fp['intStart'], fp['intEnd'], epsabs=fp['ITOL'], limit=1000,
                                        args=(fp['mMax'], resRatio, fp['cylRadius'], rPrime, thisZ, thisTheta, rElec))
            tempV = itemp / (2 * (np.pi ** 2))
            voltageVals[i, j, k] = tempV
            voltageVals[i, nYpos - (j + 1), k] = tempV
            transVoltage[j] = tempV
            transVoltage[nYpos - (j + 1)] = tempV

        # Calculate the second spatial derivative in the y dimension
        transVPrime = np.diff(np.diff(transVoltage)) / (yInc ** 2)
        activationVals[i, k] = (transVPrime[nYEval - 1])  # Value in center
        if np.isnan(activationVals[i, k]):
            print('nan value for i == ', i, " and k == ", k)

pr.disable()  # stop the profiler

if if_plot:
    # Some plots that might be helpful
    # plt.plot(rElecRange, voltageVals[:, 10], 'or')
    # plt.plot(rElecRange, activationVals[:, 0], 'ob')
    # plt.xlabel('Electrode radial position (mm)')
    # plt.ylabel('Red: voltage; Blue: activation')
    # plt.xlim(-1.0, 1.0)
    # plt.yscale('log')

    plt.figure()
    plt.xlabel('y position (mm)')
    plt.ylabel('Voltage')
    plt.plot(yVals, voltageVals[0, :, 0], '.b')
    plt.plot(yVals, voltageVals[1, :, 0], '.g')
    # plt.plot(yVals, voltageVals[2, :, 0], '.r')
    # plt.plot(yVals, voltageVals[3, :, 0], '.k')

s = io.StringIO()
sortby = SortKey.CUMULATIVE
ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
ps.print_stats(10)
print(s.getvalue())

# Save the data into a single file
with open(fp['runOutFile'], 'wb') as combined_data:
    pickle.dump([fp, voltageVals, activationVals], combined_data)
combined_data.close()
