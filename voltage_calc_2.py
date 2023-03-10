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
import mpmath as mp


def goldwyn_beta(eps, k2, rs, re, n):
    krs = k2 * rs
    # print('in gw_beta: eps, k2, rs, re, n, krs: ', eps, k2, rs, re, n, krs)
    denom = ((eps * mp.besselk(n, krs, derivative=1) * mp.besseli(n, krs)) - (mp.besselk(n, krs) *
                                                                              mp.besseli(n, krs, derivative=1)))
    # print('and denom is: ', denom)
    return mp.besseli(n, k2 * re) / (denom * krs)


def goldwyn_phi(eps, k, rs, re, reval, n):
    phi = (goldwyn_beta(eps, k, rs, re, n) * mp.besselk(n, k * reval))
    return phi


def integ_func(x, m_max, pratio, rad, reval, z, theta, relec):  # This is the Bessel function along z axis
    sum_contents = 0.0
    increments = np.zeros(m_max)
    rel_incrs = np.zeros(m_max)
    for idx in range(m_max):
        if idx == 0:
            gamma = 1.0
        else:
            gamma = 2.0

        increments[idx] = gamma * np.cos(idx * theta) * goldwyn_phi(pratio, x, rad, relec, reval, idx)
        sum_contents += increments[idx]
        rel_incrs[idx] = np.abs(increments[idx]) / sum_contents

    return np.cos(float(x * z)) * sum_contents


# See Jay Rubinstein dissertation, 1988, Ch 6.

# Main parameters to vary
radius = 1.0  # cylinder radius
res_int = 70.0  # internal resistivity
res_ext = 250.0  # external resistivity
output_filename = '26Aug2022_medres_using_mpmathintegrator.dat'
# changes for streamlining: only 3 y values; MMax 12; intEnd 200; itol 1e-4. Run time dropped from hours (overnight)
# to ~ 800 s. Need to do quality control
# Not streamlined with 3 ypos, Mmax = 47, intEnd = 600, itol = 1e-6 runtime was 5100 seconds.
# streamlined 1: ypos=3; mMax=20; intEnd 600, itol=1e-6.  runtime: 1812 sec
# streamlined 2: ypos=3; mMax=47; intEnd 200, itol:1e-6. runtime 4022 sec
# streamlined 3: ypos-3; mMax=47; intEnd 600; itol=1e-4. runtime 3151 sec

# second round. Extend spatial dimension to higher resolution
# zE2_full: ypos=3; mMax = 47; intEnd = 600; itol=1e-6; runtime 7080 sec
# 20Aug22 hires; mMax = 20, intEnd = 600, itol=1e-6
# 26 Aug 2022 med res: mMax = 12; intEnd = 400; itol=1e-4

pr = cProfile.Profile()

# Field parameters. zEval can be higher (more precise, slower) or lower resolution (less precise, faster)
fp = {'model': 'cylinder_3d', 'fieldtype': 'Voltage_Activation', 'evaltype': 'SG', 'cylRadius': radius,
      'relec': np.arange(-0.95, 0.951, 0.05), 'resInt': res_int, 'resExt': res_ext, 'rspace': 1.1, 'theta': 0.0,
      # 'zEval': (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3,
      #              1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9,
      #              3.0, 3.2, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4.0, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7,
      #              4.8, 4.9, 5.0, 5.2, 5.4, 5.6, 5.8, 6.0, 6.2, 6.4, 6.6, 6.8, 7.0, 7.2, 7.4, 7.6, 7.8,
      #              8.0, 8.2, 8.4, 8.6, 8.8, 9.0, 9.2, 9.4, 9.6, 9.8, 10.0, 10.5, 11.0, 11.5, 12.0, 12.5,
      #              13.0, 13.5, 14.0, 14.5, 15.0, 15.5, 16.0, 16.5, 17.0, 17.5, 18.0, 18.5, 19.0, 19.5, 20.0,
      #              22.0, 24.0, 26.0, 28.0, 30.0, 32.0, 34.0, 36.0, 38.0, 40.0),
      'zEval': (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2,
                1.4, 1.6, 1.8, 2.0, 2.4, 2.8, 3.2, 3.6, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0,
                8.5, 9.0, 9.5, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0, 30.0,
                32.0, 34.0, 36.0, 38.0, 40.0),
      # 'zEval': (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2,
      #           1.4, 1.6, 1.8, 2.0, 2.4, 2.8, 3.2, 3.6, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0,
      #           8.5, 9.0, 9.5, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 25.0, 30.0, 40.0),
      'mMax': 15, 'intStart': 1e-12, 'intEnd': 600.0, 'reval': 1.1,
      'ITOL': 1e-4, 'runDate': 'rundate', 'runOutFile': 'savefile'}

now = datetime.now()
date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
fp['runDate'] = date_time
fp['runOutFile'] = output_filename

# Derived values
resRatio = fp['resInt'] / fp['resExt']
rPrime = fp['reval']
nZ = len(fp['zEval'])  # this may be a problem if zEval is just a scalar
n_ypos = 3  # # of values to calculate across the y dimension

rElecRange = fp['relec']
nRElec = len(rElecRange)
voltageVals = np. zeros((nRElec, n_ypos, nZ))
activationVals = np.zeros((nRElec, nZ))
if_plot = False

pr.enable()  # Start the profiler

n_yeval = n_ypos // 2  # floor division
y_inc = 0.01  # 10 microns
yVals = np.arange(-n_yeval * y_inc, n_yeval * (y_inc * 1.01), y_inc)
transVoltage = np.empty(n_ypos)

# loop on electrode radial positions
for i, rElec in enumerate(rElecRange):
    # retval = integ_func(1e-12, mMax,resRatio,cylRadius,rPrime,zEval,thisTheta, rElec)
    print('starting electrode position ', i, ' of ', len(rElecRange))

    # Loop on Z position; but in this streamlined version, keep reval and theta constant
    for m, thisZ in enumerate(fp['zEval']):
        print('# ', m, ' of ', nZ, ' z values.')
        # loop on y positions to get 2nd spatial derivative
        for j, yVal in enumerate(yVals[0:n_yeval + 1]):
            thisTheta = np.arctan(yVal / fp['reval'])
            rPrime = np.sqrt((yVal ** 2) + (fp['reval'] ** 2))
            # [itemp, error] = integ.quad(integ_func, fp['intStart'], fp['intEnd'], epsabs=fp['ITOL'], limit=1000,
            #                             args=(fp['mMax'], resRatio, fp['cylRadius'], rPrime, thisZ, thisTheta, rElec))
            itemp = mp.quad(lambda x: integ_func(x, fp['mMax'], resRatio, fp['cylRadius'], rPrime, thisZ,
                                                          thisTheta, rElec), [fp['intStart'], fp['intEnd']], verbose=1)

            tempV = itemp / (2 * (np.pi ** 2))
            voltageVals[i, j, m] = tempV
            voltageVals[i, n_ypos - (j + 1), m] = tempV
            transVoltage[j] = tempV
            transVoltage[n_ypos - (j + 1)] = tempV

        # Calculate the second spatial derivative in the y dimension
        transVPrime = np.diff(np.diff(transVoltage)) / (y_inc ** 2)
        activationVals[i, m] = (transVPrime[n_yeval - 1])  # Value in center
        if np.isnan(activationVals[i, m]):
            print('nan value for i == ', i, " and m == ", m)

pr.disable()  # stop the profiler

s = io.StringIO()
sortby = SortKey.CUMULATIVE
ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
ps.print_stats(10)
print(s.getvalue())

# Save the data into a single file
with open(fp['runOutFile'], 'wb') as combined_data:
    pickle.dump([fp, voltageVals, activationVals], combined_data)
combined_data.close()

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
    plt.show()
