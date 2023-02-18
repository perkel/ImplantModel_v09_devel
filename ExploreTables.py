#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 25 20:04:55 2020

@author: perkel
"""

import pickle
import matplotlib.pyplot as plt
import numpy as np

FILENAME = '8Sept2022_MedResolution_Rext250_nonans.dat'
with open(FILENAME, 'rb') as combined_data:
    data = pickle.load(combined_data)
combined_data.close()

fp = data[0]
v_vals = data[1]
act_vals = data[2]

fig, ax = plt.subplots()
zpos = [0.0, 0.2, 0.4, 0.6]
for i, val in enumerate(zpos):  # a few plots for different z positions
    temparray = np.array(fp['zEval'])
    zpos_idx = np.argmin(np.abs(temparray - val))
    yvals = act_vals[:, zpos_idx]
    ax.plot(1 - fp['relec'], yvals, marker='o',)

ax.set(xlabel='Electrode distance (mm)', ylabel='Activation',
       title='Activation z = ' + str(zpos) + ' ; Rext 6400 Ohm-cm')
print('Activation values for z = 0.0: ', act_vals[:, 0])
print('Calculation runtime (s): ', fp['run_duration'])

fig2, ax2 = plt.subplots()  # Activation for radial position 0.0
rpos = [-0.5, 0.0, 0.5]
for i, val in enumerate(rpos):
    rpos_idx = np.argmin(np.abs(fp['relec'] - val))
    ax2.plot(fp['zEval'], act_vals[rpos_idx, :], marker='o')

ax2.set(xlabel='Z position', ylabel='Activation', title='Activation rpos = ' + str(rpos) + ' ; Rext 250 Ohm-cm')
ax2.set(xlim=[-1, 10])

# Now do voltage values
fig3, ax3 = plt.subplots()  # Multiple z positions
zpos = [0.0, 0.2, 0.4, 0.6]
for i, val in enumerate(zpos):
    temparray = np.array(fp['zEval'])
    idx = np.argwhere(temparray == val)[0]
    yvals = np.abs(v_vals[:, 1, idx])  # the 1 represents y position == 0.0
    ax3.plot(1 - fp['relec'], yvals, marker='o',)
ax3.set(xlabel='Electrode distance (mm)', ylabel='Voltage', title='Voltage z = 0.0, 2.0, 10, 20; Rext 250 Ohm-cm')

fig4, ax4 = plt.subplots()
for i, val in enumerate(rpos):
    rpos_idx = np.argmin(np.abs(fp['relec'] - val))
    ax4.plot(fp['zEval'], np.abs(v_vals[rpos_idx, 1, :]), marker='o')
ax4.set(xlabel='Z position', ylabel='Voltage', title='Voltage rpos = ' + str(rpos) + ' ; Rext 250 Ohm-cm')
ax4.set(xlim=[-1, 10])

fig5, ax5 = plt.subplots()

ax5.contourf(fp['zEval'], fp['relec'], act_vals, np.arange(0, 10, 1), cmap='hot')
plt.show()
