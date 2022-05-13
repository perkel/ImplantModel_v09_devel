#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 25 20:04:55 2020

@author: perkel
"""

import pickle
import matplotlib.pyplot as plt

FILENAME = '4Jan2022HighResolution_RInt70_nonans.dat'
# FILENAME = '20Jan2022MedResolution_RInt70.dat'

with open(FILENAME, 'rb') as combined_data:
    data = pickle.load(combined_data)
combined_data.close()

fp = data[0]
v_vals = data[1]
act_vals = data[2]


fig, ax = plt.subplots()
ax.plot(fp['relec'], act_vals[:, 0], marker='o',)
ax.set_yscale('log')
ax.set(xlabel='Electrode position', ylabel='Activation', title='Activation z = 0.0')
print('fT: ', act_vals[:, 0])

# fig, ax2 = plt.subplots()
# ax2.plot(fp['relec'], v_vals[:, 0, 0],  marker = '*')
# ax2.set(xlabel='Electrode position', ylabel='Voltage', title='Voltage')
# print('vT: ', v_vals[:, 0])
#
# fig, ax3 = plt.subplots()
# ax3.plot(fp['relec'], act_vals[:, 10],  marker='*')
# ax3.set(xlabel='Electrode position', ylabel='Activation', title='Activation z = 1.0')
#
# fig, ax4 = plt.subplots()
# ax4.plot(fp['relec'], act_vals[:, 15],  marker='*')
# ax4.set(xlabel='Electrode position', ylabel='Activation', title='Activation z = 1.5')
#
# fig, ax5 = plt.subplots()
# ax5.plot(fp['relec'], act_vals[:, 20],  marker='*')
# ax5.set(xlabel='Electrode position', ylabel='Activation', title='Activation z = 2.0')
#
# fig, ax6 = plt.subplots()
# ax6.plot(fp['zEval'], v_vals[20, 3, :],  marker='*')
# ax6.set(xlabel='Position along cochlea (mm)', ylabel='Voltage', title='Voltage rElec = 0.0')
#
# fig, ax7 = plt.subplots()
# ax7.plot(fp['zEval'], act_vals[19, :],  color = 'red', marker='o')
# ax7.set(xlabel='Position along cochlea (mm)', ylabel='Activation', title='Activation rElec = -.5, 0.0, 0.5')
# ax7.plot(fp['zEval'], act_vals[9, :],  color='blue', marker='o')
# ax7.plot(fp['zEval'], act_vals[29, :],  color='green', marker='o')




plt.show()
print('done')
