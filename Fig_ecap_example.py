# Fig_ecap_example  30 Nov 2022 David Perkel

# This script makes a simple figure with 4 ecap examples, with near and far electrode positions and high
# and loe neuronal density.

import matplotlib.pyplot as plt
import csv
import numpy as np
from common_params import *

filenames = []
filenames.append(FWDOUTPUTDIR + 'Dist1_5_low_density.csv')
filenames.append(FWDOUTPUTDIR + 'Dist0_5_low_density.csv')
filenames.append(FWDOUTPUTDIR + 'Dist1_5_full_density.csv')
filenames.append(FWDOUTPUTDIR + 'Dist0_5_full_density.csv')
n = 4

vals = []
with open(filenames[0], mode='r') as data_file:
    data_reader = csv.reader(data_file, delimiter=',', quotechar='"')
    for row in data_reader:
        vals.append(row)
data_file.close()
f_l_stim = np.array(vals[0])
f_l_neur = np.array(vals[1])

vals = []
with open(filenames[1], mode='r') as data_file:
    data_reader = csv.reader(data_file, delimiter=',', quotechar='"')
    for row in data_reader:
        vals.append(row)
data_file.close()
n_l_stim = vals[0]
n_l_neur = vals[1]

vals = []
with open(filenames[2], mode='r') as data_file:
    data_reader = csv.reader(data_file, delimiter=',', quotechar='"')
    for row in data_reader:
        vals.append(row)
data_file.close()
f_h_stim = vals[0]
f_h_neur = vals[1]

vals = []
with open(filenames[3], mode='r') as data_file:
    data_reader = csv.reader(data_file, delimiter=',', quotechar='"')
    for row in data_reader:
        vals.append(row)
data_file.close()
n_h_stim = vals[0]
n_h_neur = vals[1]

fig, ax = plt.subplots()
ax.plot(f_l_stim[0:100], f_l_neur[0:100], 'or', label='Dist 1.5; density 20%')
ax.plot(n_h_stim[0:50], n_h_neur[0:50], 'ob', label='Dist 1.5; density 100%')
# ax.plot(n_l_stim, n_l_neur, label='Dist 0.5; density 20%')
# ax.plot(n_h_stim, n_h_neur, label='Dist 0.5; density 100%')
ax.legend()

plt.show()

