# Script to make figure 1 for basic implant model paper on forward and inverse models

# Import required packages
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import csv
from pylab import cm

# Set default figure values
mpl.rcParams['font.family'] = 'Helvetica'
plt.rcParams['font.size'] = 16
plt.rcParams['axes.linewidth'] = 2
# Generate 2 colors from the 'tab10' colormap
colors = cm.get_cmap('tab10', 2)

# Load data
datafile = "FWD_OUTPUT/FwdModelOutput_Ramp80Rvariablecombined.csv"
file = open(datafile)
numlines = len(file.readlines())
file.close()

# Declare variables
electrodes = []
survVals = np.empty(numlines)
rpos = np.empty(numlines)
mono_thr = np.empty(numlines)
tripol_thr = np.empty(numlines)

with open(datafile, newline='') as csvfile:
    datareader = csv.reader(csvfile, delimiter=',')
    for i, row in enumerate(datareader):
        # Do the parsing
        electrodes.append(row[0])
        survVals[i] = float(row[1])
        rpos[i] = float(row[2])
        mono_thr[i] = float(row[3])
        tripol_thr[i] = float(row[4])

# Create figure and add axes object
fig = plt.figure()
ax = fig.add_axes([0.15, 0.15, 0.8, 0.8])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
# Plot and show our data
plt.plot(rpos, mono_thr, 'ok', label='Monopolar')
plt.plot(rpos, tripol_thr, 'ob', label='Tripolar')
ax.set_xlim(-0.9, 0.9)
ax.set(xlabel='Electrode radial position (mm)', ylabel='Threshold (dB)')

# Add legend to plot
ax.legend(bbox_to_anchor=(1, 1), loc=1, frameon=False, fontsize=16)
plt.show()

# Save figure
plt.savefig('Figure1.png', dpi=300, transparent=False, bbox_inches='tight')

# Figure 2 Constant position, changing survival value
# Load data
datafile = "FWD_OUTPUT/FwdModelOutput_RampSurvR07.csv"
file = open(datafile)
numlines = len(file.readlines())
file.close()

# Declare variables
electrodes = []
survVals = np.empty(numlines)
rpos = np.empty(numlines)
mono_thr = np.empty(numlines)
tripol_thr = np.empty(numlines)

with open(datafile, newline='') as csvfile:
    datareader = csv.reader(csvfile, delimiter=',')
    for i, row in enumerate(datareader):
        # Do the parsing
        electrodes.append(row[0])
        survVals[i] = float(row[1])
        rpos[i] = float(row[2])
        mono_thr[i] = float(row[3])
        tripol_thr[i] = float(row[4])

# Create figure and add axes object
fig = plt.figure()
ax = fig.add_axes([0.15, 0.15, 0.8, 0.8])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
# Plot and show our data
plt.plot(survVals, mono_thr, 'ok', label='Monopolar')
plt.plot(survVals, tripol_thr, 'ob', label='Tripolar')
ax.set_xlim(0, 0.9)
ax.set(xlabel='Survival fraction', ylabel='Threshold (dB)')

# Add legend to plot
ax.legend(bbox_to_anchor=(1, 1), loc=1, frameon=False, fontsize=16)
plt.show()

# Save figure
plt.savefig('Figure2b_07.png', dpi=300, transparent=False, bbox_inches='tight')
