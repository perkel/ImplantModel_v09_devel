# Script to make figure 1 for basic implant model paper on forward and inverse models

# Import required packages
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import csv
from pylab import cm

# Location of data files
FWDOUTPUTDIR = 'FWD_OUTPUT/'
STD_SUFFIX = 'STD0_2'

# Set default figure values
mpl.rcParams['font.family'] = 'Helvetica'
plt.rcParams['font.size'] = 16
plt.rcParams['axes.linewidth'] = 2
# Generate 2 colors from the 'tab10' colormap
colors = cm.get_cmap('tab10', 2)

# Declare variables
survVals = np.arange(0.05, 0.96, 0.05)
rposVals = np.arange(-0.95, 0.96, 0.05)

# Load monopolar data
datafile = FWDOUTPUTDIR + "testmonopolar_" + STD_SUFFIX + ".csv"
file = open(datafile)
numlines = len(file.readlines())
file.close()

with open(datafile, newline='') as csvfile:
    datareader = csv.reader(csvfile, delimiter=',')
    ncol = len(next(datareader))
    csvfile.seek(0)
    mono_thr = np.empty([numlines, ncol])
    for i, row in enumerate(datareader):
        # Do the parsing
        mono_thr[i ,:] = row

# Load tripolar data
datafile = FWDOUTPUTDIR + "testsigma09_" + STD_SUFFIX + ".csv"
file = open(datafile)
numlines = len(file.readlines())
file.close()

with open(datafile, newline='') as csvfile:
    datareader = csv.reader(csvfile, delimiter=',')
    ncol = len(next(datareader))
    csvfile.seek(0)
    tripol_thr = np.empty([numlines, ncol])
    for i, row in enumerate(datareader):
        # Do the parsing
        tripol_thr[i,:] = row

# # Create figure and add axes object
# fig = plt.figure(1)
# ax = fig.add_axes([0.15, 0.15, 0.75, 0.75])
#
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# # Plot and show our data
# plt.plot(rposVals[1:-2], mono_thr[2, 1:-2], 'ok', label='Monopolar')
# plt.plot(rposVals[1:-2], tripol_thr[2, 1:-2], 'ob', label='Tripolar')
# ax.set_xlim(-1.0, 1.0)
# titletext = 'Survival = ' + str(survVals[2])
# ax.set(xlabel='Electrode radial position (mm)', ylabel='Threshold (dB)', title=titletext)
#
# # Add legend to plot
# ax.legend(bbox_to_anchor=(1, 1), loc=1, frameon=False, fontsize=16)
#
#
# # Create 2nd figure and add axes object
# fig2 = plt.figure(2)
# ax2 = fig2.add_axes([0.15, 0.15, 0.75, 0.75])
#
# ax2.spines['right'].set_visible(False)
# ax2.spines['top'].set_visible(False)
# # Plot and show our data
# plt.plot(rposVals[1:-2], mono_thr[10, 1:-2], 'ok', label='Monopolar')
# plt.plot(rposVals[1:-2], tripol_thr[10, 1:-2], 'ob', label='Tripolar')
# ax2.set_xlim(-1.0, 1.0)
# titletext = 'Survival = 0.5'
# ax2.set(xlabel='Electrode radial position (mm)', ylabel='Threshold (dB)', title=titletext)
#
# # Add legend to plot
# ax2.legend(bbox_to_anchor=(1, 1), loc=1, frameon=False, fontsize=16)
#
# fig3 = plt.figure(3)
# ax3 = fig3.add_axes([0.15, 0.15, 0.75, 0.75])
#
# ax3.spines['right'].set_visible(False)
# ax3.spines['top'].set_visible(False)
# # Plot and show our data
# plt.plot(survVals[1:-2], mono_thr[1:-2, 18], 'ok', label='Monopolar')
# plt.plot(survVals[1:-2], tripol_thr[1:-2, 18], 'ob', label='Tripolar')
# ax3.set_xlim(0, 1.0)
# titletext = 'rpos = 0'
# ax3.set(xlabel='Survival fraction', ylabel='Threshold (dB)', title=titletext)
#
# # Add legend to plot
# ax3.legend(bbox_to_anchor=(1, 1), loc=1, frameon=False, fontsize=16)
nr = 2
nc = 3

fig, ax = plt.subplots(nrows=nr, ncols=nc, sharex='row', sharey='row')
# Plot and show our data
ax[0, 0].plot(rposVals[1:-2], mono_thr[1, 1:-2], 'ok', label='Monopolar')
ax[0, 0].plot(rposVals[1:-2], tripol_thr[1, 1:-2], 'ob', label='Tripolar')
ax[0, 0].set_ylabel("Threshold (dB)")
titletext = 'Survival = ' + str(survVals[2])

ax[0, 1].plot(rposVals[1:-2], mono_thr[10, 1:-2], 'ok', label='Monopolar')
ax[0, 1].plot(rposVals[1:-2], tripol_thr[10, 1:-2], 'ob', label='Tripolar')
titletext = 'Survival = 0.5'
ax[0, 1].set_xlabel('Electrode radial position (mm)')
ax[0, 2].plot(rposVals[1:-2], mono_thr[18, 1:-2], 'ok', label='Monopolar')
ax[0, 2].plot(rposVals[1:-2], tripol_thr[18, 1:-2], 'ob', label='Tripolar')

ax[1, 0].plot(survVals[1:-2], mono_thr[1:-2, 1], 'ok', label='Monopolar')
ax[1, 0].plot(survVals[1:-2], tripol_thr[1:-2, 1], 'ob', label='Tripolar')
ax[1, 1].plot(survVals[1:-2], mono_thr[1:-2, 20], 'ok', label='Monopolar')
ax[1, 1].plot(survVals[1:-2], tripol_thr[1:-2,20], 'ob', label='Tripolar')
ax[1, 0].set_ylabel("Threshold (dB)")

ax[1, 2].plot(survVals[1:-2], mono_thr[1:-2, 35], 'ok', label='Monopolar')

ax[1, 2].plot(survVals[1:-2], tripol_thr[1:-2, 35], 'ob', label='Tripolar')
ax[1, 2].legend(bbox_to_anchor=(1, 1), loc=1, frameon=False, fontsize=16)
ax[1, 1].set_xlabel('Survival fraction')

ylowest = np.zeros(nr)
yhighest = np.zeros(nr)
for row in range(0, nr):
    for col in range(0, nc):
        ylow, yhigh = ax[row, col].get_ylim()
        ylowest[row] = min(ylowest[row], ylow)
        yhighest[row] = max(yhighest[row], yhigh)
        ax[row, col].spines['right'].set_visible(False)
        ax[row, col].spines['top'].set_visible(False)

# match y axes across columns
for row in range(0, nr):
    for col in range(0, nc):
        # ax[row, col].set_ylim(np.around(ylowest[row], -1), np.around(yhighest[row], -1))
        ax[row, col].set_ylim(-20, 40)

print(mono_thr.shape)
# axs4[0, 0].set_xlim(-1.0, 1.0)
# axs4[0, 0].spines['right'].set_visible(False)
# axs4[0, 0].spines['top'].set_visible(False)


# axs4.set(xlabel='Electrode radial position (mm)', ylabel='Threshold (dB)', title=titletext)

# Add legend to plot
# axs4.legend(bbox_to_anchor=(1, 1), loc=1, frameon=False, fontsize=16)



plt.show()
