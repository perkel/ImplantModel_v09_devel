# Script to make figure showing scenarios for basic implant model paper on forward and inverse models

# Import required packages
import matplotlib.pyplot as plt
from matplotlib.ticker import IndexLocator
import csv
from pylab import cm
from common_params import *
import set_scenario

# Set default figure values
plt.rcParams['font.size'] = 16
plt.rcParams['axes.linewidth'] = 2
# Generate 2 colors from the 'tab10' colormap
colors = cm.get_cmap('tab10', 2)

#scenarios = ['Gradual80R00', 'RampSurvR03', 'Ramp80Rvariable1', 'RampRposSGradual80']
scenarios = ['Gradual80R00', 'RampSurvR03']
nrows = len(scenarios)
ncols = 2
fig, axs = plt.subplots(nrows, ncols, sharex='all', figsize=(10, 8))
fig.tight_layout(pad=5, w_pad=8, h_pad=3.0)
axs_r = []
posplot = []
survplot = []

# fig2 = plt.figure(figsize=(10, 8), constrained_layout=True)
# axs2 = fig2.subplots(nrows, ncols, sharex=True, sharey=True)
# axs3 = fig2.subplots(nrows, ncols, sharex=True, sharey=True)
count = 0

for scenario in scenarios:
    [survVals, ELECTRODES['rpos']] = set_scenario.set_scenario(scenario, NELEC)

    # Load data
    datafile = FWDOUTPUTDIR + 'FwdModelOutput_' + scenario + '.csv'
    file = open(datafile)
    numlines = len(file.readlines())
    file.close()

    # Declare variables
    electrodes = np.empty(numlines)
    rpos = np.empty(numlines)
    mono_thr = np.empty(numlines)
    tripol_thr = np.empty(numlines)

    with open(datafile, newline='') as csvfile:
        datareader = csv.reader(csvfile, delimiter=',')
        for i, row in enumerate(datareader):
            # Do the parsing
            electrodes[i] = float(row[0])
            survVals[i] = float(row[1])
            rpos[i] = float(row[2])
            mono_thr[i] = float(row[3])
            tripol_thr[i] = float(row[4])

    row = count
    # col = count % ncols
    col = 0
    marker_style = dict(color='black', linestyle='none', marker='o',
                        markersize=8, markerfacecoloralt='tab:red')

    axs[row, col].plot(electrodes + 1, rpos, 'ok')
    temp_axis = axs[row, col].twinx()
    # axs_r[r]
    temp_axis.plot(electrodes + 1, survVals, fillstyle='none', **marker_style)
    temp_axis.set_ylabel('Survival frac.')
    temp_axis.spines['right'].set_visible(False)
    temp_axis.spines['top'].set_visible(False)
    temp_axis.set_ylim((0.1, 0.9))
    axs[row, col].xaxis.set_major_locator(IndexLocator(base=15, offset=0))
    if row == nrows - 1:
        axs[row, col].set(xlabel='Electrode number')

    axs[row, col].set(ylabel='El. pos. (mm)')
    axs[row, col].spines['right'].set_visible(False)
    axs[row, col].spines['top'].set_visible(False)
    axs[row, col].set_ylim((-0.8, 0.8))

    col = 1  # Plot Threshold data
    axs[row, col].plot(electrodes + 1, mono_thr, 'ok', label='Monopolar')
    axs[row, col].plot(electrodes + 1, tripol_thr, 'ob', label='Tripolar')
    # axs[row, col].set_title(scenario)
    axs[row, col].annotate(scenario, (0.5, 0.9 - (0.85*(row/nrows))), xycoords='figure fraction',
                           horizontalalignment='center', fontsize=18)
    axs[row, col].spines['right'].set_visible(False)
    axs[row, col].spines['top'].set_visible(False)
    # axs[row, col].set_xlim(-0.9, 0.9)
    if row == nrows - 1:
        axs[row, col].set(xlabel='Electrode number')
    if col == 1:
        axs[row, col].set(ylabel='Threshold (dB)')

    # axs2[row, col].plot(electrodes, rpos, 'ok')
    # axs3[row, col] = axs2[row, col].twinx()
    # axs3[row, col].plot(electrodes, survVals, 'ob')
    # axs2[row, col].set_title(scenario)
    # #axs2[row, col].spines['right'].set_visible(False)
    # axs2[row, col].spines['top'].set_visible(False)
    # #axs[row, col].set_xlim(-0.9, 0.9)
    # if row == nrows-1:
    #     axs2[row, col].set(xlabel='Electrode number')
    # if col == 0:
    #     axs2[row, col].set(ylabel='Electrode position (mm)')

    count += 1

#
plt.savefig('Fig_Scenarios.pdf', format='pdf')

plt.show()

# # Add legend to plot
# ax.legend(bbox_to_anchor=(1, 1), loc=1, frameon=False, fontsize=16)
# plt.show()
#
# # Save figure
# plt.savefig('Figure1.png', dpi=300, transparent=False, bbox_inches='tight')
#
# # Figure 2 Constant position, changing survival value
# # Load data
# datafile = "FWD_OUTPUT/FwdModelOutput_RampSurvR07.csv"
# file = open(datafile)
# numlines = len(file.readlines())
# file.close()
#
# # Declare variables
# electrodes = []
# survVals = np.empty(numlines)
# rpos = np.empty(numlines)
# mono_thr = np.empty(numlines)
# tripol_thr = np.empty(numlines)
#
# with open(datafile, newline='') as csvfile:
#     datareader = csv.reader(csvfile, delimiter=',')
#     for i, row in enumerate(datareader):
#         # Do the parsing
#         electrodes.append(row[0])
#         survVals[i] = float(row[1])
#         rpos[i] = float(row[2])
#         mono_thr[i] = float(row[3])
#         tripol_thr[i] = float(row[4])
#
# # Create figure and add axes object
# fig = plt.figure()
# ax = fig.add_axes([0.15, 0.15, 0.8, 0.8])
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# # Plot and show our data
# plt.plot(survVals, mono_thr, 'ok', label='Monopolar')
# plt.plot(survVals, tripol_thr, 'ob', label='Tripolar')
# ax.set_xlim(0, 0.9)
# ax.set(xlabel='Survival fraction', ylabel='Threshold (dB)')
#
# # Add legend to plot
# ax.legend(bbox_to_anchor=(1, 1), loc=1, frameon=False, fontsize=16)
# plt.show()
#
# # Save figure
# plt.savefig('Figure2b_07.png', dpi=300, transparent=False, bbox_inches='tight')
