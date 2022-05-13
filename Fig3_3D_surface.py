# Script to make figure 3 (3D plot) for basic implant model paper on forward and inverse models

# Import required packages
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import csv
from pylab import cm
from scipy import interpolate

# Location of data files
FWDOUTPUTDIR = 'FWD_OUTPUT/ACTR0_5_STDR0_3_TARG500/'
STD_SUFFIX = 'STDR0_3'
# Set default figure values
mpl.rcParams['font.family'] = 'Helvetica'
plt.rcParams['font.size'] = 16
plt.rcParams['axes.linewidth'] = 2
# Generate 2 colors from the 'tab10' colormap
colors = cm.get_cmap('tab10', 2)

# Declare variables
survVals = np.arange(0.05, 0.96, 0.02)
rposVals = np.arange(-0.95, 0.96, 0.02)

# Load monopolar data
datafile = FWDOUTPUTDIR + "Monopolar_2D_" + STD_SUFFIX + ".csv"
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
        mono_thr[i,:] = row

# Load tripolar data
datafile = FWDOUTPUTDIR + "Tripolar_09_2D_" + STD_SUFFIX + ".csv"
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
        tripol_thr[i, :] = row

fig = plt.figure()
ax = plt.axes(projection='3d')
# set up 2D interpolation
rp_curt = rposVals[0:-2]
f_interp = interpolate.interp2d(rp_curt, survVals, mono_thr[:, 0:-2])
xnew = np.linspace(rposVals[0], rposVals[-2], 50)
ynew = np.linspace(survVals[0], survVals[-2], 50)
XN_mp, YN_mp = np.meshgrid(xnew, ynew)
znew_mp = f_interp(xnew, ynew)
ax.plot_surface(XN_mp, YN_mp, znew_mp, rstride = 1, cstride=1, cmap='viridis', linewidth=0.5)
ax.set_xlabel('Electrode position')
ax.set_ylabel('Survival')
ax.set_zlabel('Monopol. threshold (dB)')
m_min, m_max = ax.get_zlim()


fig2 = plt.figure()
ax2 = plt.axes(projection='3d')
# set up 2D interpolation
f_interp = interpolate.interp2d(rp_curt, survVals, tripol_thr[:, 0:-2])
xnew = np.linspace(rposVals[8], rposVals[88], 50)
ynew = np.linspace(survVals[3], survVals[42], 50)
XN, YN = np.meshgrid(xnew, ynew)
znew_tp = f_interp(xnew, ynew)
ax2.plot_surface(XN, YN, znew_tp, rstride = 1, cstride=1, cmap='viridis', linewidth=0.5)
ax2.set_xlabel('Electrode position')
ax2.set_ylabel('Survival')
ax2.set_zlabel('Tripol. threshold (dB)')
t_min, t_max = ax2.get_zlim()

ax.set_zlim(min(m_min, t_min), max(m_max, t_max))
ax2.set_zlim(min(m_min, t_min), max(m_max, t_max))

fig3 = plt.figure()
ax3 = plt.contourf(XN, YN, znew_mp, np.arange(m_min, m_max, (m_max-m_min)/10))
ax4 = plt.contourf(XN, YN, znew_tp, np.arange(t_min, t_max, (t_max-t_min)/10))


mpcontour = ax3.allsegs[0]
tpcontour = ax4.allsegs[0]

nmp = len(mpcontour[0])
ntp = len(tpcontour[0])

dists = np.zeros([nmp, ntp])
for i, mp in enumerate(mpcontour[0]):
    # find point in tpcontour with minimum distance to this point
    for j, tp in enumerate(tpcontour[0]):
        thisd = (mp[0]-tp[0])**2 + (mp[1] - tp[1])**2
        dists[i, j] = thisd

mycoords = np.unravel_index(np.argmin(dists), dists.shape)

print("coords: ", mpcontour[0][mycoords[0]], mpcontour[0][mycoords[1]])
np.save('testcontourdata.npy', [mpcontour[0], tpcontour[0]])


plt.show()