# PlotSubjectThresholds
# plots the threshold values for a given subject

import subject_data
import numpy as np
import matplotlib.pyplot as plt

SUBJECT = 'S43'
NELEC = 16

xvals = np.arange(0, NELEC) + 1
l_e = NELEC - 1  # last electrode to plot

thrData = subject_data.subj_thr_data(SUBJECT)

plt.plot(xvals, thrData[0][:], marker='s', color='blue')
plt.plot(xvals[1:l_e], thrData[1], marker='s', color='red')
plt.xlabel('Electrode #')
plt.ylabel('Threshold (dB)')
plt.title('Subject S43')
plt.show()

#'Threshold (dB)'


