#  summary_stats.py
#  David Perkel 18 November 2022

from common_params import *  # import common values across all models
import matplotlib.pyplot as plt
import seaborn as sns
import csv
import scipy.stats as stats

# Reads a summary file, and tests whether average rpos error is less than chance based on shuffling
summary_file_name = INVOUTPUTDIR + 'summary_inverse_fit_results.npy'

[scenarios, rpos_summary] = np.load(summary_file_name, allow_pickle=True)
nscen = len(scenarios)
rpos_vals = []
rpos_fit_vals = []
thresh_err_summary = np.zeros((nscen, 2))
rpos_err_summary = np.zeros(nscen)
dist_corr = np.zeros(nscen)
dist_corr_p = np.zeros(nscen)

for i, scen in enumerate(scenarios):
    rpos_vals.append(rpos_summary[i][0])
    rpos_fit_vals.append(rpos_summary[i][1])

# Now should have actual and fit values. Write them to CSV
summ_output_rpos = INVOUTPUTDIR + 'summary_actual_position.csv'
with open(summ_output_rpos, mode='w') as data_file:
    data_writer = csv.writer(data_file, delimiter=',', quotechar='"')
    for i, scen in enumerate(scenarios):
        data_writer.writerow([scen, rpos_vals[i][0], rpos_vals[i][1], rpos_vals[i][2], rpos_vals[i][3],
                              rpos_vals[i][4], rpos_vals[i][5], rpos_vals[i][6], rpos_vals[i][6], rpos_vals[i][7],
                              rpos_vals[i][8], rpos_vals[i][9], rpos_vals[i][10], rpos_vals[i][11], rpos_vals[i][12],
                              rpos_vals[i][13], rpos_vals[i][14]])
    data_file.close()

summ_output_rpos = INVOUTPUTDIR + 'summary_fit_position.csv'
with open(summ_output_rpos, mode='w') as data_file:
    data_writer = csv.writer(data_file, delimiter=',', quotechar='"')
    for i, scen in enumerate(scenarios):
        data_writer.writerow([scen, rpos_fit_vals[i][0], rpos_fit_vals[i][1], rpos_fit_vals[i][2], rpos_fit_vals[i][3],
                              rpos_fit_vals[i][4], rpos_fit_vals[i][5], rpos_fit_vals[i][6], rpos_fit_vals[i][6],
                              rpos_fit_vals[i][7], rpos_fit_vals[i][8], rpos_fit_vals[i][9], rpos_fit_vals[i][10],
                              rpos_fit_vals[i][11], rpos_fit_vals[i][12], rpos_fit_vals[i][13], rpos_vals[i][14]])
    data_file.close()

# Now get detailed data from the CSV summary file
dummy__ = 0.0
summary_csv_file_name = INVOUTPUTDIR + 'summary_inverse_fit_results.csv'
with open(summary_csv_file_name, mode='r') as data_file:
    entire_file = csv.reader(data_file, delimiter=',', quotechar='"')
    for row, row_data in enumerate(entire_file):
        print('row_data is ', row_data)
        if row == 0:
            pass
        else:
            [dummy__, thresh_err_summary[row-1, 0], thresh_err_summary[row-1, 1],
             rpos_err_summary[row-1], dist_corr[row-1], dist_corr_p[row-1]] = row_data

    data_file.close()

# Loop on scenarios again. Compute pairwise mean absolute position error
mean_errs = np.zeros([nscen, nscen])
median_errors = np.zeros((nscen, nscen))
corr_vals = np.zeros([nscen, nscen])
corr_p = np.zeros([nscen, nscen])

for i, scen_i in enumerate(scenarios):
    for j, scen_j in enumerate(scenarios):
        rposerrs = np.subtract(rpos_fit_vals[i], rpos_vals[j])
        mean_errs[i, j] = np.mean(np.abs(rposerrs))
        [dist_corr, dist_corr_p] = stats.pearsonr(1.0 - (rpos_fit_vals[i]), 1.0 - (rpos_vals[j]))
        corr_vals[i, j] = dist_corr
        corr_p[i, j] = dist_corr_p

# now we have the matrix. Let's plot a histogram of all the values
a = mean_errs.flatten()
mean = np.mean(a)
std = np.std(a)
fig1, axes1 = plt.subplots()
axes1.hist(a, 50)
diag = np.diag(mean_errs)
for k in range(len(diag)):
    plt.plot(diag[k], 1, 'or')

axes1.set_xlabel("Mean error (mm)")
axes1.set_ylabel("# observations")
axes1.plot(mean, 0.2, 'ob')
axes1.plot(mean-std, 0.2, 'og')
axes1.plot(mean+std, 0.2, 'og')

median = np.median(median_errors.flatten())
fig_med, axes_med = plt.subplots()
axes_med.hist(median, 50)
diag_med = np.diag(median_errors)
for kk in range(len(diag_med)):
    plt.plot(diag_med[kk], 1, 'or')

b = corr_vals.flatten()
b_p = corr_p.flatten()
mean_cr = np.mean(b)
std_cr = np.std(b)
signif_corr = b[np.argwhere(b_p < 0.05)]
fig2, axes2 = plt.subplots()
axes2.hist(signif_corr, 50)
diag_cr = np.diag(corr_vals)
diag_p = np.diag(corr_p)
for k in range(len(diag_cr)):
    if diag_p[k] < 0.05:
        plt.plot(diag_cr[k], 1, 'or')

axes2.set_xlabel("Correlation")
axes2.set_ylabel("# observations")
axes2.plot(mean_cr, 0.2, 'ob')
axes2.plot(mean_cr - std_cr, 0.2, 'og')
axes2.plot(mean_cr + std_cr, 0.2, 'og')

fig3, axes3 = plt.subplots()
c = np.sort(a)
nvals = len(a)
yvals = np.arange(nvals)/nvals
plt.plot(c, yvals)
axes3.set_xlabel('Mean error (mm)')
axes3.set_ylabel('Fraction')
print("mean absolute errors:")
for k, val in enumerate(diag):
    print(k, val)
    d = np.abs(val-c)
    e = np.argmin(d)
    plt.plot(val, yvals[e], 'or')
    txt = scenarios[k]
    plt.text(val-0.015, yvals[e]+0.1, txt, rotation=90)

fig4, axes4 = plt.subplots()
# for each subject, plot all points, with CT on x axis, fit on y axis, corr line

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
n_col = len(colors)

for q, scen in enumerate(scenarios):
    plt.plot(rpos_vals[q], rpos_fit_vals[q], 'o', color=colors[q % n_col])
    x = rpos_vals[q]
    xx = np.ones(len(x))
    A = np.vstack([x, xx]).T
    slope, intercept = np.linalg.lstsq(A, rpos_fit_vals[q], rcond=None)[0]
    plt.plot(rpos_vals[q], slope * rpos_vals[q] + intercept, color=colors[q % n_col])

# Overlay fit across all points in black


axes4.set_xlabel('rpos')
axes4.set_ylabel('fit rpos')

# Plot #5 is mean error v. correlation
fig5, axes5 = plt.subplots()
plt.plot(diag, diag_cr, 'o')
axes5.set_xlabel('Mean error (mm)')
axes5.set_ylabel('Correlation slope')


sns.set_style()

plt.figure()
ax7 = sns.swarmplot(thresh_err_summary)  # , x=['Monopolar', 'Tripolar'], y='Threshold error (dB)')
ax7.set_ylabel('Threshold error (dB)')

fig8, axs8 = plt.subplots(1, 3)

axs8[0].plot(thresh_err_summary[:, 0], thresh_err_summary[:, 1], 'o', color='black')
axs8[0].plot([0, 0.8], [0, 0.8], linestyle='dashed', color='black')  # line of slope 1.0
axs8[0].set_xlabel('Monopolar threshold error (dB)')
axs8[0].set_ylabel('Tripolar threshold error (dB)')
axs8[0].spines['top'].set_visible(False)
axs8[0].spines['right'].set_visible(False)

sns.swarmplot(diag, ax=axs8[1], color='black')
axs8[1].set_ylabel('Distance error (mm)')
axs8[1].spines['top'].set_visible(False)
axs8[1].spines['right'].set_visible(False)
plt.tick_params(
    axis='x',           # changes apply to the x-axis
    which='both',       # both major and minor ticks are affected
    bottom=False,       # ticks along the bottom edge are off
    top=False,          # ticks along the top edge are off
    labelbottom=False)  # labels along the bottom edge are off

sns.swarmplot(diag_cr, ax=axs8[2], color='black')
axs8[2].set_ylabel('Pearson\'s r')
axs8[2].spines['top'].set_visible(False)
axs8[2].spines['right'].set_visible(False)
plt.tick_params(
    axis='x',           # changes apply to the x-axis
    which='both',       # both major and minor ticks are affected
    bottom=False,       # ticks along the bottom edge are off
    top=False,          # ticks along the top edge are off
    labelbottom=False)  # labels along the bottom edge are off

plt.show()

print('done')
