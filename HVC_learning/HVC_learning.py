# HVC_learning.  This is code to replicate the work by Duffy, Abe et al.

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as io
import scipy.integrate as integrate
import time as ti


def int_fun(t_prime, tmp0, tmp1, tmp2, tmp3):
    return np.power(tmp3 - t_prime, 5) * np.exp(-(tmp3 - t_prime) / 5) * tmp2 * tmp1 * tmp0


start_time = ti.time()

n_hvc = 20
n_ra = 10
n_mn = 2
n_songs = 20

# Load target motor patterns
data = io.loadmat('MotorPool7.mat')
# unpack variables
m_1_bar = np.ravel(data['m_1_bar'])
m_2_bar = np.ravel(data['m_2_bar'])

# Key features of a song
song_dur = 200  # milliseconds
dt = 0.5  # timestep within song
hvc_burst_int = 2.0

# synaptic properties
tau_syn = 5.0
n_times = int(song_dur/dt) + 1
times = np.linspace(0, song_dur, num=n_times)
g_syn = np. zeros(len(times))
g_s_peak = 0.10  # was 0.13 in the Duffy/Abe paper
v_excite = 0.0
v_inhib = -70.0
v_leak = -60.0
g_syn[0] = g_s_peak
g_syn_lman_peak = 0.05

# motor neuron activity and fixed ra-motor neuron weights
mn = np.zeros((n_mn, n_times))
w_mn = np.zeros((n_mn, n_ra))
w_mn[0, :int(n_ra/2)] = 1.0  # first half of RA neurons projects to M1
w_mn[1, int(n_ra/2):] = 1.0  # second half of RA neurons projects to M2
mn_drive = 60  # motor pool drive
tau_mn = 5.0  # time constant for motor pool decay

# TODO make this a separate function
for i, t in enumerate(times):  # make generic synaptic conductance trace
    if i > 0:
        if t == hvc_burst_int or t == 2.0*hvc_burst_int:
            g_syn[i] = g_syn[i-1] + g_s_peak  # add some synaptic conductance
        else:
            g_syn[i] = g_syn[i-1] * (1.0 - (dt/tau_syn))

# Key features of HVC neurons
time_buf_frac = 0.1
first_burst = time_buf_frac*song_dur
last_burst = song_dur - first_burst
hvc_btime = np.linspace(first_burst, last_burst, num=n_hvc)  # hvc burst times
hvc_g_syn = np.zeros((n_hvc, n_times))
for i in range(n_hvc):
    this_burst = hvc_btime[i]
    start_pt = int(np.argwhere(times > this_burst)[0])
    remain_pts = n_times - start_pt
    hvc_g_syn[i, start_pt:] = g_syn[:remain_pts]
    # plt.plot(times, hvc_g_syn[i])
    # plt.show()

# Initialize Synaptic Weights with Uniform Probability on interval [0,2.0]
w_master = 2.0 * np.random.rand(n_ra, n_hvc)
w_trace = np.zeros((n_ra, n_hvc, n_songs))

# initialize RA neurons
ra_tau = 20.0  # membrane time constant (ms)
ra_vrest = -65.0
v_leak = -65.0
ra_vspike = 10.0
ra_vahp = -70.0
ra_thresh = -50.0
ra_v_trace = np.zeros((n_ra, n_times)) + ra_vrest
hvc_ra_syn_trace = np.zeros((n_ra, n_hvc, n_times))
ra_sp_trace = np.zeros((n_ra, n_times), dtype=int)
lman_syn_ex_trace = np.zeros((n_ra, n_times))
lman_syn_in_trace = np.zeros((n_ra, n_times))
eligibility = np.zeros((n_ra, n_hvc, n_times))
eligibility_trace = np.zeros((n_ra, n_hvc, n_times, n_songs))

# initialize error log

error_trace = np.zeros((n_songs, n_times))
overall_error_trace = np.zeros(n_songs)
eta = 0.00005  # learning rate


for song in range(n_songs):  # loop on songs
    song_start = ti.time()
    print('starting song ', song, ' of ', n_songs)

    # calculate poisson-distributed LMAN firing times
    # TODO check whether these are poisson distributed.
    # Generating LMAN Spikes. There is one excitatory and one "inhibitory" lman neuron
    v_lman1 = np.random.rand(n_ra, n_times)
    lman_spikes_ex = (0.08 * dt) > v_lman1  # lambda = .08, .2
    v_lman2 = np.random.rand(n_ra, n_times)
    lman_spikes_in = (0.28 * dt) > v_lman2  # lambda = .08, .2
    g_syn_lman_template = np.zeros(n_times)
    g_syn_lman_template[0] = g_syn_lman_peak
    g_syn_lman_ex = np.zeros((n_ra, n_times))
    g_syn_lman_in = np.zeros((n_ra, n_times))

    for i, t in enumerate(times):  # make generic synaptic conductance trace
        if i > 0:
            g_syn_lman_template[i] = g_syn_lman_template[i - 1] * (1.0 - (dt / tau_syn))

    # convolve with synaptic conductance time courses (should be a function call, and use it for hvc-ra above, too
    for i in range(n_ra):
        g_syn_lman_ex[i, :] = np.convolve(lman_spikes_ex[i, :], g_syn_lman_template, 'same')
        g_syn_lman_in[i, :] = np.convolve(lman_spikes_ex[i, :], -g_syn_lman_template, 'same')

    # loop on times and update ra voltage
    for i, t in enumerate(times):
        if i > 0:

            # first set cells that previously had spikes to an AHP potential
            spiked = np.argwhere(ra_v_trace[:, i-1] > 0.0)
            if spiked.size > 0:
                ra_v_trace[spiked, i] = ra_vahp

            # calculate conductances and currents
            g_syn_ra = hvc_g_syn[:, i]  # conductance array for this time step
            ra_g_syn = np.matmul(w_master, g_syn_ra)  # array of synaptic inputs to RA cells
            lman_i_syn_ex = np.multiply(g_syn_lman_ex[:, i], (ra_v_trace[:, i-1] - v_excite))
            lman_i_syn_in = np.multiply(g_syn_lman_in[:, i], (ra_v_trace[:, i - 1] - v_inhib))
            lman_syn_ex_trace[:, i] = lman_i_syn_ex
            lman_syn_in_trace[:, i] = lman_i_syn_in

            ra_i_syn = np.multiply(ra_g_syn, (ra_v_trace[:, i-1] - v_excite))
            temp00 = np.multiply(w_master, g_syn_ra)  # critical for eligibility
            hvc_ra_syn_trace[:, :, i] = temp00
            dv_ra = -dt * (ra_i_syn + lman_i_syn_ex + lman_i_syn_in + (ra_v_trace[:, i-1] - ra_vrest))
            ra_v_trace[:, i] = ra_v_trace[:, i-1] + dv_ra
            if spiked.size > 0:
                ra_v_trace[spiked, i] = ra_vahp

            # Now detect threshold crossings
            spiking = np.argwhere(ra_v_trace[:, i] > ra_thresh)
            spiking2 = ra_v_trace[:, i] > ra_thresh
            ra_v_trace[spiking, i] = ra_vspike
            # Need to keep track of RA spike times
            ra_sp_trace[spiking, i] = 1

            # calculate motor neuron activity
            d_mn = np.matmul(w_mn, spiking2) - dt*(mn[:, i-1] - mn_drive)/tau_mn
            mn[:, i] = mn[:, i-1] + d_mn


    # End of song. Calculate error
    error0 = np.square(np.subtract(mn[0, :], m_1_bar))
    error1 = np.square(np.subtract(mn[1, :], m_2_bar))
    error_trace[song, :] = np.add(error0, error1)
    overall_error_trace[song] = np.sum(error_trace[song, :])

    # Calculate where to do reinforcement
    if song < 5:  # average all previous songs
        recent_error_mean = np.mean(error_trace[:, :])
    else:
        recent_error_mean = np.mean(error_trace[song-5:song, :])

    # Calculate recent average of lman activity
    if song < 5:  # average all previous songs
        recent_lman_ex = np.mean(lman_syn_ex_trace[:, :])
        recent_lman_in = np.mean(lman_syn_in_trace[:, :])
    else:
        recent_lman_ex = np.mean(lman_syn_ex_trace[song-5:song, :])
        recent_lman_in = np.mean(lman_syn_in_trace[song - 5:song, :])

    theta = error_trace - recent_error_mean
    reinf = np.zeros(n_times)
    reinf[np.argwhere(theta > 0)] = 1
    reinf[np.argwhere(theta < 0)] = -1

    # Calculate eligibility trace
    for i in range(n_hvc):
        # loop on RA neurons
        for j in range(n_ra):
            # loop on time since onset

            for k, time in enumerate(times):
                # need to integrate from time == 0 to this time
                temp0 = hvc_ra_syn_trace[j, i, k]
                temp1 = np.add(recent_lman_ex, recent_lman_in)
                temp2 = np.subtract(lman_syn_ex_trace[j, k], temp1)
                [result, accuracy] = integrate.quad(int_fun, 0, time, args=(temp0, temp1, temp2, time))
                eligibility[j, i, k] = result

    # save weight matrix
    w_trace[:, :, song] = w_master
    # update weights
    temp000 = np.multiply(reinf, eligibility)
    temp001 = np.sum(temp000, axis=2)
    dw = dt*eta*temp001
    w_master += dw
    song_end = ti.time()
    print('song took ', song_end - song_start, ' s')

    # plt.plot(times, ra_v_trace[1, :], 'r')
    # plt.plot(times, ra_v_trace[2, :], 'k')
    # plt.plot(times, ra_v_trace[4, :], 'b')
    #
    # plt.figure()
    plt.plot(times, mn[0], 'g')
    plt.plot(times, m_1_bar, 'g-')
    plt.plot(times, mn[1], 'r')
    plt.plot(times, m_2_bar, 'r-')
    plt.show()
    #
    # plt.figure()
    # plt.plot(times, error_trace[song, :])
    # plt.show()

end_time = ti.time()
elapsed = end_time - start_time
print('time elapsed = ', elapsed, ' s.')
plt.plot(overall_error_trace)
plt.show()
print('another line')