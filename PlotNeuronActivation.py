# Import critical packages
import matplotlib.pyplot as plt
from common_params import *

#FWDOUTPUTDIR = 'FWD_OUTPUT/ACTR1000_STDR0_3_TARG100/'
#STD_SUFFIX = 'STDR0_3'

def PlotNeuronActivation():
    ACTIVATION_FILE = FWDOUTPUTDIR + 'neuronact_' + STD_TEXT + '.npy'
    print("Activation file: ", ACTIVATION_FILE)
    neuronact = np.load(ACTIVATION_FILE, allow_pickle=True)

    # print("max for surv 0.2, rpos(9): ", max(neuronact[4, 9, 0, 0, :]))
    # print("sum: ", sum(neuronact[4, 9, 0, 0, :]))
    # print("max for surv 0.8, rpos(29): ", max(neuronact[16, 29, 0, 0, :]))
    # print("sum: ", sum(neuronact[15, 29, 0, 0, :]))

    # Sanity check: is the sum across neuronact values really 100?
    posvals = np.arange(0, 33, 0.1) - 14.6 - 3 * ESPACE

    survVals = np.arange(0.2, 0.96, 0.05)
    rposVals = np.arange(-0.8, 0.96, 0.05)

    # Make plots
    nsurv = 3
    nrpos = 3
    fig, axs = plt.subplots(nsurv, nrpos, sharex=True, sharey=True)
    survidxs = [4, 9, 11]  # must have length nsurv
    rposidxs = [9, 19, 29]  # must have length nrpos

    # Set labels and plot
    for i, ax in enumerate(axs.flat):
        row = np.int(i / 3)
        col = int(np.mod(i, 3))
        if row == 0:
            titletext = 'rpos = %.2f' % rposVals[rposidxs[col]]
            ax.set_title(titletext)

        ax.plot(posvals + 1.1, neuronact[survidxs[row], rposidxs[col], 0, 0, :], 'b.')
        ax.plot(posvals + 1.1, neuronact[survidxs[row], rposidxs[col], 1, 0, :], 'r.')
        ax.set(xlabel='Dist from electrode (mm)', ylabel='# neurons/cluster')
        # place threshold values here
        ax.set_xlim((-5.2, 5.2))
        ax.label_outer()
    # plt.show()

    # Different graph, more similar to the one I made manually in Prism
    # nrows = 3
    # fig2, axs2 = plt.subplots(nrows, nsurv, sharex=True, sharey=True)
    # for i, ax in enumerate(axs2.flat):
    #     row = np.int(i / nrows)
    #     col = int(np.mod(i, nsurv))
    #     if row == 0:
    #         titletext = 'surv = %.2f' % survVals[survidxs[col]]
    #         ax.set_title(titletext)
    #     ax.plot(posvals + 1.1, neuronact[survidxs[col], rposidxs[row], 0, 0, :], 'b')
    #     ax.plot(posvals, neuronact[survidxs[col], rposidxs[row], 0, 0, :], 'b--', linewidth=1)
    #     ax.plot(posvals + 2.2, neuronact[survidxs[col], rposidxs[row], 0, 0, :], 'b--', linewidth=1)
    #
    #     ax.plot(posvals - 1.1, neuronact[survidxs[col], rposidxs[row], 1, 0, :], 'r')
    #     ax.set(xlabel='Dist from electrode (mm)', ylabel='# neurons/cluster')
    #     ax.set_xlim((-5.2, 5.2))
    #     ax.label_outer()
    #
    # fig3, ax3 = plt.subplots(1, 1)
    # ax3.plot(posvals + 1.1, neuronact[5, 30, 0, 0, :], 'b')
    # ax3.plot(posvals + 1.1, neuronact[10, 30, 0, 0, :], 'b')
    # ax3.plot(posvals + 1.1, neuronact[15, 30, 0, 0, :], 'b')
    # ax3.plot(posvals + 1.1, neuronact[18, 30, 0, 0, :], 'b')
    # ax3.set_xlim((-3, 3))
    # ax3.set(xlabel='Dist from electrode (mm)', ylabel='# neurons/cluster')

    # fig4, ax4 = plt.subplots(1, 1)
    # ax4.plot(posvals - 1.1, neuronact[5, 30, 1, 1, :], 'r')
    # ax4.plot(posvals - 1.1, neuronact[10, 30, 1, 1, :], 'r')
    # ax4.plot(posvals - 1.1, neuronact[15, 30, 1, 1, :], 'r')
    # ax4.plot(posvals - 1.1, neuronact[18, 30, 1, 1, :], 'r')
    # ax4.set_xlim((-3, 3))
    # ax4.set(xlabel='Dist from electrode (mm)', ylabel='# neurons/cluster')
    #
    # fig5, ax5 = plt.subplots(3, 2, sharex=True, sharey=True)
    # fig5.tight_layout()
    # ax5[0, 0].plot(posvals + 1.1, neuronact[10, 24, 0, 0, :], 'b')
    # ax5[0, 0].plot(posvals + 1.1, neuronact[20, 24, 0, 0, :], 'b')
    # ax5[0, 0].plot(posvals + 1.1, neuronact[30, 24, 0, 0, :], 'b')
    # ax5[0, 0].plot(posvals + 1.1, neuronact[40, 24, 0, 0, :], 'b')
    # ax5[0, 0].set_title('Monopolar')
    # ax5[0, 0].annotate('Pos = -0.5', (-2, 60))
    # ax5[0, 1].plot(posvals - 1.1, neuronact[10, 24, 1, 1, :], 'r')
    # ax5[0, 1].plot(posvals - 1.1, neuronact[20, 24, 1, 1, :], 'r')
    # ax5[0, 1].plot(posvals - 1.1, neuronact[30, 24, 1, 1, :], 'r')
    # ax5[0, 1].plot(posvals - 1.1, neuronact[40, 24, 1, 1, :], 'r')
    # ax5[0, 1].set_title('Tripolar')
    #
    # ax5[1, 0].plot(posvals + 1.1, neuronact[10, 48, 0, 0, :], 'b')
    # ax5[1, 0].plot(posvals + 1.1, neuronact[20, 48, 0, 0, :], 'b')
    # ax5[1, 0].plot(posvals + 1.1, neuronact[30, 48, 0, 0, :], 'b')
    # ax5[1, 0].plot(posvals + 1.1, neuronact[40, 48, 0, 0, :], 'b')
    # ax5[1, 0].set_ylabel('Neurons per cluster')
    # ax5[1, 1].plot(posvals - 1.1, neuronact[10, 48, 1, 1, :], 'r')
    # ax5[1, 1].plot(posvals - 1.1, neuronact[20, 48, 1, 1, :], 'r')
    # ax5[1, 1].plot(posvals - 1.1, neuronact[30, 48, 1, 1, :], 'r')
    # ax5[1, 1].plot(posvals - 1.1, neuronact[40, 48, 1, 1, :], 'r')
    #
    # ax5[2, 0].plot(posvals + 1.1, neuronact[10, 72, 0, 0, :], 'b')
    # ax5[2, 0].plot(posvals + 1.1, neuronact[20, 72, 0, 0, :], 'b')
    # ax5[2, 0].plot(posvals + 1.1, neuronact[30, 72, 0, 0, :], 'b')
    # ax5[2, 0].plot(posvals + 1.1, neuronact[40, 72, 0, 0, :], 'b')
    # ax5[2, 0].annotate('Pos = 0.5', (-2, 60))
    # ax5[2, 0].set_xlabel('Distance from electrode (mm)')
    #
    # ax5[2, 1].plot(posvals - 1.1, neuronact[10, 72, 1, 1, :], 'r')
    # ax5[2, 1].plot(posvals - 1.1, neuronact[20, 72, 1, 1, :], 'r')
    # ax5[2, 1].plot(posvals - 1.1, neuronact[30, 72, 1, 1, :], 'r')
    # ax5[2, 1].plot(posvals - 1.1, neuronact[40, 72, 1, 1, :], 'r')
    # ax5[2, 1].set_xlabel('Distance from electrode (mm)')
    # ax5[0, 0].set_xlim((-3, 3))
    # ax5[0, 0].set_ylim((0, 80))

    plt.show()

# Save graphs? What format?

if __name__ == '__main__':
    PlotNeuronActivation()
