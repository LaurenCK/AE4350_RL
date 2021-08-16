import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

######################################################################################
## Generates the sensitivity analysis plots for the PG models,
## the final model training performance and the policy entropy during training
######################################################################################


fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
fig.suptitle("Sensitivity Analysis Hyper-parameters PG Method",fontsize=25)


## Learning Rate Sensitivity Analysis ##
dict_LR0_0001 = np.load("PG_data/PG_2000_PG_LR0_0001.npy", allow_pickle=True).item()
dict_LR0_001 = np.load("PG_data/PG_2000_PG_LR0_001FINAL.npy", allow_pickle=True).item()
dict_LR0_1 = np.load("PG_data/PG_2000_PG_LR0_1.npy", allow_pickle=True).item()
#dict_LR0_8 = np.load("PG_data/PG_1000_PG_LR0_8.npy", allow_pickle=True).item()

legend_elements = [Line2D([0],[0], color='r',lw=2,linestyle='-', label='LR_avg = 0.0001'),
                   Line2D([0],[0], color='g',lw=2,linestyle='-', label='LR_avg = 0.001'),
                   Line2D([0],[0], color='b',lw=2,linestyle='-', label='LR_avg = 0.1'),
                   #Line2D([0],[0], color='k',lw=2,linestyle='-', label='LR_avg = 0.8'),
                   Line2D([0], [0], color='k', lw=2,linestyle='--', label='Min.'),
                   Line2D([0], [0], color='k', linestyle='-.', label='Max.')]

ax1.set_title("Learning Rate", fontsize=18)
ax1.set_xlabel("Number of Episodes [-]", fontsize=15)
ax1.set_ylabel("Score [-]",fontsize=15)
ax1.grid()
ax1.plot(dict_LR0_0001["episode"], dict_LR0_0001["avg"], "r-", linewidth=2.5)
ax1.plot(dict_LR0_001["episode"], dict_LR0_001["avg"], "g-", linewidth=2.5)
ax1.plot(dict_LR0_1["episode"], dict_LR0_1["avg"], "b-", linewidth=2.5)
ax1.plot(dict_LR0_0001["episode"], dict_LR0_0001["min"], "r--", linewidth=2)
ax1.plot(dict_LR0_001["episode"], dict_LR0_001["min"], "g--", linewidth=2)
ax1.plot(dict_LR0_1["episode"], dict_LR0_1["min"], "b--", linewidth=2)
ax1.plot(dict_LR0_0001["episode"], dict_LR0_0001["max"], "r-.", linewidth=2)
ax1.plot(dict_LR0_001["episode"], dict_LR0_001["max"], "g-.", linewidth=2)
ax1.plot(dict_LR0_1["episode"], dict_LR0_1["max"], "b-.", linewidth=2)
ax1.legend(handles=legend_elements, loc=4, fontsize=15, framealpha=0.9)



legend_elements2 = [Line2D([0],[0], color='r',lw=2,linestyle='-', label='Discount_avg = 0.1'),
                   Line2D([0],[0], color='g',lw=2,linestyle='-', label='Discount_avg = 0.5'),
                   Line2D([0],[0], color='b',lw=2,linestyle='-', label='Discount_avg = 0.9'),
                   Line2D([0], [0], color='k', lw=2,linestyle='--', label='Min.'),
                   Line2D([0], [0], color='k', linestyle='-.', label='Max.')]


#3 Discount sensitivity ##
dict_DC_1 = np.load("PG_data/PG_2000_PG_DC0_1.npy", allow_pickle=True).item()
dict_DC_5 = np.load("PG_data/PG_2000_PG_DC0_5.npy", allow_pickle=True).item()
dict_DC_9 = np.load("PG_data/PG_2000_PG_DC0_9.npy", allow_pickle=True).item()

ax2.set_title("Discount Reduction Factor",fontsize=18)
ax2.set_xlabel("Number of Episodes [-]",fontsize=15)
ax2.set_ylabel("Score [-]",fontsize=15)
ax2.grid()
ax2.plot(dict_DC_1["episode"], dict_DC_1["avg"], "r-", label="gamma_avg = 0.1", linewidth=2.5)
ax2.plot(dict_DC_5["episode"], dict_DC_5["avg"], "g-", label="gamma_avg = 0.5", linewidth=2.5)
ax2.plot(dict_DC_9["episode"], dict_DC_9["avg"], "b-", label="gamma_avg = 0.9", linewidth=2.5)
ax2.plot(dict_DC_1["episode"], dict_DC_1["min"], "r--", label="gamma_min = 0.1", linewidth=2)
ax2.plot(dict_DC_5["episode"], dict_DC_5["min"], "g--", label="gamma_min = 0.5", linewidth=2)
ax2.plot(dict_DC_9["episode"], dict_DC_9["min"], "b--", label="gamma_min = 0.9", linewidth=2)
ax2.plot(dict_DC_1["episode"], dict_DC_1["max"], "r-.", label="gamma_max = 0.1", linewidth=2)
ax2.plot(dict_DC_5["episode"], dict_DC_5["max"], "g-.", label="gamma_max = 0.5", linewidth=2)
ax2.plot(dict_DC_9["episode"], dict_DC_9["max"], "b-.", label="gamma_max = 0.9", linewidth=2)
ax2.legend(handles=legend_elements2, loc=4, fontsize=15,framealpha=0.9)


legend_elements3 = [Line2D([0],[0], color='r',lw=2,linestyle='-', label='#Layers = 2'),
                   Line2D([0],[0], color='g',lw=2,linestyle='-', label='#Layers = 4'),
                   Line2D([0],[0], color='b',lw=2,linestyle='-', label='#Layers = 6'),
                   Line2D([0], [0], color='k', lw=2,linestyle='--', label='Min.'),
                   Line2D([0], [0], color='k', linestyle='-.', label='Max.')]


## Layers sensitivity ##
dict_layer_2 = np.load("PG_data/PG_2000_PG_2LAYERS.npy", allow_pickle=True).item()
dict_layer_4 = np.load("PG_data/PG_2000_PG_4LAYERS.npy", allow_pickle=True).item()
dict_layer_6 = np.load("PG_data/PG_2000_PG_6LAYERS.npy", allow_pickle=True).item()

ax3.set_title("Number of Layers",fontsize=18)
ax3.set_xlabel("Number of Episodes [-]",fontsize=15)
ax3.set_ylabel("Score [-]",fontsize=15)
ax3.grid()
ax3.plot(dict_layer_2["episode"], dict_layer_2["avg"], "r-", linewidth=2.5)
ax3.plot(dict_layer_4["episode"], dict_layer_4["avg"], "g-", linewidth=2.5)
ax3.plot(dict_layer_6["episode"], dict_layer_6["avg"], "b-", linewidth=2.5)
ax3.plot(dict_layer_2["episode"], dict_layer_2["min"], "r--", linewidth=2)
ax3.plot(dict_layer_4["episode"], dict_layer_4["min"], "g--", linewidth=2)
ax3.plot(dict_layer_6["episode"], dict_layer_6["min"], "b--", linewidth=2)
ax3.plot(dict_layer_2["episode"], dict_layer_2["max"], "r-.", linewidth=2)
ax3.plot(dict_layer_4["episode"], dict_layer_4["max"], "g-.", linewidth=2)
ax3.plot(dict_layer_6["episode"], dict_layer_6["max"], "b-.", linewidth=2)
ax3.legend(handles=legend_elements3, loc=4, fontsize=15, framealpha=0.9)


legend_elements4 = [Line2D([0],[0], color='r',lw=2,linestyle='-', label='#Neurons = 32'),
                   Line2D([0],[0], color='g',lw=2,linestyle='-', label='#Neurons = 64'),
                   Line2D([0],[0], color='b',lw=2,linestyle='-', label='#Neurons = 128'),
                   Line2D([0], [0], color='k', lw=2,linestyle='--', label='Min.'),
                   Line2D([0], [0], color='k', linestyle='-.', label='Max.')]

## Neurons sensitivity ##
dict_neuron_32 = np.load("PG_data/PG_2000_PG_NEURONS_32.npy", allow_pickle=True).item()
dict_neuron_64 = np.load("PG_data/PG_2000_PG_NEURONS_64.npy", allow_pickle=True).item()
dict_neuron_128 = np.load("PG_data/PG_2000_PG_NEURONS_128.npy", allow_pickle=True).item()

ax4.set_title("Number of Neurons",fontsize=18)
ax4.set_xlabel("Number of Episodes [-]",fontsize=15)
ax4.set_ylabel("Score [-]",fontsize=15)
ax4.grid()
ax4.plot(dict_neuron_32["episode"], dict_neuron_32["avg"], "r-", linewidth=2.5)
ax4.plot(dict_neuron_64["episode"], dict_neuron_64["avg"], "g-", linewidth=2.5)
ax4.plot(dict_neuron_128["episode"], dict_neuron_128["avg"], "b-", linewidth=2.5)
ax4.plot(dict_neuron_32["episode"], dict_neuron_32["min"], "r--", linewidth=2)
ax4.plot(dict_neuron_64["episode"], dict_neuron_64["min"], "g--", linewidth=2)
ax4.plot(dict_neuron_128["episode"], dict_neuron_128["min"], "b--", linewidth=2)
ax4.plot(dict_neuron_32["episode"], dict_neuron_32["max"], "r-.", linewidth=2)
ax4.plot(dict_neuron_64["episode"], dict_neuron_64["max"], "g-.", linewidth=2)
ax4.plot(dict_neuron_128["episode"], dict_neuron_128["max"], "b-.", linewidth=2)
ax4.legend(handles=legend_elements4, loc=4, fontsize=15, framealpha=0.9)


fig2, (ax_fig2) = plt.subplots(1,1)
final_model = np.load("PG_data/PG_2000_PG_FINAL_FINAL_0.001.npy", allow_pickle=True).item()
ax_fig2.plot(final_model["episode"], final_model["avg"], "r-", label="Average", linewidth=2.5)
ax_fig2.plot(final_model["episode"], final_model["min"], "b-", label="Min", linewidth=2.5)
ax_fig2.plot(final_model["episode"], final_model["max"], "g-", label="Max", linewidth=2.5)
ax_fig2.set_xlabel("Number of Episodes [-]", fontsize=20)
ax_fig2.set_ylabel("Score", fontsize=20)
ax_fig2.legend(fontsize=20, loc=4, framealpha=0.9)
ax_fig2.grid()
ax_fig2.set_title("Final Policy Gradient Model",fontsize=25)


# # Plot entropy for the training of the final model ##
entropy = np.load("PG_data/PG_2000_PG_FINAL_FINAL_ENTROPY_0.001_entropy.npy", allow_pickle=True)

# average for every n entropy samples #
n = 65141
avg_entropy = np.average(entropy.reshape(-1, n), axis=1)

fig3, (ax_fig3) = plt.subplots(1,1)
ax_fig3.grid()
ax_fig3.plot(avg_entropy, "r-", linewidth=1)
ax_fig3.set_xlabel("N x Time-frame [-]", fontsize=20)
ax_fig3.set_ylabel("Entropy [-]", fontsize=20)
ax_fig3.set_title("Entropy of the Policy Gradient Model During Training",fontsize=25)




