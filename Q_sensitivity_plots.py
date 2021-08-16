import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

#########################################################################################################
## Generates the sensitivity analysis plots for the QL models and the final model training performance ##
#########################################################################################################

## Learning Rate Sensitivity Analysis ##
dict_0_1 = np.load("Q_data/Q_data_10000_LR-01.npy", allow_pickle=True).item()
dict_0_5 = np.load("Q_data/Q_data_10000_LR-05.npy",allow_pickle=True).item()
dict_0_9 = np.load("Q_data/Q_data_10000_LR-09.npy",allow_pickle=True).item()

legend_elements = [Line2D([0],[0], color='r',lw=2,linestyle='-', label='LR_avg = 0.1'),
                   Line2D([0],[0], color='g',lw=2,linestyle='-', label='LR_avg = 0.5'),
                   Line2D([0],[0], color='b',lw=2,linestyle='-', label='LR_avg = 0.9'),
                   Line2D([0], [0], color='k', lw=2,linestyle='--', label='Min.'),
                   Line2D([0], [0], color='k', linestyle='-.', label='Max.')]

fig, (ax1, ax2,ax3) = plt.subplots(1,3)
fig.suptitle("Sensitivity Analysis Hyper-parameters Q-Learning",fontsize=25)

ax1.set_title("Learning Rate", fontsize=18)
ax1.set_xlabel("Number of Episodes [-]",fontsize=15)
ax1.set_ylabel("Score [-]",fontsize=15)
ax1.grid()
ax1.plot(dict_0_1["episode"], dict_0_1["avg"], "r-", linewidth=2.5)
ax1.plot(dict_0_5["episode"], dict_0_5["avg"], "g-", linewidth=2.5)
ax1.plot(dict_0_9["episode"], dict_0_9["avg"], "b-", linewidth=2.5)
ax1.plot(dict_0_1["episode"], dict_0_1["min"], "r--", linewidth=2)
ax1.plot(dict_0_5["episode"], dict_0_5["min"], "g--", linewidth=2)
ax1.plot(dict_0_9["episode"], dict_0_9["min"], "b--", linewidth=2)
ax1.plot(dict_0_1["episode"], dict_0_1["max"], "r-.", linewidth=2)
ax1.plot(dict_0_5["episode"], dict_0_5["max"], "g-.", linewidth=2)
ax1.plot(dict_0_9["episode"], dict_0_9["max"], "b-.", linewidth=2)
ax1.legend(handles=legend_elements, loc=8, fontsize=15, framealpha=0.9)



legend_elements2 = [Line2D([0],[0], color='r',lw=2,linestyle='-', label='gamma_avg = 0.1'),
                   Line2D([0],[0], color='g',lw=2,linestyle='-', label='gamma_avg = 0.5'),
                   Line2D([0],[0], color='b',lw=2,linestyle='-', label='gamma_avg = 0.9'),
                   Line2D([0], [0], color='k', lw=2,linestyle='--', label='Min.'),
                   Line2D([0], [0], color='k', linestyle='-.', label='Max.')]


#3 Discount sensitivity ##
dict_DC_1 = np.load("Q_data/Q_data_10000_DC-01.npy", allow_pickle=True).item()
dict_DC_5 = np.load("Q_data/Q_data_10000_DC-05.npy",allow_pickle=True).item()
dict_DC_9 = np.load("Q_data/Q_data_10000_DC-09.npy",allow_pickle=True).item()

ax2.set_title("Discount Factor",fontsize=18)
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
ax2.legend(handles=legend_elements2, loc=8, fontsize=15,framealpha=0.9)


legend_elements3 = [Line2D([0],[0], color='r',lw=2,linestyle='-', label='N_step_avg = 8'),
                   Line2D([0],[0], color='g',lw=2,linestyle='-', label='N_step_avg = 14'),
                   Line2D([0],[0], color='b',lw=2,linestyle='-', label='N_step_avg = 25'),
                   Line2D([0], [0], color='k', lw=2,linestyle='--', label='Min.'),
                   Line2D([0], [0], color='k', linestyle='-.', label='Max.')]


## Step sensitivity ##
dict_step_8 = np.load("Q_data/Q_data_10000_STEPS-8.npy", allow_pickle=True).item()
dict_step_14 = np.load("Q_data/Q_data_10000_STEPS-14.npy",allow_pickle=True).item()
dict_step_24 = np.load("Q_data/Q_data_10000_STEPS-24.npy",allow_pickle=True).item()

ax3.set_title("Number of Discretization Steps",fontsize=18)
ax3.set_xlabel("Number of Episodes [-]",fontsize=15)
ax3.set_ylabel("Score [-]",fontsize=15)
ax3.grid()
ax3.plot(dict_step_8["episode"], dict_step_8["avg"], "r-", label="Step_avg = 8", linewidth=2.5)
ax3.plot(dict_step_14["episode"], dict_step_14["avg"], "g-", label="Step_avg = 14", linewidth=2.5)
ax3.plot(dict_step_24["episode"], dict_step_24["avg"], "b-", label="Step_avg = 24", linewidth=2.5)
ax3.plot(dict_step_8["episode"], dict_step_8["min"], "r--", label="Step_min = 8", linewidth=2)
ax3.plot(dict_step_14["episode"], dict_step_14["min"], "g--", label="Step_min = 14", linewidth=2)
ax3.plot(dict_step_24["episode"], dict_step_24["min"], "b--", label="Step_min = 24", linewidth=2)
ax3.plot(dict_step_8["episode"], dict_step_8["max"], "r-.", label="Step_max = 8", linewidth=2)
ax3.plot(dict_step_14["episode"], dict_step_14["max"], "g-.", label="Step_max = 14", linewidth=2)
ax3.plot(dict_step_24["episode"], dict_step_24["max"], "b-.", label="Step_max = 24", linewidth=2)
ax3.legend(handles=legend_elements3, loc=8, fontsize=15, framealpha=0.9)


## Final model learning performance ##
fig2, (ax_fig2) = plt.subplots(1,1)
final_model = np.load("Q_data/Q_data_10000_Lr05_DC01_step20_no_decay.npy", allow_pickle=True).item()
ax_fig2.plot(final_model["episode"], final_model["avg"], "r-", label="Average", linewidth=2.5)
ax_fig2.plot(final_model["episode"], final_model["min"], "b-", label="Min", linewidth=2.5)
ax_fig2.plot(final_model["episode"], final_model["max"], "g-", label="Max", linewidth=2.5)
ax_fig2.set_xlabel("Number of Episodes [-]", fontsize=20)
ax_fig2.set_ylabel("Score", fontsize=20)
ax_fig2.legend(fontsize=20, loc=3, framealpha=0.9)
ax_fig2.grid()
ax_fig2.set_title("Final Q-Learning Model",fontsize=25)




