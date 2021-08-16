import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

#######################################################
## Plot the actions taken given the states for: (data imported from Behavioural_data)
#   - final PG model
#   - final QL model
## States plotted:
#   - horizontal velocity vs vertical velcotiy
#   - angle vs angular rate
#######################################################

# PG Method #
action_PG = np.load("Behavioural_data/PG_actions.npy")
states_PG = np.load("Behavioural_data/PG_states.npy")

PG_hor = []
PG_vert = []
PG_hor_vel = []
PG_vert_vel = []
PG_angle = []
PG_angular_rate = []

markers = ["Do Nothing", "Fire Right Engine", "Fire Main Engine", "Fire Left Engine"]
marker_lst = []

for state in range(len(states_PG)):
    PG_hor.append(states_PG[state][0])
    PG_vert.append(states_PG[state][1])
    PG_hor_vel.append(states_PG[state][2])
    PG_vert_vel.append(states_PG[state][3])
    PG_angle.append(states_PG[state][4])
    PG_angular_rate.append(states_PG[state][5])

    marker_lst.append(markers[action_PG[state]])

df_PG = pd.DataFrame(list(zip(PG_hor,PG_vert,PG_hor_vel,PG_vert_vel,PG_angle,PG_angular_rate)),columns=["Horizontal Position", "Vertical Position", "Horizontal Velocity", "Vertical Velocity", "Angle", "Angular Rate"])
df_PG["actions"] = action_PG
df_PG["Action Labels"] = marker_lst

fig1, (ax2, ax3) = plt.subplots(1,2)
fig1.suptitle("Behavioural Analysis for Policy Gradient Method",fontsize=25)
#ax1.grid()
ax2.grid()
ax3.grid()

# ax1.set_title("Horizontal versus Vertical Position")
# sns.scatterplot(ax=ax1, data=df_PG, x="Horizontal Position", y="Vertical Position", hue="Action Labels", alpha=1)

ax2.set_title("Horizontal Velocity versus Vertical Velocity",fontsize=20)
sns.scatterplot(ax=ax2, data=df_PG, x="Horizontal Velocity", y="Vertical Velocity", hue="Action Labels",alpha=0.7)

ax3.set_title("Angle versus Angular Rate",fontsize=20)
sns.scatterplot(ax=ax3, data=df_PG, x="Angle", y="Angular Rate", hue="Action Labels",alpha=0.7)


# QL  #
action_QL = np.load("Behavioural_data/QL_actions.npy")
states_QL = np.load("Behavioural_data/QL_states.npy")

QL_hor = []
QL_vert = []
QL_hor_vel = []
QL_vert_vel = []
QL_angle = []
QL_angular_rate = []

marker_lst_QL = []

for item in range(len(states_QL)):
    QL_hor.append(states_QL[item][0])
    QL_vert.append(states_QL[item][1])
    QL_hor_vel.append(states_QL[item][2])
    QL_vert_vel.append(states_QL[item][3])
    QL_angle.append(states_QL[item][4])
    QL_angular_rate.append(states_QL[item][5])

    marker_lst_QL.append(markers[action_QL[item]])

df_QL = pd.DataFrame(list(zip(QL_hor,QL_vert,QL_hor_vel,QL_vert_vel,QL_angle,QL_angular_rate)),columns=["Horizontal Position", "Vertical Position", "Horizontal Velocity", "Vertical Velocity", "Angle", "Angular Rate"])
df_QL["actions"] = action_QL
df_QL["Action Labels"] = marker_lst_QL

fig2, (ax12, ax22, ax32) = plt.subplots(1,3)
fig2.suptitle("Behavioural Analysis for Q-Learning",fontsize=25)
ax12.grid()
ax22.grid()
ax32.grid()

ax12.set_title("Horizontal versus Vertical Position",fontsize=25)
sns.scatterplot(ax=ax12, data=df_QL, x="Horizontal Position", y="Vertical Position", hue="Action Labels", alpha=1)

ax22.set_title("Horizontal Velocity versus Vertical Velocity",fontsize=20)
sns.scatterplot(ax=ax22, data=df_QL, x="Horizontal Velocity", y="Vertical Velocity", hue="Action Labels",alpha=0.7)

ax32.set_title("Angle versus Angular Rate",fontsize=20)
sns.scatterplot(ax=ax32, data=df_QL, x="Angle", y="Angular Rate", hue="Action Labels",alpha=0.7)

## Seperate plot angle vs angular rate for PG model ##
# fig3, (ax33) = plt.subplots(1,1)
# ax33.set_title("Angle versus Angular Rate for PG Model 1", fontsize=20)
# ax33.grid()
# sns.scatterplot(ax=ax33, data=df_PG, x="Angle", y="Angular Rate", hue="Action Labels",alpha=0.7)
