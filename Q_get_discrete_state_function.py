import numpy as np

####################################################
## Define the custom state interval boundaries
## Return the discretized state given observation
####################################################


# custom boundaries #
os_low = np.array([-1,-1,-3,-3,-8,-5])    # excluding landing leg left and right
os_high = np.array([5,5,3,3,9,8])

## Q table generation ##
discrete_os_size = [20]*6
discrete_step_size = (os_high - os_low)/discrete_os_size # 6 states excluding the landing gear legs

discrete_step_size_complete = list(discrete_step_size)
discrete_step_size_complete.append(1)
discrete_step_size_complete.append(1)
discrete_step_size_complete = np.array(discrete_step_size_complete)
discrete_os_size_complete = [20,20,20,20,20,20,2,2]


def get_discrete_state(observation, os_low, discrete_step_size):
    discrete_state = (observation[0:-2]-os_low)/discrete_step_size
    discrete_observation = list(discrete_state)
    discrete_observation.append(observation[-2])
    discrete_observation.append(observation[-1])
    discrete_observation = np.array(discrete_observation)
    return tuple(discrete_observation.astype(np.int))