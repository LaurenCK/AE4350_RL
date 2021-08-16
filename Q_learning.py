import gym
import Box2D
from gym import envs
import numpy as np
import matplotlib.pyplot as plt
from Q_get_discrete_state_function import get_discrete_state,os_low,discrete_step_size,discrete_os_size_complete

################################################################
## Construction and updating of Q-table
## Q_table is saved to:         Q_tables
## Training data is saved to:   Q_data
# Exploration factor is commented out
################################################################


## Observations ##
#                   s[0] is the horizontal coordinate
#                   s[1] is the vertical coordinate
#                   s[2] is the horizontal speed
#                   s[3] is the vertical speed
#                   s[4] is the angle
#                   s[5] is the angular speed
#                   s[6] 1 if first leg has contact, else 0
#                   s[7] 1 if second leg has contact, else 0

description = "Lr05_DC01_step20_no_decay"

LR = 0.5
discount = 0.1
episodes = 10001
max_Q_val = 1
min_Q_val = -1

epsilon = 0.9 # probability of taking a random action (for exploring)
epsilon_decay_episode_start = (2*episodes)//5
epsilon_decay_episode_end = (3*episodes)//5 #episodes-(episodes//4)
epsilon_decay = epsilon/(epsilon_decay_episode_end-epsilon_decay_episode_start)

show_every_episode = 200
episode_rewards = []
accum_episode_data = {'episode': [], 'avg': [], 'min': [], 'max': []}

env = gym.make("LunarLander-v2")
env.reset()

# random initialization #
print("Constructing Q table...")
Q_table = np.random.uniform(low=min_Q_val, high=max_Q_val, size=(discrete_os_size_complete+[env.action_space.n]))
print("Done with constructing Q table")


for game in range(episodes):
    # if game % show_every_episode == 0:
    #     render = True
    # else:
    #     render = False

    done = False
    discrete_state = get_discrete_state(env.reset(), os_low, discrete_step_size)
    score = 0
    while not done:
        # if render:
        #     env.render()

        #if np.random.random() > epsilon:
        action = np.argmax(Q_table[discrete_state])
        #else: #random action
        #    action = np.random.randint(0,env.action_space.n)


        observation, reward, done, info = env.step(action)
        score += reward
        new_discrete_state = get_discrete_state(observation, os_low, discrete_step_size)

        if not done:
            current_Q_value = Q_table[discrete_state+(action,)]
            max_future_Q = np.max(Q_table[new_discrete_state])
            ## Updating the Q-value using the Q-function ##
            new_Q_value = (1-LR)*current_Q_value + LR*(reward+discount*max_future_Q-current_Q_value)
            Q_table[discrete_state+(action,)] = new_Q_value

        discrete_state = new_discrete_state

    #if epsilon_decay_episode_end >= game >= epsilon_decay_episode_start:
    #    epsilon -= epsilon_decay

    episode_rewards.append(score)

    if not game % show_every_episode:
        average_reward = sum(episode_rewards[-show_every_episode:])/show_every_episode
        accum_episode_data['episode'].append(game)
        accum_episode_data['avg'].append(average_reward)
        accum_episode_data['min'].append(np.min(episode_rewards[-show_every_episode:]))
        accum_episode_data['max'].append(np.max(episode_rewards[-show_every_episode:]))

        print(f"Episode: {game}, avg: {average_reward}, min: {np.min(episode_rewards[-show_every_episode:])}, max: {np.max(episode_rewards[-show_every_episode:])}")
env.close()

print("Saving Q table...")
np.save(f"Q_tables/Qtable_{game}_{description}.npy",Q_table)
print("Q table is saved.")


plt.plot(accum_episode_data['episode'], accum_episode_data['avg'], label="average")
plt.plot(accum_episode_data['episode'], accum_episode_data['min'], label="min")
plt.plot(accum_episode_data['episode'], accum_episode_data['max'], label="max")
plt.legend()
plt.savefig(f'Q_tables/training_{game}_{description}.png')

np.save(f"Q_data/Q_data_{game}_{description}.npy", accum_episode_data)