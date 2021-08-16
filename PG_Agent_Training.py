from PG_Agent_Class import PG_Agent
import numpy as np
import matplotlib.pyplot as plt
import gym
import Box2D
from gym import envs
import time
import keras

#############################################################
## Training environment of an PG_Agent class object
## Model of the agent is saved to:   PG_models
## Training data is saved to:        PG_data
## Training plots are saved to:      PG_plots
# Exploration factor is commented out
#############################################################


tensorboard = keras.callbacks.TensorBoard(log_dir=f'logs/Model_{int(time.time())}')

agent = PG_Agent()

episodes = 2001
show_every_episode = 100
accum_episode_data = {'episode': [], 'avg': [], 'min': [], 'max': []}

# epsilon = 0.8
# epsilon_decay_episode_start = 1
# epsilon_decay_episode_end = (episodes)//2 #episodes-(episodes//4)
# epsilon_decay = epsilon/(epsilon_decay_episode_end-epsilon_decay_episode_start)

env = gym.make("LunarLander-v2")
env.reset()

scores_per_game = []
description = "PG_FINAL_ENTROPY_0.001"
entropy = []


for game in range(episodes):
    done = False
    score = 0
    prev_observation = env.reset()
    while not done:

        # if np.random.random() > epsilon:
        action = agent.GetAction(prev_observation)

        # else:
        #     action = np.random.randint(0, env.action_space.n)

        observation, reward, done, info = env.step(action)
        agent.SaveMemory(prev_observation, action, reward)

        # Calculate policy entropy #
        probs = np.array(agent.GetProbs(prev_observation))
        entropy.append(-np.sum(probs*np.log(probs)))

        prev_observation = observation
        #actions_taken.append(action)
        score += reward

    # Game done
    agent.TrainModel(tensorboard, game)

    # if epsilon_decay_episode_end >= game >= epsilon_decay_episode_start:
    #    epsilon -= epsilon_decay

    scores_per_game.append(score)
    if not game % show_every_episode:
        average_reward = sum(scores_per_game[-show_every_episode:])/show_every_episode
        accum_episode_data['episode'].append(game)
        accum_episode_data['avg'].append(average_reward)
        accum_episode_data['min'].append(np.min(scores_per_game[-show_every_episode:]))
        accum_episode_data['max'].append(np.max(scores_per_game[-show_every_episode:]))

        print(f"Episode: {game}, avg: {average_reward}, min: {np.min(scores_per_game[-show_every_episode:])}, max: {np.max(scores_per_game[-show_every_episode:])}")

print("Saving model...")
agent.model.save_weights(f'PG_models/PG_Model_Weights_{game}_{description}')
print("Model saved")
np.save(f"PG_data/PG_{game}_{description}.npy", accum_episode_data)
np.save(f"PG_data/PG_{game}_{description}_entropy.npy", entropy)

print("------Number of trained games---------: ", len(scores_per_game))
plt.plot(accum_episode_data['episode'], accum_episode_data['avg'], label="average")
plt.plot(accum_episode_data['episode'], accum_episode_data['min'], label="min")
plt.plot(accum_episode_data['episode'], accum_episode_data['max'], label="max")
plt.legend()
plt.savefig(f"PG_plots/PG_training_{game}_{description}.png")




