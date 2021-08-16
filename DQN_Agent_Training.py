from DQN_Agent_Class import DQN_Agent
import gym
import Box2D
from gym import envs
import numpy as np
import matplotlib.pyplot as plt


#############################################################
## Training environment of an DQN_Agent_Class object
## Model of the agent is saved to:              DQN_models
## Training data and plots are saved to:        DQN_data
# Exploration factor is included
#############################################################


episodes = 2
epsilon = 0.8
min_epsilon = 0.001
epsilon_decay = 0.8
scores = []
accum_episode_data = {'episode': [], 'avg': [], 'min': [], 'max': []}

show_every_episode = 10
show_every = False
env = gym.make("LunarLander-v2")

agent = DQN_Agent()
description = "MB256_LR0_001"

for game in range(episodes):
    episode_reward = 0
    actions = []
    current_state = env.reset()
    done = False
    while not done:
        if np.random.random() > epsilon:
            action = np.argmax(agent.GetQValues(current_state))
        else:
            action = np.random.randint(0, 4)
        actions.append(action)
        new_state, reward, done, info = env.step(action)

        episode_reward += reward

        if show_every and not game%show_every_episode:
            env.render()

        agent.UpdateReplayMemory((current_state, action, reward, new_state, done))
        agent.TrainModel(done)

        current_state = new_state

    # Decay epsilon
    if epsilon > min_epsilon:
        epsilon *= epsilon_decay
        epsilon = max(min_epsilon, epsilon)

    print('Action 0:{}  Action 1:{} Action 2:{}  Action 3:{}'.format(actions.count(0), actions.count(1), actions.count(2),
                                                                   actions.count(3)))

    scores.append(episode_reward)
    print(f"Game {game} score: {episode_reward}")
    if not game % show_every_episode:
        average_reward = sum(scores[-show_every_episode:])/show_every_episode
        accum_episode_data['episode'].append(game)
        accum_episode_data['avg'].append(average_reward)
        accum_episode_data['min'].append(np.min(scores[-show_every_episode:]))
        accum_episode_data['max'].append(np.max(scores[-show_every_episode:]))

        print(f"Episode: {game}, avg: {average_reward}, min: {np.min(scores[-show_every_episode:])}, max: {np.max(scores[-show_every_episode:])}")

np.save(f"DQN_data/DQN_data_{game}_{description}.npy", accum_episode_data)

plt.plot(accum_episode_data['episode'], accum_episode_data['avg'], label="average")
plt.plot(accum_episode_data['episode'], accum_episode_data['min'], label="min")
plt.plot(accum_episode_data['episode'], accum_episode_data['max'], label="max")
plt.legend()
plt.savefig(f'DQN_data/DQN_training_{game}_{description}.png')

agent.model.save(f'DQN_models/Model_{game}_{description}')