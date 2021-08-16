import tensorflow as tf
import gym
import Box2D
from gym import envs
import numpy as np
from Q_get_discrete_state_function import get_discrete_state, os_low, discrete_step_size
from DQN_Agent_Class import DQN_Agent

######################################################################
## Displays trained DQN model agent
## Displays average of played games and number of actions performed
######################################################################


env = gym.make("LunarLander-v2")

print("Loading DQN model...")
DQN_model = tf.keras.models.load_model('DQN_models/Model_1_MB256_LR0_001')
print("Loaded DQN model")

TrainedAgent = DQN_Agent()
TrainedAgent.SetModel(DQN_model)

scores = []
actions = []
for game in range(10):
    prev_obs = env.reset()
    score = 0
    done = False
    while not done:
        env.render()
        action = np.argmax(TrainedAgent.GetQValues(prev_obs))
        actions.append(action)

        observation, reward, done, info = env.step(action)
        score += reward

        prev_obs = observation

        if done:
            break
    print(f"Score of game {game}: {score}")
    scores.append(score)
env.close()

print(f"Average score of {game} games: {np.mean(scores)}")
print('Action 0:{}  Action 1:{} Action 2:{}  Action 3:{}'.format(actions.count(0),actions.count(1),actions.count(2),actions.count(3)))
