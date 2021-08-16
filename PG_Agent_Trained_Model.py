from tensorflow import keras
import tensorflow as tf
import numpy as np
from PG_Agent_Class import PG_Agent
import gym
import Box2D
from gym import envs

################################################################
## Displays trained PG model agent
## Displays average, min., max. and variance of played games
## Optional: save data for behavioural analysis
################################################################

env = gym.make("LunarLander-v2")
env.reset()

save_run = False # data used for the behavioural analysis

trained_agent = PG_Agent()
trained_agent.ReturnModel().load_weights('PG_models/PG_Model_Weights2000') #PG_models/PG_Final_Lr_0_001
# PG_models/PG_Model_Weights2000

scores_per_game = []
actions = []
states = []

for game in range(20):
    score = 0
    prev_obs = env.reset()
    done = False
    #print(f"Game {game}")
    while not done:
        env.render()

        action = trained_agent.GetAction(prev_obs)

        observation, reward, done, info = env.step(action)
        states.append(prev_obs)
        actions.append(action)

        prev_obs = observation
        score += reward

        if -1e-5 < observation[3] < 1e-5:
            done = True

        if done:
            break

    env.close()
    scores_per_game.append(score)
    print(f"Score: {score}")


if save_run:
    np.save("Behavioural_data/PG_states.npy", states)
    np.save("Behavioural_data/PG_actions.npy", actions)

average = np.mean(scores_per_game)
print("Average score of played games: ", average)
print("Min. score of played games: ", np.min(scores_per_game))
print("Max. score of played games: ", np.max(scores_per_game))
print('Action 0:{}  Action 1:{} Action 2:{}  Action 3:{}'.format(actions.count(0),actions.count(1),actions.count(2),actions.count(3)))


## Calculate the variance of the result ##
var = (np.sum((np.array(scores_per_game)-average)**2))/(len(scores_per_game)-1)
print(f"Variance of the score of all games: {var}")

