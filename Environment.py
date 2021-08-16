import numpy as np
from PG_Agent_Class import PG_Agent
import gym
import Box2D
from gym import envs

####################################################
## Environment only
## Agent just takes random actions
## Plot avg, min., max. and variance of run games
####################################################

env = gym.make("LunarLander-v2")

scores_per_game = []

for game in range(100):
    score = 0
    prev_obs = env.reset()
    done = False
    print(f"Game {game}")
    while not done:
        #env.render()

        action = env.action_space.sample()

        observation, reward, done, info = env.step(action)
        prev_obs = observation
        score += reward

        #if -1e-3 < observation[3] < 1e-3:
        #    done = True

        if done:
            break

    env.close()
    scores_per_game.append(score)
    print(f"Score: {score}")

average = np.mean(scores_per_game)
print("Average score of played games: ", average)
print("Min. score of played games: ", np.min(scores_per_game))
print("Max. score of played games: ", np.max(scores_per_game))

var = (np.sum((np.array(scores_per_game)-average)**2))/(len(scores_per_game)-1)
print(f"Variance of the score of all games: {var}")