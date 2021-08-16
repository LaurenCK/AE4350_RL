import gym
import Box2D
from gym import envs
import numpy as np
from Q_get_discrete_state_function import get_discrete_state, os_low, discrete_step_size

################################################################
## Displays trained QL model agent
## Displays average, min., max. and variance of played games
## Optional: save data for behavioural analysis
################################################################


env = gym.make("LunarLander-v2")
save_run = False

print("Loading Q table...")
Q_table = np.load('Q_tables/Qtable_FINAL_10000_Lr05_DC01_step20_no_decay.npy')
print("Loaded Q table")

scores = []
actions = []
states = []
for game in range(40):
    discrete_state = get_discrete_state(env.reset(), os_low, discrete_step_size)
    score = 0
    done = False
    while not done:
        env.render()
        action = np.argmax(Q_table[discrete_state])
        actions.append(action)
        states.append(discrete_state)

        observation, reward, done, info = env.step(action)
        score += reward
        new_discrete_state = get_discrete_state(observation, os_low,discrete_step_size)

        discrete_state = new_discrete_state

        if -1e-5 < observation[3] < 1e-5:
            done = True
        if done:
            break
    print(f"Score of game {game}: {score}")
    scores.append(score)
env.close()

if save_run:
    np.save("Behavioural_data/QL_states.npy", states)
    np.save("Behavioural_data/QL_actions.npy", actions)

average = np.mean(scores)
print(f"Average score of {game} games: {average}")
print(f"Min. score of {game} games: {np.min(scores)}")
print(f"Max. score of {game} games: {np.max(scores)}")
print('Action 0:{}  Action 1:{} Action 2:{}  Action 3:{}'.format(actions.count(0),actions.count(1),actions.count(2),actions.count(3)))

## Variance of score from the games ##
var = (np.sum((np.array(scores)-average)**2)/(len(scores)-1))
print(f"Variance of the score of all games: {var}")