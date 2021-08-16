import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adam
import gym
import Box2D
from gym import envs
from collections import deque
import numpy as np
import random
from keras.callbacks import TensorBoard
import time

##########################################################################
## Class of the Deep Q-Learning method
## Constructs Agent object with following class functions:
# - GetModel: return model structure
# - SetModel: replace self.model
# - UpdateReplayMemory: append observations
# - GetQValues: return the Q-values for a given state from the model
# - TrainModel: update model weight using Q-function and backprop
#########################################################################

env = gym.make("LunarLander-v2")
replay_memory_size = 3000 # ~ 150 frames per game
mini_batch_size = 256
min_replay_mem_size = 300
discount = 0.95
N_games_update = 5
LR = 0.001

class DQN_Agent:
    def __init__(self):
        # main model: want to update .fit this model every step #
        self.model = self.GetModel()

        # target model: use .predict for every step
        self.target_model = self.GetModel()
        self.target_model.set_weights(self.model.get_weights())
        self.target_update_counter = 0

        # we want to have a batch to train the NN else NN overfits on every one sample #
        # - improves stability
        # - smoothing out predictions (prediction model not updated every step but every N episodes) #
        self.replay_memory = deque(maxlen=replay_memory_size)

    def GetModel(self):
        model = Sequential()
        model.add(Dense(64, input_dim=8, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(4, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=LR), metrics=['accuracy'])
        return model

    def SetModel(self, model):
        self.model = model

    def UpdateReplayMemory(self, transition):
        self.replay_memory.append(transition)

    def GetQValues(self, state):
        state = state[np.newaxis, :]
        return self.model.predict(state)[0] # add normalizing [0,1]?

    def TrainModel(self, terminal_state):
        if len(self.replay_memory) < min_replay_mem_size:
            return

        minibatch = random.sample(self.replay_memory, mini_batch_size)
        current_states = np.array([transition[0] for transition in minibatch]) # add normalizing [0,1]
        current_Qvals_list = self.model.predict(current_states)

        new_current_states = np.array([transition[3] for transition in minibatch]) # add normalizing [0,1]
        future_Qvals_list = self.target_model.predict(new_current_states)

        X = []
        y = []

        for index, (current_state, action, reward, new_current_states, done) in enumerate(minibatch):
            if not done:
                max_future_Q = np.max(future_Qvals_list[index])
                new_Q = reward + discount * max_future_Q
            else: # game over, no future step available
                new_Q = reward

            current_Q = current_Qvals_list[index]
            current_Q[action] = new_Q

            X.append(current_state)
            y.append(current_Q)

        # Converting to array #
        for item in range(len(X)):
            X[item] = np.array(X[item])
        for item in range(len(y)):
            y[item] = np.array(y[item])
        X = np.array(X)
        y = np.array(y)

        self.model.fit(X, y, batch_size= mini_batch_size, verbose=0, shuffle=False, epochs=1) #normalize X!!!
        # verbose: show progress bar or not #

        # if we want to update target model #
        if terminal_state == True: # game over
            self.target_update_counter += 1
            print("Game over.")

        if self.target_update_counter >= N_games_update:
            # copy over de weight from initial model #
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0
            print("Updated target model")