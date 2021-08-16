from tensorflow import keras
from keras.models import Model, load_model
from keras.layers import Dense, Activation, Input
from keras.optimizers import Adam
import keras.backend as K
from tensorflow.python.framework.ops import disable_eager_execution
import numpy as np

###############################################################################################################
## Class of the Policy-Gradient Method
## Constructs Agent object with following class functions:
# - GetModel: construction of the NN training model structure, loss function and output NN model structure
# - GetAction: given observation output action from NN model
# - GetProbs: output probabilities of the actions from NN model given observation
# - SaveMemory: save states, actions and rewards
# - Trainmodel: given training data update the model weights of NN model using backprop
# - SetModel: replace self.model
# - GetModel: return self.model
################################################################################################################

class PG_Agent(object):
    def __init__(self, input_size=8, n_actions=4, layer1=64, layer2 = 64, LR=0.001, gamma = 0.99, filename='trained_model'):
        self.input_size = input_size        # Number of states
        self.n_actions = n_actions          # nothing, left engine, main engine, right engine
        self.layer1_size = layer1
        self.layer2_size = layer2
        self.LR = LR                        # Learning Rate
        self.gamma = gamma                  # Discount rate
        self.file_name = filename

        self.StateMemory = []
        self.ActionMemory = []
        self.RewardMemory = []
        self.G = 0                          # Accumulated sum of future rewards including discount factor
        self.model, self.model_predict = self.GetModel()

    def GetModel(self):
        input = Input(shape=(self.input_size,))
        layer1 = Dense(self.layer1_size, activation='relu')(input)
        layer2 = Dense(self.layer2_size, activation='relu')(layer1)
        output_probs = Dense(self.n_actions, activation='softmax')(layer2)

        accum_score = Input(shape=[1])                       # sum of future rewards after taken an action
        def policy_gradient_loss_function(y_true, y_pred):   # y_pred: actions taken, y_true: probability distributions for each observation from the NN
            out = K.clip(y_pred, 1e-8,1-1e-8)                # Element-wise clip [min, max] to prevent we do not take log of zero
            log_prob = y_true*K.log(out)            # Batch of log probabilities for each action taken (k.log = element-wise log):
                                                    # it goes through the list of actions and for each action it looks at the probability distribution
                                                    # of the corresponding observation
            return -1*K.sum((log_prob*accum_score))    # negative added to invert for default gradient-descent: ! sum vs. mean !

        disable_eager_execution()

        model = Model([input, accum_score], [output_probs])
        # Compile: select best way to represent the network for training and making predictions to run on the current hardware #
        model.compile(optimizer=Adam(learning_rate=self.LR), loss=policy_gradient_loss_function)

        model_predict = Model([input], [output_probs])

        return model, model_predict

    def GetAction(self, observation):
        # Add additional dimension #
        state = observation[np.newaxis, :]                  #(N,) np.array([]) to (1,N) row vector np.array([[]])
        probs = self.model_predict.predict(state)[0]        # gives tuple
        action = np.random.choice(self.n_actions, p=probs)  # take action 0 to 3 with corresponding probabilities from the model

        return action

    def GetProbs(self,observation):
        state = observation[np.newaxis, :]
        return self.model_predict.predict(state)[0]

    def SaveMemory(self, observation, action, reward):
        self.StateMemory.append(observation)
        self.ActionMemory.append(action)
        self.RewardMemory.append(reward)

    def TrainModel(self, tensorboard, game):
        state_memory = np.array(self.StateMemory)
        reward_memory = np.array(self.RewardMemory)
        action_memory = np.array(self.ActionMemory)

        # one hot encoding:
        actions = np.zeros([len(action_memory),self.n_actions])
        actions[np.arange(len(action_memory)), action_memory] = 1

        # tensorboard.set_model(self.model)

        # calculate accumulative sum of rewards after each action with a discount!! #
        G = np.zeros_like(reward_memory)
        for time_step in range(len(reward_memory)):
            discount = 1
            G_sum = 0
            for i in range(time_step, len(reward_memory)):
                G_sum += reward_memory[i]*discount
                discount *= self.gamma
            G[time_step] = G_sum
        # normalize the rewards before inputting to NN #
        mean = np.mean(G)
        std = np.std(G) if np.std(G) > 0 else 1
        self.G = (G-mean)/std

        # tensorboard reconstruction #
        # def named_logs(model, logs):
        #     result = {}
        #     for l in zip(model.metrics_names, logs):
        #         result[l[0]] = l[1]
        #     return result

        # execute a gradient update on one particular batch of training data, performs backprop and updates model
        logs = self.model.train_on_batch([state_memory, self.G], actions)
        # tensorboard.on_epoch_end(game, named_logs(self.model, [logs]))

        # clear memory after training: not so sample efficient :( #
        self.StateMemory = []
        self.ActionMemory = []
        self.RewardMemory = []

    def SetModel(self, model):
        self.model = model

    def ReturnModel(self):
        return self.model






