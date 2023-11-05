from collections import deque
from Functions import total_
from keras.callbacks import TensorBoard
from keras.optimizers import Adam
import random
from tensorflow.keras.callbacks import Callback
import tensorflow as tf
import keras as kr
import numpy as np
import time
import datetime

EPOCH = 10

REPLAY_MEMORY_SIZE = 100000
START_TRAIN_replay_memory = 200
MINIBATCH_SIZE = 64
DISCOUNT = 0.99
UPDATE_TARGET_EVERY = 20

class CustomScalarLogger(Callback):
    def __init__(self, log_dir):
        super().__init__()
        self.log_dir = log_dir
        self.summary_writer = tf.summary.create_file_writer(self.log_dir)
        self.step = 1

    def _write_logs(self, logs, index):
        with self.summary_writer.as_default():
            for name, value in logs.items():
                tf.summary.scalar(name, value, step=index)
                self.summary_writer.flush()

    def update_stats(self, **stats): #IMPORTANT
        self._write_logs(stats, self.step)
        self.step += 1

class DQN_Agent():

    def __init__(self, M, N, MODEL_NAME):

        self.model = self.create_model(M, N)
        self.target_model = self.create_model(M, N)
        self.target_model.set_weights(self.model.get_weights())
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)
        # self.tensorboard= CustomScalarLogger(log_dir="logs/{}-{}".format(MODEL_NAME, int(time.time())))
        log_dir = "logs/" + MODEL_NAME #+ datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.tensorboard = CustomScalarLogger(log_dir=log_dir)
        self.target_update_counter = 0
        self.M= M
        self.N= N

    def create_model(self, M, N):
        model= kr.Sequential()
        model.add(kr.layers.Flatten(input_shape=(2*M+N,2)))
        model.add(kr.layers.Dense(2*(2*M+N), activation='relu'))
        model.add(kr.layers.Dense(2*(2*M+N), activation='relu'))
        model.add(kr.layers.Dense(2*(2*M+N), activation='relu'))
        model.add(kr.layers.Dense(2*(2*M+N), activation='relu'))
        model.add(kr.layers.Dense(2*(2*M+N), activation='relu'))
        model.add(kr.layers.Dense(M*N, activation='relu'))
        model.add(kr.layers.Dense(M*N, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=0.01), metrics=['accuracy'])
        return model

    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)
    
    def get_qs(self, state):
        return self.model.predict(state.reshape(-1, *state.shape))

    def train(self): #GIVE CONDITION FOR TOTAL INVALID STATE
        if len(self.replay_memory) < START_TRAIN_replay_memory:
            return

        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)
        states_t = np.array([transition[0] for transition in minibatch])
        # states_t = states_t / 600
        qs_t = self.model.predict(states_t) # (64,24)
        qs_t = np.array([i.reshape(self.M,self.N) for i in qs_t]) #(64,8,3)

        qs_t_ = self.target_model.predict(states_t)
        qs_t_ = np.array([i.reshape(self.M,self.N) for i in qs_t_]) 

        Y=[] #learned values
        for index, (state, alloc, reward, current_state, tot) in enumerate(minibatch): #(current_state, action, reward, current_state, tot)
            q= qs_t[index]
            if tot:
                y = reward + DISCOUNT * np.array([np.max(i) for i in qs_t_[index]])
                for i,j in enumerate(alloc):
                    q[i,j] = y[i]
            else:
                y = reward
                for i,j in enumerate(alloc):
                    q[i,j] = y
            Y.append(q)

        self.model.fit(states_t, np.array([np.array(i).flatten() for i in Y]),
                       batch_size=MINIBATCH_SIZE,
                       verbose=0,
                       shuffle=False,
                       epochs=EPOCH)

        self.target_update_counter += 1
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

        