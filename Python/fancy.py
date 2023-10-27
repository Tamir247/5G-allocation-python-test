import numpy as np
from Env import _5G
import tensorflow as tf
import keras as kr
from keras.optimizers import Adam
from collections import deque
from keras.callbacks import TensorBoard
import random

REPLAY_MEMORY_SIZE = 10000
MINIBATCH_SIZE = 64
START_TRAIN_replay_memory = 1000
DISCOUNT = 0.99
UPDATE_TARGET_EVERY = 5
MODEL_NAME = '_5G'

# Own Tensorboard class
class ModifiedTensorBoard(TensorBoard):

    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.FileWriter(self.log_dir)

    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        pass

    # Overrided, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Overrided
    # We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    # Overrided, so won't close writer
    def on_train_end(self, _):
        pass

    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        self._write_logs(stats, self.step)

env = _5G(8,3)
state = env.reset()
count= 0
total_memory= []
for _ in range(10):
    out= np.random.uniform(size=(8,3))
    alloc= np.array([np.argmax( row ) for row in out])
    reward, done, total= env.step(alloc)
    if total:
        count+= 1
        total_memory.append(total)
    if total and count>=2:
        reward= total_memory[count-1] - total_memory[count-2]
    print(reward)

alloc = np.array([1,0,0,0,1,2,0,2])
memory= []
for _ in range(1000):
    env.change_DTUE()
    reward, done, total= env.step(alloc)
    memory.append(total)
# print(np.sqrt(np.array(memory).var()))
a = np.array([1,2,3])
a.var()
env.render()

class DQN_Agent():
    def __init__(self):
        self.model = self.create_model()
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())
    
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)
        self.tensorboard = ModifiedTensorBoard(log_dir="logs/{}-{}".format(MODEL_NAME, int(time.time())))
        self.target_update_counter = 0

    def create_model():
        model= kr.Sequential()
        model.add(kr.layers.Flatten(input_shape=(19,)))
        model.add(kr.layers.Dense(32, activation='relu'))
        model.add(kr.layers.Dense(32, activation='relu'))
        model.add(kr.layers.Dense(32, activation='relu'))
        model.add(kr.layers.Dense(8,), activation='sigmoid')
        model.compile(loss='mse', optimizer=Adam(learning_rate=0.01), metrics=['accuracy'])
        return model

    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)
    
    def get_qs(self, state):
        return self.model.predict(state.reshape(-1, *state.shape))
    
    def train(self): #DEFINE TRANSITIONS!!!
        if len(self.replay_memory) < START_TRAIN_replay_memory:
            return
        
        ######
        #
        #   WRITE DATA STRUCTURES HERE
        #
        ######

        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)
        
        states_t = np.array([transition[0] for transition in minibatch])
        qs_t = self.model.predict(states_t)
        # states_t_ = np.array([transition[3] for transition in minibatch])
        qs_t_ = self.target_model.predict(states_t)
        Y=[]
        for index, (state, alloc, reward, total) in enumerate(minibatch):
            y = reward + DISCOUNT * np.array([np.max(i) for i in qs_t_[index]])
            q= qs_t[index]
            for i,j in enumerate(alloc):
                q[i,j] = y[i,j]
            Y.append(q)
        self.model.fit(states_t, np.array(Y), 
                       batch_size=MINIBATCH_SIZE, 
                       verbose=0,
                       shuffle=False,
                       callbacks=[self.tensorboard])
        self.target_update_counter += 1
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0