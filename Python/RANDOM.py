import numpy as np
from Env import _5G
import tensorflow as tf
import keras as kr
from collections import deque
from DQN_AGENT import CustomScalarLogger
from tqdm import tqdm
import time
import datetime

N= 5
M= 14


EPISODES = 2000
NUM_OF_ITER = 10


Smoothing_memory = 100


env = _5G(M,N)
log_dir="logs/RANDOM" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard = CustomScalarLogger(log_dir)

total_memory= deque(maxlen=Smoothing_memory)
for episode in tqdm(range(1, EPISODES + 1), ascii=False, unit='episodes'):

    current_state_ = env.reset()
    current_state = current_state_['DRUE'][np.argsort(current_state_['DRUE'][:, 0]), :]/20
    current_state= np.concatenate((current_state, current_state_['DTUE'][np.argsort(current_state_['DTUE'][:, 0]), :]))/600
    current_state= np.concatenate((current_state, current_state_['CUE'][np.argsort(current_state_['CUE'][:, 0]), :]))/600

    for _ in range(NUM_OF_ITER):
        

        action = np.random.randint(0, N, size=(M,))
        counter = 0
        while not env.step(action)[1] and counter<50:
            action = np.random.randint(0, N, size=(M,))
            counter += 1

        #ARRANGE FOR SPECIFIC SELECTION
        done, tot, std = env.step(action)

        if tot:
            total_memory.append(tot)
            min_total = min(total_memory)
            max_total = max(total_memory)
            avg_total = np.mean(total_memory)
            tensorboard.update_stats(total_avg=avg_total,
                                        total_max=max_total,
                                        total_min=min_total)