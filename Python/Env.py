from Functions import *
from GlobalVariables import *
from Spaces import *
import matplotlib.pyplot as plt
import numpy as np
from gym import Env, spaces

class _5G(Env):
    def __init__(self, M, N, render_mode = None):
        super(_5G, self).__init__()
        self.observation_space = spaces.Dict({'DRUE': DiscBoxSpace(20,M),
                                              'DTUE': DiscBoxSpace(600,M),
                                              'CUE': DiscBoxSpace(600,N),
                                              })
        self.action_space = n_Discrete(N, M)
        self.M = M     #M
        self.N = N     #N

    def reset(self):
        self.state = self.observation_space.sample()
        self.details = details(self.N, self.M, self.state['CUE'], self.state['DRUE'], self.state['DTUE'])
        self.done = False
        self.repeat = 50
        self.alloc = None
        return self.state

    def Total(self, action):
        try:
            self.total = total_(action, *self.details)
            return self.total
        except:
            self.total = None
            return None

    def step(self, action):
        self.alloc = action
        self.repeat -= 1
        self.total = self.Total(package_(self.alloc, self.N))
        #REWARD
        if self.total ==None:
            reward = -1
        else:
            reward = 1

        if(self.repeat > 0):
            self.done = False
        else:
            self.done = True

        return reward

    def render(self):
        _, ax = plt.subplots()
        plt.scatter(self.state['CUE'].T[0], self.state['CUE'].T[1], c='r')
        plt.scatter(self.state['DRUE'].T[0] + self.state['DTUE'].T[0],self.state['DRUE'].T[1] + self.state['DTUE'].T[1], c='cyan')
        plt.scatter(self.state['DTUE'].T[0] , self.state['DTUE'].T[1], c='blue')
        ax.quiver(self.state['DTUE'].T[0], self.state['DTUE'].T[1], self.state['CUE'][self.alloc].T[0] - self.state['DTUE'].T[0],  self.state['CUE'][self.alloc].T[1] - self.state['DTUE'].T[1], angles='xy', scale_units='xy', scale=1, color='red', width=0.001)
        ax.set_xlim(-600, 600)
        ax.set_ylim(-600, 600)
        plt.show()
