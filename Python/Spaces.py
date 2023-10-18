import numpy as np
from gym import spaces

class DiscBoxSpace(spaces.Box):
    def __init__(self, radius, length, dtype_=np.float32):
        low = -radius * np.ones(2)
        high = radius * np.ones(2)
        super(DiscBoxSpace, self).__init__(low=low, high=high, dtype= dtype_)
        self.radius = radius
        self.length = length


    def sample(self):
        arr = []
        for _ in range(self.length):
            angle = np.random.uniform(0, 2 * np.pi)
            distance = np.sqrt(np.random.uniform(0, 1))*self.radius
            arr.append([distance * np.cos(angle), distance * np.sin(angle)])
        return np.array(arr)

    def contains(self, x):
        distance = np.linalg.norm(x)
        return distance <= self.radius


class n_Discrete(spaces.Discrete):
    def __init__(self, max, number):
        super(n_Discrete, self).__init__(max)
        self.number = number
        self.max = max
    
    def sample(self):
        arr = []
        for _ in range(self.number):
            arr.append(np.random.randint(self.max))
        return np.array(arr)