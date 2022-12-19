import gym
import numpy as np
from gym import spaces

class testContEnv(gym.Env):
    def __init__(self, file=None):
        
        self.TMAX = 10
        
        self.action_space = spaces.Box(
            np.array([0], dtype=np.float32),
            np.array([1], dtype=np.float32),
            dtype = np.float32
        )
        
        self.observation_space = spaces.Box(
            np.array([-1, -1, -1], dtype=np.float32),
            np.array([1, 1, 1], dtype=np.float32),
            dtype=np.float32,
        )
        
        self.init_state = np.array([0, 0, 0], dtype=np.float32)
        self.state = self.init_state
        self.reward = 0
        self.time = 0
    
    def step(self, action):
        done = False
        self.time += 1
        if self.action[0] > 0.5:
            self.reward += 1/self.action[0]
        if self.time >= 10:
            done = True
        return self.state, self.reward, done, {}
    
    def reset(self):
        self.state = self.init_state
        return self.state
    
    