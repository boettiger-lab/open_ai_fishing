
import math
import numpy as np
import gym
import gym_fishing.envs.base_fishing_env
from gym_fishing.envs.base_fishing_env import BaseFishingEnv


class FishingModelError(BaseFishingEnv):
    metadata = {'render.modes': ['human']}

    def __init__(self,
                 K_mean = 1.0,
                 r_mean = 0.3,
                 price = 1.0,
                 sigma = 0.02,
                 sigma_p = 0.1,
                 init_state = 0.75,
                 Tmax = 100,
                 file = None):
                   
        super().__init__(params = {"r": r_mean,
                                   "K": K_mean,
                                   "sigma": sigma,
                                   "r_mean": r_mean,
                                   "K_mean": K_mean,
                                   "sigma_p": sigma_p},
                         init_state = init_state,
                         Tmax = Tmax,
                         file = file)
        ## parameters
        self.K_mean = K_mean
        self.r_mean = r_mean
        self.K = np.clip(np.random.normal(K_mean, sigma_p), 0, 1e6)
        self.r = np.clip(np.random.normal(r_mean, sigma_p), 0, 1e6)
        self.sigma_p = sigma_p

    def reset(self):
        self.K = np.clip(np.random.normal(self.K_mean, self.sigma_p), 0, 1e6)
        self.r = np.clip(np.random.normal(self.r_mean, self.sigma_p), 0, 1e6)
        self.state = np.array([self.init_state])
        self.fish_population = self.init_state
        self.harvest = 0
        self.years_passed = 0
        return self.state
  



      
 
