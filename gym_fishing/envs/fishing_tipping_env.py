import math
import numpy as np
from gym_fishing.envs.base_fishing_env import BaseFishingEnv

class FishingTippingEnv(BaseFishingEnv):
    def __init__(self,
                 r = 0.3, 
                 K = 1, 
                 C = 0.5,
                 sigma = 0.1,
                 init_state = 0.75,
                 Tmax = 100,
                 file = None):
        super().__init__(params = {"r": r, "K": K, "sigma": sigma, "C": C},
                         init_state = init_state,
                         Tmax = Tmax,
                         file = file)
        self.C = C

    
    def population_draw(self):
        self.fish_population = np.maximum(
          self.fish_population * np.exp(self.r *
                             (1 -self.fish_population / self.K) *
                             (self.fish_population - self.C) +
                             self.fish_population * self.sigma * np.random.normal(0,1)
                             ), 
          0)
        return self.fish_population



      
 
