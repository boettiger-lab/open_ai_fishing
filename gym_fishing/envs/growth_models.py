import math
import numpy as np
from gym_fishing.envs.base_fishing_env import BaseFishingEnv

class Allen(BaseFishingEnv):
    def __init__(self,
                 r = 0.3,
                 K = 1,
                 C = 0.5,
                 sigma = 0.1,
                 init_state = 0.75, Tmax = 100, file = None):
        super().__init__(params = {"r": r, "K": K, "sigma": sigma, "C": C},
                         init_state = init_state,
                         Tmax = Tmax,
                         file = file)
        self.C = C


    def population_draw(self):
        x = self.fish_population
        mu = np.log(x) + self.r * (1 - x / self.K) * (1 - self.C) / self.K
        self.fish_population = max(0, np.random.lognormal(mu, self.sigma) )
        return self.fish_population


class BevertonHolt(BaseFishingEnv):
    def __init__(self,
                 r = 0.3,
                 K = 1,
                 sigma = 0.1,
                 init_state = 0.75, Tmax = 100, file = None):
        super().__init__(params = {"r": r, "K": K, "sigma": sigma, "C": C},
                         init_state = init_state,
                         Tmax = Tmax,
                         file = file)
        self.C = C


    def population_draw(self):
         ## mu[t] <- log(r0)  + theta * log(y[t]) - log(1 + pow(abs(y[t]), theta) / K)
         ## y[t+1] ~ dlnorm(mu[t], iQ)
        x = self.fish_population
        mu = np.log(self.r) + np.log(x) - np.log(1 + x / self.K )
        self.fish_population = max(0, np.random.lognormal(mu, self.sigma) )
        return self.fish_population

class Myers(BaseFishingEnv):
    def __init__(self,
                 r = 0.3,
                 K = 1,
                 C = 0.5,
                 sigma = 0.1,
                 init_state = 0.75, Tmax = 100, file = None):
        super().__init__(params = {"r": r, "K": K, "sigma": sigma, "C": C},
                         init_state = init_state,
                         Tmax = Tmax,
                         file = file)
        self.C = C


    def population_draw(self):
        
         ## theta == 1 ~ BevertonHolt
         ## theta > 2 tipping
        x = self.fish_population
        mu = np.log(self.r) + self.theta * np.log(x) - np.log(1 + np.power(abs(x), self.theta) / self.K )
        self.fish_population = max(0, np.random.lognormal(mu, self.sigma) )
        return self.fish_population


class May(BaseFishingEnv):
    def __init__(self,
                 r = 0.3,
                 K = 1,
                 C = 0.5,
                 sigma = 0.1,
                 init_state = 0.75, Tmax = 100, file = None):
        super().__init__(params = {"r": r, "K": K, "sigma": sigma, "C": C},
                         init_state = init_state,
                         Tmax = Tmax,
                         file = file)
        self.C = C


    def population_draw(self):
        self.fish_population = max(
          self.fish_population * np.exp(self.r *
                             (1 -self.fish_population / self.K) *
                             (self.fish_population - self.C) +
                             self.fish_population * self.sigma * np.random.normal(0,1)
                             ),
          0)
        return self.fish_population


class Ricker(BaseFishingEnv):
    def __init__(self,
                 r = 0.3,
                 K = 1,
                 C = 0.5,
                 sigma = 0.1,
                 init_state = 0.75, Tmax = 100, file = None):
        super().__init__(params = {"r": r, "K": K, "sigma": sigma, "C": C},
                         init_state = init_state,
                         Tmax = Tmax,
                         file = file)
        self.C = C


    def population_draw(self):
         # mu[t] <- log(y[t]) + r0 * (1 - y[t]/K)
         # y[t+1] ~ dlnorm(mu[t], iQ)
        self.fish_population = max(
          self.fish_population * np.exp(self.r *
                             (1 -self.fish_population / self.K) *
                             (self.fish_population - self.C) +
                             self.fish_population * self.sigma * np.random.normal(0,1)
                             ),
          0)
        return self.fish_population








