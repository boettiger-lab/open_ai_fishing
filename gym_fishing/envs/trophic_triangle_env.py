import gym
import numpy as np
from gym import spaces

from gym_fishing.envs.shared_env import (
    csv_entry,
    estimate_policyfn,
    plot_mdp,
    plot_policyfn,
    simulate_mdp,
)

# Based on the "food chain triangle" of the 3 fish populations
# in the 5D model of https://doi.org/10.1007/s00285-019-01358-z

class trophicTriangleEnv(gym.Env):
    metadata = {"render.modes": ["human"]}
    
    def __init__(
        self, 
        params = { # taken from Table 1 in reference
            "s": 0.6, # Survival rate of juveniles to maturation
            "mA": 0.4, # Mortality rate of adults
            "f": 2, # Fecundity of adult bass
            "cFA": 0.3, # Predation of planktivores by adult bass
            "cJA": 0.1, # Predation of juvenile bass on by adult bass
            "cJF": 0.5, # Predation of juvenile bass by planktivorous fish
            "Fo": 200, # Abundance of planktivorous fish in non-foraging arena
            "DF": 0.09, # Diffusion of planktivores between refuge and foraging arena
            "v": 80, # Rate at which J enter a foraging arena, becoming vulnerable
            "h": 80, # Rate at which J hide in a refuge
            "A0": 12, # initial Adult Bass pop, taken by eye from paper
            "J0": 10, # initial Juvenile Bass pop, taken by eye from paper
            "F0": 8, # initial Forage Fish pop , taken by eye from paper
        },
    ):
        self.s = params["s"]
        self.mA = params["mA"]
        self.f = params["f"]
        self.cFA = params["cFA"]
        self.cJA = params["cJA"]
        self.cJF = params["cJF"]
        self.Fo = params["Fo"]
        self.DF = params["DF"]
        self.v = params["v"]
        self.h = params["h"]
        self.A0 = params["A0"]
        self.J0 = params["J0"]
        self.F0 = params["F0"]
        
        # enclose state space in a finite box: (TBC)
        