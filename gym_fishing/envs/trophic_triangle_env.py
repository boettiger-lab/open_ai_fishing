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
        
        # enclose state space in a finite box: (artificial boundary chosen large enoug:
        # for now, simply choose 2*maximum in the simulations of the reference)
        self.Ahalf = 12. # Amax is 2*Ahalf
        self.Fhalf = 200.
        self.Jhalf = 8.
        self.maxPops = np.array(
            [2*self.Ahalf, 2*self.Fhalf, 2*self.Jhalf],
            dtype = np.float32
        )
        
        self.initial_pop = np.array([A0, J0, F0], dtype=np.float32)
        self.initial_state = np.array(
            [self.A0/self.Ahalf - 1., 
            self.F0/self.Fhalf - 1.,
            self.J0/self.Jhalf - 1.],
            dtype=np.float32
        )

        # Preserve these for reset
        self.fish_population = self.initial_pop
        self.smaller_population = np.min(
            self.initial_pop
        )  # the smaller of the populations
        self.reward = 0
        self.harvest = 0
        self.years_passed = 0
        self.Tmax = Tmax
        self.file = file

        # for render() method only
        if file is not None:
            self.write_obj = open(file, "w+")
        
        
        










