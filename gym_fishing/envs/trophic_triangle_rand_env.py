import gym
import numpy as np
from gym import spaces

from gym_fishing.envs.trophic_triangle_env import trophicTriangleEnv


class trophicTriangleRandEnv(trophicTriangleEnv):
    def __init__(
        self,
        params={  # taken from Table 1 in reference
            "s": 0.6,  # Survival rate of juveniles to maturation
            "mA": 0.4,  # Mortality rate of adults
            "f": 2,  # Fecundity of adult bass
            "cFA": 0.3,  # Predation of planktivores by adult bass
            "cJA": 0.1,  # Predation of juvenile bass on by adult bass
            "cJF": 0.5,  # Predation of juvenile bass by planktivorous fish
            "Fo": 200,  # Abundance of planktivorous fish in non-foraging arena
            "DF": 0.09,  # Diffusion of planktivores between refuge and foraging arena
            "v": 80,  # Rate at which J enter a foraging arena, becoming vulnerable
            "h": 80,  # Rate at which J hide in a refuge
            "A0": 12,  # initial Adult Bass pop, taken by eye from paper
            "F0": 8,  # initial Forage Fish pop , taken by eye from paper
            "J0": 10,  # initial Juvenile Bass pop, taken by eye from paper
            "n_actions": 200,  # number of possible actions (harvests) evenly spaced out
            "sigmaA": 0.1,
            "sigmaF": 0.1,
            "sigmaJ": 0.1,
        },
        Tmax=102000,  # taken from reference (1020y with 0.01y steps)
        file=None,
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
        self.F0 = params["F0"]
        self.J0 = params["J0"]
        self.n_actions = params["n_actions"]
        self.Tmax = Tmax
        self.sigmaA = params["sigmaA"]
        self.sigmaF = params["sigmaF"]
        self.sigmaJ = params["sigmaJ"]

        # for the tests in github's action
        self.sigma = self.sigmaA

        # test using temp-script.py gave an idea of how large can A, F and J get.
        # It uses a constant quota policy, loops over possible quotas and outputs
        # the largest. Here I use that guide and set the box as 2*that guide.
        self.Ahalf = 88  # Amax is 2*Ahalf
        self.Fhalf = 200.0
        self.Jhalf = 19
        self.maxPops = np.array(
            [2 * self.Ahalf, 2 * self.Fhalf, 2 * self.Jhalf], dtype=np.float32
        )

        self.init_pop = np.array([self.A0, self.J0, self.F0], dtype=np.float32)
        self.state = np.array(
            [
                self.A0 / self.Ahalf - 1.0,
                self.F0 / self.Fhalf - 1.0,
                self.J0 / self.Jhalf - 1.0,
            ],
            dtype=np.float32,
        )

        # Preserve these for reset
        self.init_state = self.state
        self.fish_population = self.init_pop
        self.smaller_population = min(
            self.init_pop
        )  # the smaller of the populations
        self.reward = 0
        self.harvest = 0
        self.years_passed = 0
        self.Tmax = Tmax
        self.file = file

        # for render() method only
        if file is not None:
            self.write_obj = open(file, "w+")

        self.action_space = spaces.Discrete(self.n_actions)
        # self.action_space = np.linspace(-1,1,num=self.n_actions,dtype=np,float32) # would this work?
        self.observation_space = spaces.Box(
            np.array([-1, -1, -1], dtype=np.float32),
            np.array([1, 1, 1], dtype=np.float32),
            dtype=np.float32,
        )


        # for BMSY model - set sigmas to zero
    def turn_noise_off(self):
        sigmaA = self.sigmaA
        sigmaF = self.sigmaF
        sigmaJ = self.sigmaJ
        self.sigmaA = 0
        self.sigmaF = 0
        self.sigmaJ = 0
        return [sigmaA, sigmaF, sigmaJ]

    def turn_noise_on(self, sigmaArr):
        # notice that,
        # sigmaArr = obj.turn_noise_off()
        # obj.turn_noise_on(sigmaArr)
        # leaves the object as it began
        self.sigmaA = sigmaArr[0]
        self.sigmaF = sigmaArr[1]
        self.sigmaJ = sigmaArr[2]

    # Redefine the dynamic functions to include a random term
    def Adraw(self, fd) -> None:
        self.fish_population[0] += 0.01 * (
            self.s * fd["J"]
            - self.mA * fd["A"]
            + self.sigmaA * np.random.normal(0, 1)
        )
        self.fish_population[0] = max(0.0, self.fish_population[0])

    def Fdraw(self, fd) -> None:
        self.fish_population[1] += 0.01 * (
            self.DF * (self.Fo - fd["F"])
            - self.cFA * fd["F"] * fd["A"]
            + self.sigmaF * np.random.normal(0, 1)
        )
        self.fish_population[1] = max(0.0, self.fish_population[1])

    def Jdraw(self, fd) -> None:
        self.fish_population[2] += 0.01 * (
            self.f * fd["A"]
            - self.cJA * fd["J"] * fd["A"]
            - (self.cJF * self.v * fd["J"] * fd["F"])
            / (self.h + self.v + self.cJF * fd["F"])
            - self.s * fd["J"]
            + self.sigmaJ * np.random.normal(0, 1)
        )
        self.fish_population[2] = max(0.0, self.fish_population[2])

