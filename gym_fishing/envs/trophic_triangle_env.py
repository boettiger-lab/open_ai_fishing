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

    def reset(self):
        self.state = self.init_state
        self.fish_population = self.init_pop
        self.years_passed = 0

        # for tracking only
        self.reward = 0
        self.harvest = 0
        return self.state

    def step(self, action):
        quota_normalized = self.get_quota(action)
        quota = 2 * self.Ahalf * quota_normalized
        self.fish_population = self.get_fish_population(self.state)
        # self.test_state_boundaries()

        # Apply harvest and population growth
        self.harvest = self.harvest_draw(quota)
        self.population_draw()

        # Map population back to system state (normalized space):
        self.state = self.get_state()

        # should be the instanteous reward, not discounted
        self.reward = max(self.harvest, 0.0)
        self.years_passed += 1
        done = bool(self.years_passed > self.Tmax)

        if any(x <= 0.0 for x in self.fish_population):
            done = True

        return self.state, self.reward, done, {}

    # use dt = 0.01 as in Carl's code (see bs-tipping/analysis/stability_report.R line 84)
    def population_draw(self):
        #
        # gets snapshot of pop right before the draw, to use within the individual
        # draw functions. Dictionary format used for readability.
        fish_dict = self.get_fish_dict()
        self.Adraw(fish_dict)
        self.Fdraw(fish_dict)
        self.Jdraw(fish_dict)
        return self.fish_population

    # fd is the fish dictionary (see population_draw and get_fish_dict)
    def Adraw(self, fd) -> None:
        self.fish_population[0] += 0.01 * (
            self.s * fd["J"] - self.mA * fd["A"]
        )
        self.fish_population[0] = max(0.0, self.fish_population[0])

    def Fdraw(self, fd) -> None:
        self.fish_population[1] += 0.01 * (
            self.DF * (self.Fo - fd["F"]) - self.cFA * fd["F"] * fd["A"]
        )
        self.fish_population[1] = max(0.0, self.fish_population[1])

    def Jdraw(self, fd) -> None:
        self.fish_population[2] += 0.01 * (
            self.f * fd["A"]
            - self.cJA * fd["J"] * fd["A"]
            - (self.cJF * self.v * fd["J"] * fd["F"])
            / (self.h + self.v + self.cJF * fd["F"])
            - self.s * fd["J"]
        )
        self.fish_population[2] = max(0.0, self.fish_population[2])

    def harvest_draw(self, quota):
        harvest = min(self.fish_population[0], quota)
        self.fish_population = np.array(
            [
                max(self.fish_population[0] - self.harvest, 0.0),
                self.fish_population[1],
                self.fish_population[2],
            ],
            dtype=np.float32,
        )
        self.smaller_population = min(self.fish_population)
        return harvest

    def variable_names(self):
        return ["A", "F", "J"]

    def get_fish_dict(self):
        fish_dict = {
            "A": self.fish_population[0],
            "F": self.fish_population[1],
            "J": self.fish_population[2],
        }
        return fish_dict

    def get_quota(self, action):
        # actions are 0, ..., 99. Must be mapped to -0.01 to 0.01
        return action / (50 * self.n_actions) - 0.01

    def get_action(self, quota):
        # inverse of get_quota
        return int(round((quota + 0.01) * 50 * self.n_actions))

    def get_fish_population(self, state):
        # include state for ease of writing test msy function
        A_pop = (state[0] + 1) * self.Ahalf
        F_pop = (state[1] + 1) * self.Fhalf
        J_pop = (state[2] + 1) * self.Jhalf
        return np.array([A_pop, F_pop, J_pop], dtype=np.float32)

    def get_state(self):
        A_st = self.fish_population[0] / self.Ahalf - 1
        F_st = self.fish_population[1] / self.Fhalf - 1
        J_st = self.fish_population[2] / self.Jhalf - 1
        return np.array([A_st, F_st, J_st], dtype=np.float32)

    def test_state_boundaries(self) -> None:
        M = max(self.state)
        m = min(self.state)
        if -1 <= m <= M <= 1:
            return None
        else:
            print(
                """
            #
            #
            #
            #
            Following state is out of bounds: {}
                
            The state boundaries used by this environment are artificial and do not
            follow naturally from the dynamics of the system.
                
            Consider increasing the magnitude of these boundaries.
            #
            #
            #
            #
            """.format(
                    self.state
                )
            )

    def simulate(env, model, reps=1):
        return simulate_mdp(env, model, reps)

    def render(self, mode="human"):
        return csv_entry(self)

    def close(self):
        if self.file is not None:
            self.write_obj.close()

    def plot(self, df, output="results.png"):
        return plot_mdp(self, df, output)

    def policyfn(env, model, reps=1):
        return estimate_policyfn(env, model, reps)

    def plot_policy(self, df, output="results.png"):
        return plot_policyfn(self, df, output)

    # for models used in tests (specifically, multiSpecies_singleHarvestBMSY)
    def get_harvested_fish_population(self, Apop):
        return (Apop + 1) * self.Ahalf

    def harvested_population_draw(self):
        self.population_draw()
        return self.fish_population[0]
