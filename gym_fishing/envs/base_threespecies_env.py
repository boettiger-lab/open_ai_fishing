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


class baseThreeSpeciesEnv(gym.Env):
    """
    Don't include dynamic parameters in the params dict.
    Variable names A, B, C by default. Only A is harvested.
    """

    def __init__(
        self,
        params={
            "n_actions": 100,
            "dt": 0.05,
        },
        Tmax=100,
        file=None,
    ):
        self.varname_index = {"A": 0, "B": 1, "C": 2}
        self.action_space = spaces.Discrete(self.n_actions)
        self.observation_space = spaces.Box(
            np.array([-1, -1, -1], dtype=np.float32),
            np.array([1, 1, 1], dtype=np.float32),
            dtype=np.float32,
        )

        self.Tmax = Tmax
        self.n_actions = params["n_actions"]
        self.init_pop = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        self.population = self.init_pop
        self.init_state = self.pop_to_state(self.population)
        self.state = self.init_state
        self.set_dynamics()
        self.bound_popspace()

        self.quota = 0
        self.reward = 0
        self.harvest = 0
        self.years_passed = 0

    def set_dynamics(self) -> None:
        """
        Only quadratic interactions for this generic class.
        """
        self.K = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        self.r = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        self.q_inter = np.array(  # quadratic interactions
            [[0.0, +0.1, -0.1], [-0.1, 0.0, +0.1], [+0.1, -0.1, 0.0]],
            dtype=np.float32,
        )  # particular values tbd
        self.sigma = 0.05
        self.sigmas = np.array([0.05, 0.05, 0.05], dtype=np.float32)

    def bound_popspace(self) -> None:
        """
        Enclose the space of populations in a box with
        sort-of ad-hoc boundaries. Population is not straightforward to bound
        in the multi-species case (other populations modify the carrying capacity).

        I will enclose population in a box with bound 3x the single-species carrying
        capacity. This is artificial, might need to change.
        """
        self.bounds = 3 * self.K  # ad-hoc
        self.boundA = self.bounds[0]
        self.boundB = self.bounds[1]
        self.boundC = self.bounds[2]

    def reset(self):
        self.state = self.init_state
        self.population = self.init_pop

        self.reward = 0
        self.harvest = 0
        self.years_passed = 0
        return self.state

    def step(self, action):
        quota = self.get_quota(action)
        self.population, self.reward = self.harvest_draw(quota)
        pop = (
            self.population
        )  # copy to pass into dynamics (dyn. starts after harvest draw)
        self.population = self.population_draw(pop)

    """
    Dynamic functions will be passed a copy of self.population -- this is because
    we will be updating self.population along the way in these functions.
    
    It also allows us to separate neatly the dynamics into several terms.
    """

    def quad_interactions_all(self, pop):
        delA = self.quad_interactions(pop, "A")
        delB = self.quad_interactions(pop, "B")
        delC = self.quad_interactions(pop, "C")
        return np.array([delA, delB, delC], dtype=np.float32)

    def quad_interactions(self, pop, varname):
        i = self.varname_index[varname]
        return pop[i] * (pop.dot(self.q_inter))[i]

    def logistic_all(self, pop):
        delA = self.logistic(pop, "A")
        delB = self.logistic(pop, "B")
        delC = self.logistic(pop, "C")
        return np.array([delA, delB, delC], dtype=np.float32)

    def logistic(self, pop, varname):
        i = self.varname_index[varname]
        return self.r[i] * pop[i] * (1 - pop[i] / self.K[i])

    def rnd(self, varname, pop):
        i = self.varname_index[varname]
        return pop[i] * self.sigmas[i] * np.random.normal(0, 1)

    def rnd_all(self, pop):
        delA = self.rnd(pop, "A")
        delB = self.rnd(pop, "B")
        delC = self.rnd(pop, "C")
        return np.array([delA, delB, delC], dtype=np.float32)

    def population_draw(self, pop):
        # pop = self.population -> actually do it in the self.step()
        self.population += self.logistic_all(pop) * self.dt
        self.population += self.quad_interactions_all(pop) * self.dt
        self.population += self.rnd_all(pop) * self.dt
        return self.population

    def harvest_draw(self, quota):
        harvest = quota*self.population[0]
        return (
            self.population - np.array([harvest, 0.0, 0.0], dtype=np.float32),
            harvest,
        )

    def pop_to_state(self, pop):
        """
        State components lives in [-1,1].
        Pop components lives in [0,self.K[i]]
        """
        # self.boundA = 3 * self.K[0]
        # self.boundB = 3 * self.K[1]
        # self.boundC = 3 * self.K[2]

        stateA = 2 * pop[0] / self.boundA - 1
        stateB = 2 * pop[1] / self.boundB - 1
        stateC = 2 * pop[2] / self.boundC - 1
        return np.array([stateA, stateB, stateC], dtype=np.float32)

    def state_to_pop(self, state):
        popA = (state[0] + 1) * self.boundA / 2
        popB = (state[1] + 1) * self.boundA / 2
        popC = (state[2] + 1) * self.boundA / 2
        return np.array([popA, popB, popC], dtype=np.float32)

    def get_quota(self, action):
        """
        quota = fraction of population[0] to be fished, in [0,1]
        """
        return action/self.n_actions
        
    def get_action(self, quota):
        """
        Inverse of get_quota
        """
        return round(quota*n_actions)
