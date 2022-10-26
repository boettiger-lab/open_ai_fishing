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
    varname_index = {"A": 0, "B": 1, "C": 2}
    """
    Don't include dynamic parameters in the params dict.
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
        self.set_dynamics()

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
        i = varname_index[varname]
        return pop[i] * (pop.dot(self.q_inter))[i]

    def logistic_all(self, pop):
        delA = self.logistic(pop, "A")
        delB = self.logistic(pop, "B")
        delC = self.logistic(pop, "C")
        return np.array([delA, delB, delC], dtype=np.float32)

    def logistic(self, pop, varname):
        i = varname_index[varname]
        return self.r[i] * pop[i] * (1 - pop[i] / self.K[i])

    def rnd(self, varname, pop):
        i = varname_index[varname]
        return pop[i] * sigmas[i] * np.random.normal(0, 1)

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

    def pop_to_state(self, pop):
        ...

    def state_to_pop(self, state):
        ...
