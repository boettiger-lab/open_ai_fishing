import gym
import numpy as np
from gym import spaces

from gym_fishing.envs.base_threespecies_env import baseThreeSpeciesEnv
from gym_fishing.envs.shared_env import (
    csv_entry,
    estimate_policyfn,
    plot_mdp,
    plot_policyfn,
    simulate_mdp,
)


class forageVVH(gym.Env):
    """
    We will expand the 1D tipping point model of May '77 to a three-species model.
    For may, V is a resource (e.g. grass) that grows logistically and is foraged
    by a herbivore species H which is constant.

    In our model we expand this to a three species model: V1, V2 and H.
    V1 ->   patch of grass foraged by H with high rate.
            intrinsic growth is logistic
            exchanges population with patch 2
    V2 ->   patch of grass foraged by H at lower rate AND by us.
            intrinsic growth is logistic
            exchanges population with patch 1
    H ->    "Lotka-Volterra-type" predator (linear growth prop to l.c. of V1, V2,
            and death rate).
            Overall factor called self.alpha multiplying dH/dt. As it goes
            to zero it freezes H.
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
        self.v_ind = {"V1": 0, "V2": 1, "H": 2}
        self.ind_v = {0: "V1", 1: "V2", 2: "H"}

        self.Tmax = Tmax
        self.n_actions = params["n_actions"]
        self.dt = params["dt"]
        self.init_pop = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        self.pop = self.init_pop
        self.set_dynamics()
        self.bound_popspace()
        # self.pop_dict = {self.ind_v[i]: self.pop[i] for i in range(3)}
        # -> better do in step function
        self.init_state = self.pop_to_state(self.pop)
        self.state = self.init_state
        
        self.action_space = spaces.Discrete(self.n_actions)
        self.observation_space = spaces.Box(
            np.array([-1, -1, -1], dtype=np.float32),
            np.array([1, 1, 1], dtype=np.float32),
            dtype=np.float32,
        )

        self.quota = np.float32(0)
        self.reward = np.float32(0)
        self.harvest = 0
        self.years_passed = 0

    def set_dynamics(self) -> None:
        """
        Only quadratic interactions for this generic class.
        """
        self.K = {"V1": np.float32(1.0), "V2": np.float32(1.0)}
        self.r = {"V1": np.float32(1.0), "V2": np.float32(1.0)}
        #
        # quad. interactions: only quad. interaction term is the one
        # on H's equation coming from the fact that H forages both Vi.
        #
        # Recall that interaction is pop*q_inter, where pop is a row
        # vector.
        # self.q_inter = np.array(
        #     [[0.0, 0.0, 0.5], [0.0, 0.0, 0.5], [0.0, 0.0, 0.0]],
        #     dtype=np.float32,
        # )  # particular values tbd
        self.tau12 = np.float32(0.01)
        self.tau21 = np.float32(0.01)
        self.sigma = np.float32(0.05)
        self.sigmas = {
            "V1": np.float32(0.05),
            "V2": np.float32(0.05),
            "H": np.float32(0.05),
        }

        self.alpha = np.float32(
            1.0
        )  # later on used to 'turn off' model complexity
        self.beta = np.float32(0.5)
        self.f = np.float(0.5)
        self.D = np.float32(1.0)  # no discrepancy for now!
        self.V0 = np.float32(self.K["V1"] / 2)
        self.dH = np.float(0.3)
        self.f = np.float(1)

    def bound_popspace(self) -> None:
        """
        Enclose the space of populations in a box with
        sort-of ad-hoc boundaries. Population is not straightforward to bound
        in the multi-species case (other populations modify the carrying capacity).

        I will enclose population in a box with bound 3x the single-species carrying
        capacity for vegetations and simply choose a very high value for H.
        """
        self.boundV1 = 3 * self.K["V1"]
        self.boundV2 = 3 * self.K["V2"]
        self.boundH = np.float32(1000.0)

    def reset(self):
        self.state = self.init_state
        self.pop = self.init_pop

        self.reward = 0
        self.harvest = 0
        self.years_passed = 0
        return self.state

    def step(self, action):
        quota = self.get_quota(action)
        self.pop, self.reward = self.harvest_draw(quota)
        pop = {
            "V1": self.pop[0],
            "V2": self.pop[1],
            "H": self.pop[2],
        }  # copy to pass into dynamics (dyn. starts after harvest draw)
        self.pop = self.population_draw(pop)
        self.state = self.pop_to_state(self.pop)

        self.years_passed += 1
        done = bool(self.years_passed > self.Tmax)
        if any(x <= 0 for x in self.pop):
            done = True

        return self.state, self.reward, done, {}

    def harvest_draw(self, quota):
        """
        Quota is in [0,1] -- it is the faction of self.pop[0] harvested.
        """
        harvest = quota * self.pop[0]
        return (
            self.population - np.array([harvest, 0.0, 0.0], dtype=np.float32),
            harvest,
        )

    """
    Dynamics:
    
    Dynamic functions will be passed a copy of self.pop -- this is because
    we will be updating self.pop along the way in these functions.
    
    It also allows us to separate neatly the dynamics into several terms.
    """

    def population_draw(self, pop):
        Delta_V1 = self.V1_draw(pop)
        Delta_V2 = self.V2_draw(pop)
        Delta_H = self.H_draw(pop)
        self.pop += np.array(
            [Delta_V1, Delta_V2, Delta_H],
            dtype=np.float32,
        )
        return self.pop

    def V1_draw(self, pop):
        DeltaPop = 0
        DeltaPop += self.logistic(pop["V1"], self.K["V1"], self.r["V1"])
        DeltaPop += self.forage(pop["H"], pop["V1"], self.beta, self.V0)
        DeltaPop += self.tau21 * pop["V2"] - self.tau12 * pop["V1"]
        DeltaPop += pop["V1"] * self.sigmas["V1"] * np.random.normal(0, 1)
        DeltaPop = DeltaPop * self.dt
        return np.float32(DeltaPop)

    def V2_draw(self, pop):
        DeltaPop = 0
        DeltaPop += self.logistic(pop["V2"], self.K["V2"], self.r["V2"])
        DeltaPop += self.D * self.forage(
            pop["H"], pop["V2"], self.beta, self.V0
        )
        DeltaPop += self.tau12 * pop["V1"] - self.tau21 * pop["V2"]
        DeltaPop += pop["V2"] * self.sigmas["V2"] * np.random.normal(0, 1)
        DeltaPop = DeltaPop * self.dt
        return np.float32(DeltaPop)

    def H_draw(self, pop):
        DeltaPop = 0
        DeltaPop += pop["H"] * (
            self.f * (pop["V1"] + self.D * pop["V2"]) - self.dH
        )
        DeltaPop += pop["H"] * self.sigmas["H"] * np.random.normal(0, 1)
        DeltaPop = DeltaPop * self.dt
        return np.float32(DeltaPop)

    """
    Function shorthands:
    """

    def logistic(self, v, K, r):
        """
        params:
            r = growth rate
            K = carrying capacity
            v = population
        """
        return r * v * (1 - v / K)

    def forage(self, h, v, beta, v0):
        """
        params:
            h       = foraging population
            v       = foraged population
            v0      = half-saturation point for v
            beta    = foraging intensity
        """
        return beta * h * (v ^ 2) / (v0 ^ 2 + v ^ 2)

    """
    Interconversions:
    """

    def pop_to_state(self, pop):
        """
        State components lives in [-1,1].
        Pop components lives in [0,self.K[i]]
        """
        # self.boundA = 3 * self.K[0]
        # self.boundB = 3 * self.K[1]
        # self.boundC = 3 * self.K[2]

        stateA = 2 * pop[0] / self.boundV1 - 1
        stateB = 2 * pop[1] / self.boundV2 - 1
        stateC = 2 * pop[2] / self.boundH - 1
        return np.array([stateA, stateB, stateC], dtype=np.float32)

    def state_to_pop(self, state):
        """
        Inverse of pop_to_state
        """
        popA = (state[0] + 1) * self.boundV1 / 2
        popB = (state[1] + 1) * self.boundV2 / 2
        popC = (state[2] + 1) * self.boundH / 2
        return np.array([popA, popB, popC], dtype=np.float32)

    def get_quota(self, action):
        """
        quota = fraction of population[0] to be fished, in [0,1]
        """
        return action / self.n_actions

    def get_action(self, quota):
        """
        Inverse of get_quota
        """
        return round(quota * self.n_actions)

    """
    Other testing / helpers
    """

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
