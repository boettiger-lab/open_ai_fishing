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
        self.init_pop = np.array([0.7, 0.7, 0.5], dtype=np.float32)
        self.pop = self.init_pop
        # self.set_dynamics()
        self.set_May_dynamics()
        self.bound_popspace()
        self.randomize_reset = False
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

    def set_May_dynamics(self) -> None:
        self.K = {"V1": np.float32(1.0), "V2": np.float32(1.0)}
        self.r = {"V1": np.float32(1.0), "V2": np.float32(1.0)}
        """
        Using K(V1) = r(V1) = 1, to reproduce May we need 
        
        gamma:=beta*H in {0.35 (high - after tipping), 0.22 (mid - close
        to tipping), 0.10 (low - far from tipping)}.
        
        Use ad hoc beta in that order of magnitude and see what happens - this
        will be particularly good for very small alpha.
        """
        self.tau12 = np.float32(0.0)
        self.tau21 = np.float32(0.0)
        self.sigma = np.float32(0.0)
        self.sigmas = {
            "V1": np.float32(0.0),
            "V2": np.float32(0.0),
            "H": np.float32(0.0),
        }

        self.alpha = np.float32(
            0.0
        )  # later on used to 'turn off' model complexity
        self.beta = np.float32(0.4)
        self.f = np.float32(0.5)
        self.D = np.float32(1.0)  # no discrepancy for now!
        self.V0 = np.float32(0.1 * self.K["V1"])
        self.dH = np.float32(0.1)

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
        self.f = np.float32(0.5)
        self.D = np.float32(1.0)  # no discrepancy for now!
        self.V0 = np.float32(self.K["V1"] / 2)
        self.dH = np.float32(0.3)

    def bound_popspace(self) -> None:
        """
        Enclose the space of populations in a box with
        sort-of ad-hoc boundaries. Population is not straightforward to bound
        in the multi-species case (other populations modify the carrying capacity).

        I will enclose population in a box with bound equal to the single-species
        carrying capacity for vegetations. For H, I numerically tested (by evolving
        with a constant action, for different values of self.alpha between 0 and 1)
        and found that 2. is a good enough bound. These evolutions also give evidence
        that the carrying capacity bounds are not exceeded (i.e. that there is no
        overshooting above the carrying capacity).

        All of these are still pending to change if new data shows they're not large
        enough.
        """
        self.boundV1 = 1 * self.K["V1"]
        self.boundV2 = 1 * self.K["V2"]
        self.boundH = np.float32(2.0)

    def reset(self):
        rand_part = np.array([0, 0, 0], dtype=np.float32)
        if self.randomize_reset == True:
            rand_part = 0.01 * np.array(
                [
                    (1 - self.init_state[0])  # bound away from +1 boundary
                    * np.random.normal(0, 1),
                    (1 - self.init_state[1]) * np.random.normal(0, 1),
                    (1 - self.init_state[2]) * np.random.normal(0, 1),
                ],
                dtype=np.float32,
            )

        self.state = self.init_state + rand_part
        self.pop = self.state_to_pop(self.state)
        self.reward = 0
        self.harvest = 0
        self.years_passed = 0
        return self.state

    def step(self, action):
        """
        Steps will be 1 full t step (i.e. self.dt^-1 individual time steps).
        Ie. I call population_draw self.dt^-1 times.
        """
        thresh = 1e-5
        quota = self.get_quota(action)
        self.pop, self.reward = self.harvest_draw(quota)
        STEP = round(self.dt ** (-1))
        for _ in range(STEP):
            pop = {
                "V1": self.pop[0],
                "V2": self.pop[1],
                "H": self.pop[2],
            }  # copy to pass into dynamics (dyn. starts after harvest draw)
            self.pop = self.population_draw(pop)
            self.state = self.pop_to_state(self.pop)
            self.test_state_boundaries()

        self.years_passed += 1
        done = bool(self.years_passed > self.Tmax)
        if any(x <= thresh for x in self.pop):
            done = True

        return self.state, self.reward, done, {}

    def harvest_draw(self, quota):
        """
        Quota is in [0,1] -- it is the faction of self.pop[0] harvested.
        """
        harvest = quota * self.pop[0]
        return (
            self.pop - np.array([harvest, 0.0, 0.0], dtype=np.float32),
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
        DeltaPop -= self.forage(pop["H"], pop["V1"], self.beta, self.V0)
        DeltaPop += self.tau21 * pop["V2"] - self.tau12 * pop["V1"]
        DeltaPop += pop["V1"] * self.sigmas["V1"] * np.random.normal(0, 1)
        DeltaPop = DeltaPop * self.dt
        return np.float32(DeltaPop)

    def V2_draw(self, pop):
        DeltaPop = 0
        DeltaPop += self.logistic(pop["V2"], self.K["V2"], self.r["V2"])
        DeltaPop -= self.D * self.forage(
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
        DeltaPop = self.alpha * DeltaPop * self.dt
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
        return beta * h * (v**2) / (v0**2 + v**2)

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

    def scan_fixed_points(self):
        fixed_point_table = {}
        grain = 50
        for i in range(1, grain + 1):
            beta = i * (grain ** (-1))
            fixed_point_table[beta] = self.find_fixed_points(beta)
        print(fixed_point_table)
        return fixed_point_table

    def save_fixed_points(self):
        self.fixed_points = self.find_fixed_points(self.beta)
        """
        List at the bottom for a variety of beta values.
        """

    def find_fixed_points(self, beta):
        self.beta = beta
        incr = 0.01
        steps = round(incr**-1)
        prev = ""
        curr = ""
        fixed_points = []
        skip = False
        for i in range(1, steps):
            self.reset()
            self.pop[0] = incr * i
            pop = {
                "V1": self.pop[0],
                "V2": self.pop[1],
                "H": self.pop[2],
            }
            pop_prime = self.population_draw(pop)
            if pop_prime[0] - pop["V1"] > 0:
                curr = "+"
                if curr != prev and prev != "0" and i > 1:
                    # print(pop["V1"], end="")
                    fixed_points.append(pop["V1"])
                # else:
                # print("+", end="")
            if pop_prime[0] - pop["V1"] < 0:
                curr = "-"
                if curr != prev and prev != "0" and i > 1:
                    # print(pop["V1"], end="")
                    fixed_points.append(pop["V1"])
                # else:
                # print("-", end="")
            if pop_prime[0] == pop["V1"]:
                curr = "0"
                # print(pop["V1"], end="")
                fixed_points.append(pop["V1"])
            prev = curr
        # print("\n")
        return fixed_points

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
            
            Current boundaries are: boundV1 = {}, boundV2 = {}, boundH = {}
                
            Consider increasing the magnitude of these boundaries.
            #
            #
            #
            #
            """.format(
                    self.state,
                    self.boundV1,
                    self.boundV2,
                    self.boundH,
                )
            )


"""
TABLE OF FIXED POINTS

beta:    array of fixed points
==============================
0.04:    [0.98]
0.06:    [0.97]
0.08:    [0.96]
0.10:    [0.95]
0.12:    [0.94]
0.14:    [0.93]
0.16:    [0.92]
0.18:    [0.91]
0.20:    [0.89]
0.22:    [0.88]
0.24:    [0.87]
0.26:    [0.85]
0.28:    [0.84]
0.30:    [0.82]
0.32:    [0.81]
0.34:    [0.79]
0.36:    [0.1, 0.13, 0.78]
0.38:    [0.08, 0.18, 0.76]
0.40:    [0.07, 0.2, 0.74]
0.42:    [0.07, 0.23, 0.71]
0.44:    [0.06, 0.26, 0.69]
0.46:    [0.06, 0.29, 0.66]
0.48:    [0.05, 0.33, 0.63]
0.50:    [0.05, 0.37, 0.59]
0.52:    [0.05, 0.46, 0.5]
0.54:    [0.05]
0.56:    [0.04]
0.58:    [0.04]
0.60:    [0.04]
0.62:    [0.04]
0.64:    [0.04]
0.66:    [0.04]
0.68:    [0.04]
0.70:    [0.04]
0.72:    [0.03]
"""
