import time

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
        Tmax=200,
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
        self.randomize_reset = True
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
        self.tot_reward = 0.0
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
            1.0
        )  # later on used to 'turn off' model complexity

        """
        The failure thresh is the value of the lower fixed point
        (see fixed point table at the end of file). I use 1.1 times
        the FP value to be able to more quickly "catch" these cases
        and not have to wait long times for convergence.
        """
        self.beta = np.float32(0.4)
        self.failure_thresh = np.float32(0.07 * 1.1)

        self.f = np.float32(0.5)
        self.D = np.float32(1.0)  # no discrepancy for now!
        self.V0 = np.float32(0.1 * self.K["V1"])
        self.dH = np.float32(0.5)

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
        self.boundV2 = 1.5 * self.K["V2"]
        self.boundH = np.float32(2.0)

    def reset(self):
        rand_part = np.array([0, 0, 0], dtype=np.float32)
        if self.randomize_reset == True:
            rand_part = 0.05 * np.array(
                [
                    np.random.normal(0, 1),
                    np.random.normal(0, 1),
                    np.random.normal(0, 1),
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

        Done: if V1 decays to lower equilibrium, or, if V2 or H go near-extinct.
        """
        thresh = 1e-4
        thresh_arr = [self.failure_thresh, thresh, thresh]

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
        if any(self.pop[i] <= thresh_arr[i] for i in range(3)):
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

    def print_pop(self) -> None:
        # for pretty printing
        LINE_UP = "\033[1A"
        LINE_CLEAR = "\x1b[2K"

        s1 = round(30 * self.pop[0])
        s2 = round(30 * self.pop[1])
        s3 = round(30 * self.pop[2])
        l1 = (
            " | "
            + (s1 - 1) * " "
            + "x"
            + (30 - s1) * " "
            + " | "
            + f"{self.pop[0]:.2f}"
        )
        l2 = (
            " | "
            + (s2 - 1) * " "
            + "x"
            + (30 - s2) * " "
            + " | "
            + f"{self.pop[1]:.2f}"
        )
        l3 = (
            " | "
            + (s3 - 1) * " "
            + "x"
            + (30 - s3) * " "
            + " | "
            + f"{self.pop[2]:.2f}"
        )

        print(l1)
        print(l2)
        print(l3)
        print(f"{self.tot_reward:.2f}")
        print(LINE_UP, end=LINE_CLEAR)
        print(LINE_UP, end=LINE_CLEAR)
        print(LINE_UP, end=LINE_CLEAR)
        print(LINE_UP, end=LINE_CLEAR)

        # l4 = f"{self.reward:.2f}"
        # print(l0+l1+l2+l3+l4, end="\r")

    def uncontrolled_dynamics(self, T) -> None:
        self.reset()
        for t in range(T):
            self.step(0)
            # print(f"Pop: [{self.pop[0]:.2f},  {self.pop[1]:.2f}, {self.pop[2]:.2f}]")
            self.print_pop()
            time.sleep(0.03)
        self.reset()

    def find_msy(self):
        """
        Logic: each year, people estimate v1, v2, and h, and calculate which is the msy
        level for v1 assuming h and v2 are constant. (In fact, the msy in this constant
        picture only depends on h, not on v2, since v2 doesn't appear directly in v1's eq.
        """
        # first turn off noise to find maximum yield:
        sigmas = self.sigmas
        self.sigmas = {"V1": 0.0, "V2": 0.0, "H": 0.0}

        # now scan over the space of v1 possibilities.
        v1space = np.linspace(0, 1, 500)
        growth = {}
        for v1 in v1space:
            pop = {"V1": v1, "V2": self.pop[1], "H": self.pop[2]}
            growth[v1] = V1_draw(self, pop)[0]
            # V1_draw outputs the 'delta', the growth.
        return max(growth, key=growth.get)  # returns optimal key (v1)

    def msy_control(self):
        msy = self.find_msy()
        if msy <= 0:
            print("Problem: MSY isn't positive, it's {}".format(msy))
            return 0.0
        if self.pop[0] > msy:
            return (self.pop[0] - msy) / self.pop[0]
        else:
            return 0.0

    def simple_control(self):
        # V1 = self.pop[0]
        self.simple_bang = 0.5
        if self.pop[0] > self.simple_bang:
            # harvest up to V1 = 0.5
            return 0.4
        else:
            return 0.0

    def controlled_dynamics(self, T, verbose=True, ctrl="simple"):
        self.reset()
        self.tot_reward = 0.0
        #
        if ctrl == "simple":
            ctrl_fn = self.simple_control
        if ctrl == "msy":
            ctrl_fn = self.msy_control
        else:
            print("ctrl variable can only be = 'simple' or 'msy'")
            return None
        #
        for t in range(T):
            harv = ctrl_fn()
            act = round(harv * self.n_actions)
            self.step(act)
            self.tot_reward += self.reward
            if verbose:
                self.print_pop()
                time.sleep(0.03)
        self.reset()
        return self.tot_reward

    def scan_fixed_points(self):
        fixed_point_table = {}
        grain = 100
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
TABLE OF FIXED POINTS: 

the condition for done=True will be a decay to the lowest
fixed point. Notice that the middle fixed point is unstable.

beta:    array of fixed points
==============================
0.17:    [0.91]
0.18:    [0.91]
0.19:    [0.9]
0.20:    [0.89]
0.21:    [0.89]
0.22:    [0.88]
0.23:    [0.87]
0.24:    [0.87]
0.25:    [0.86]
0.26:    [0.85]
0.27:    [0.85]
0.28:    [0.84]
0.29:    [0.83]
0.30:    [0.82]
0.31:    [0.82]
0.32:    [0.81]
0.33:    [0.8]
0.34:    [0.79]
0.35:    [0.78]
0.36:    [0.1, 0.13, 0.78]
0.37:    [0.09, 0.16, 0.77]
0.38:    [0.08, 0.18, 0.76]
0.39:    [0.08, 0.19, 0.75]
0.40:    [0.07, 0.2, 0.74]
0.41:    [0.07, 0.22, 0.73]
0.42:    [0.07, 0.23, 0.71]
0.43:    [0.06, 0.25, 0.7]
0.44:    [0.06, 0.26, 0.69]
0.45:    [0.06, 0.28, 0.68]
0.46:    [0.06, 0.29, 0.66]
0.47:    [0.06, 0.31, 0.65]
0.48:    [0.05, 0.33, 0.63]
0.49:    [0.05, 0.35, 0.61]
0.50:    [0.05, 0.37, 0.59]
0.51:    [0.05, 0.4, 0.56]
0.52:    [0.05, 0.46, 0.5]
0.53:    [0.05]
0.54:    [0.05]
0.55:    [0.05]
0.56:    [0.04]
0.57:    [0.04]
0.58:    [0.04]
0.59:    [0.04]
0.60:    [0.04]
0.61:    [0.04]
0.62:    [0.04]
0.63:    [0.04]
0.64:    [0.04]
0.65:    [0.04]
0.66:    [0.04]
0.67:    [0.04]
0.68:    [0.04]
0.69:    [0.04]
0.70:    [0.04]
0.71:    [0.03]
0.72:    [0.03]
0.73:    [0.03]
"""
