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

# consider adding support for gym logger, error, and seeding

# based on Example 3.2 of https://arxiv.org/abs/1810.05609v1
# For now I only allow harvest and not seeding.
# (i.e. dY = 0)
class BaseCompetingPairFishingEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        params={
            "x1": 0.4,  # initial state pop 1
            "x2": 0.0,  # initial state pop 2
            "K1": 0.5,  # maximal number pop 1 is 2*K1 (for normalization)
            "K2": 0.5,  # maximal number pop 2 2*K2 (for normalization)
            "b1": 3.0,
            "a11": 0.01,
            "a12": 0.1,
            "sigma1": 0.15,
            "b2": 2.0,
            "a21": 0.2,
            "a22": 0.01,
            "sigma2": 0.15,
            "n_actions": 100,
        },
        Tmax=100,
        file=None,
    ):

        # parameters
        self.x1 = params["x1"]
        self.k1 = params["K1"]
        self.b1 = params["b1"]
        self.a11 = params["a11"]
        self.a12 = params["a12"]
        self.sigma1 = params["sigma1"]
        self.x2 = params["x2"]
        self.k2 = params["K2"]
        self.b2 = params["b2"]
        self.a21 = params["a21"]
        self.a22 = params["a22"]
        self.sigma2 = params["sigma2"]
        self.n_actions = params["n_actions"]  # For discrete actions

        # For the MSY test -> needs a self.sigma parameter.
        # Use the sigma of the harvested population.
        self.sigma = self.sigma1

        # episode counter for printing
        self.ep_num = 1

        # Initial, unnormalized state
        self.init_state = np.array([self.x1, self.x2], dtype=np.float32)

        # Preserve these for reset
        self.fish_population = self.init_state
        self.smaller_population = np.min(
            self.init_state
        )  # the smaller of the two populations
        self.reward = 0
        self.harvest = 0
        self.years_passed = 0
        self.Tmax = Tmax
        self.file = file

        # for render() method only
        if file is not None:
            self.write_obj = open(file, "w+")

        # Initial normalized state, entries in [-1,1]
        self.state = np.array(
            [self.x1 / self.k1 - 1.0, self.x2 / self.k2 - 1], dtype=np.float32
        )

        # Best if cts actions / observations are normalized to a [-1, 1] domain.
        # This was the continuous action space. Only use discrete for now.
        # self.action_space = spaces.Box(
        #     np.array([-1], dtype=np.float32),
        #     np.array([1], dtype=np.float32),
        #     dtype=np.float32,
        # )

        self.action_space = spaces.Discrete(self.n_actions)
        # self.action_space = np.linspace(-1,1,num=self.n_actions,dtype=np,float32) # would this work?
        self.observation_space = spaces.Box(
            np.array([-1, -1], dtype=np.float32),
            np.array([1, 1], dtype=np.float32),
            dtype=np.float32,
        )

    def step(self, action):

        # print("STEP ")

        # Map from re-normalized model space to [0,2K] real space
        quota = self.get_quota(action)
        self.get_fish_population(
            self.state
        )  # this assigns self.fish_population to the new self.state

        # Apply harvest and population growth
        self.harvest = self.harvest_draw(quota)
        self.population_draw()

        # Map population back to system state (normalized space):
        self.get_state(self.fish_population)

        # should be the instanteous reward, not discounted
        self.reward = max(self.harvest, 0.0)
        self.years_passed += 1
        done = bool(self.years_passed > self.Tmax)

        if self.smaller_population <= 0.0:
            done = True
        #        print(
        #          "Step in Ep. {}. Final state: ".format(self.ep_num),
        #          "[ {:.3} , {:.3} ]".format(self.state[0],self.state[1]), end="\r"
        #        )
        return self.state, self.reward, done, {}

    def reset(self):
        self.state = np.array(
            [
                self.init_state[0] / self.k1 - 1,
                self.init_state[1] / self.k2 - 1,
            ],
            np.float32,
        )
        self.fish_population = self.init_state
        self.years_passed = 0

        # for tracking only
        self.reward = 0
        self.harvest = 0
        self.ep_num += 1
        return self.state

    def render(self, mode="human"):
        return csv_entry(self)

    def close(self):
        if self.file is not None:
            self.write_obj.close()

    def simulate(env, model, reps=1):
        return simulate_mdp(env, model, reps)

    def plot(self, df, output="results.png"):
        return plot_mdp(self, df, output)

    def policyfn(env, model, reps=1):
        return estimate_policyfn(env, model, reps)

    def plot_policy(self, df, output="results.png"):
        return plot_policyfn(self, df, output)

    def harvest_draw(self, quota):
        """
        Select a value to harvest at each time step.

        Population is changed, and harvested amount is returned to compute reward.
        """
        self.harvest = min(self.fish_population[0], quota)
        self.fish_population = np.array(
            [
                max(self.fish_population[0] - self.harvest, 0.0),
                self.fish_population[1],
            ],
            dtype=np.float32,
        )
        self.smaller_population = np.min(self.fish_population)
        #
        return self.harvest

    def population_draw(self):
        """
        Select a value for population to grow or decrease at each time step.

        For growth, I slightly modified Henning's model to have logistic growth instead of exponential.
        (This way, the populations are always guaranteed to be bounded and the state is always in the
        defined observation space.)
        """
        self.fish_population[0] = np.maximum(
            self.fish_population[0]
            + self.fish_population[0]
            * (
                self.b1
                * (
                    1 - self.fish_population[0] / self.k1
                )  # so that growth is logistic rather than exponential
                - self.a11 * self.fish_population[0]
                - self.a12 * self.fish_population[1]
            )
            + self.fish_population[0] * self.sigma1 * np.random.normal(0, 1),
            0.0,
        )
        self.fish_population[1] = np.maximum(
            self.fish_population[1]
            + self.fish_population[1]
            * (
                self.b2 * (1 - self.fish_population[1] / self.k2)
                - self.a21 * self.fish_population[1]
                - self.a22 * self.fish_population[0]
            )
            + self.fish_population[1] * self.sigma1 * np.random.normal(0, 1),
            0.0,
        )
        self.smaller_population = np.min(self.fish_population)
        #
        # re-standardize to float32
        self.fish_population = np.array(
            [self.fish_population[0], self.fish_population[1]],
            dtype=np.float32,
        )
        return self.fish_population

    def get_quota(self, action):
        """
        Convert action into quota
        """
        # Discrete Actions:
        if isinstance(self.action_space, gym.spaces.discrete.Discrete):
            quota = (action / self.n_actions) * self.k1
        # Continuous Actions:
        else:
            action = np.clip(
                action, self.action_space.low, self.action_space.high
            )[0]
            quota = (action + 1) * self.k1
        return quota

    def get_action(self, quota):
        """
        Convert quota into action
        """
        if isinstance(self.action_space, gym.spaces.discrete.Discrete):
            return round(quota * self.n_actions / self.K)
        else:
            return quota / self.k1 - 1

    def get_fish_population(self, state):
        self.fish_population = np.array(
            [(state[0] + 1) * self.k1, (state[1] + 1) * self.k2]
        )
        return self.fish_population

    def get_state(self, fish_population):
        self.state = np.array(
            [
                fish_population[0] / self.k1 - 1,
                fish_population[1] / self.k2 - 1,
            ]
        )
        return self.state
