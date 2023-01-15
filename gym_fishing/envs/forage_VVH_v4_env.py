""" The gym.env class forageVVHv4 which is the same as forageVVHcont
but with parametric noise on reset. """

from gym_fishing.envs.forage_VVH_cont_env import forageVVHcont
import numpy as np
from gym import spaces
from pandas import DataFrame

class forageVVHv4(forageVVHcont):
	def __init__(
		self,
		params={
			"dt": 1.,
		},
		Tmax=600,
		file=None,
		):
		self.training=True # manually set to false for non-training (e.g. for controlled_dynamics)

		self.v_ind = {"V1": 0, "V2": 1, "H": 2}
		self.ind_v = {0: "V1", 1: "V2", 2: "H"}

		self.Tmax = Tmax
		self.dt = params["dt"]
		self.init_pop = np.array([0.8396102377828771, 0.05489978383850558, 0.3773367609828674], dtype=np.float32) # a fixed point for current param values
		self.pop = self.init_pop
		self.set_dynamics()
		# self.set_May_dynamics()
		self.set_randomness()
		self.bound_popspace()
		self.randomize_reset = True
		# self.pop_dict = {self.ind_v[i]: self.pop[i] for i in range(3)}
		# -> better do in step function
		self.init_state = self.pop_to_state(self.pop)
		self.state = self.init_state

		self.action_space = spaces.Box(
			np.array([0], dtype=np.float32),
			np.array([1], dtype=np.float32),
			dtype = np.float32
		)
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

	def reset(self):
		"""
		No reset noise for state. Reset noise for some parameters though.
		structure:
			parameter += sigma * parameter * np.normal(0,1)

		so that the variance scales down with the parameter size
		(and, e.g., sign is preserved.)
		"""
		self.state = self.init_state 
		self.pop = self.state_to_pop(self.state)
		self.reward = 0
		self.harvest = 0
		self.years_passed = 0

		self.set_dynamics()
		self.parametric_noise(sigma = 0.05)

		return self.state

	def parametric_noise(self, sigma = 0.05):
		"""
		Parameters:
		r1, K1, r2, K2, cV, beta, dH, D, f

		I'll start by varying one parameter at a time
		"""
		# self.r["V1"] += sigma * self.r["V1"] * np.random.normal()
		# self.beta += sigma * self.beta * np.random.normal()
		# self.r["V2"] += sigma * self.r["V2"] * np.random.normal()
		# self.K["V2"] += sigma * self.K["V2"] * np.random.normal()
		self.K["V1"] += sigma * self.K["V1"] * np.random.normal()
		# self.cV += sigma * self.cV * np.random.normal()

	def step(self, action):
		"""
		Steps will be 1 full t step (i.e. self.dt^-1 individual time steps).
		Ie. I call population_draw self.dt^-1 times.
		"""
		thresh = 1e-4
		thresh_arr = [self.failure_thresh, thresh, thresh]

		action = np.clip(action, [0], [1])
		quota = self.get_quota(action)
		
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
		self.pop, self.reward = self.harvest_draw(quota)

		self.years_passed += 1
		done = bool(self.years_passed > self.Tmax)
		if any(self.pop[i] <= thresh_arr[i] for i in range(3)) and self.training:
			done = True
			self.reward -= self.Tmax/self.years_passed

		return self.state, self.reward, done, {}
