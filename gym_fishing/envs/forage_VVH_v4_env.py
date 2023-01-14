""" The gym.env class forageVVHv4 which is the same as forageVVHcont
but with parametric noise on reset. """

from gym_fishing.envs.forage_VVH_cont_env import forageVVHcont
import numpy as np
from gym import spaces
from pandas import DataFrame

class forageVVHv4(forageVVHcont):
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

		self.parametric_noise(sigma = 0.1)

		return self.state

	def parametric_noise(self, sigma = 0.1):
		"""
		Parameters:
		r1, K1, r2, K2, cV, beta, dH, D, f

		I'll start by varying one parameter at a time
		"""
		self.cV += sigma * self.cV * np.random.normal()