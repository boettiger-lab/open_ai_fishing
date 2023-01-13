""" The gym.env class forageVVHv3 which is the same as forageVVHcont
but with a quadratic error cost. """

from gym_fishing.envs.forage_VVH_cont_env import forageVVHcont
import numpy as np
from gym import spaces
from pandas import DataFrame

class forageVVH_v3(forageVVHcont):
	def step(self, action):
	    """
	    Steps will be 1 full t step (i.e. self.dt^-1 individual time steps).
	    Ie. I call population_draw self.dt^-1 times.
	    reward = harvest - (coeff) * (effort) ** 2,
	    where effort = action = harvest / population, so,
	   	reward = action * population - coeff * (action ** 2).
	    """
        thresh = 1e-4
        thresh_arr = [self.failure_thresh, thresh, thresh]
        
        action = np.clip(action, [0], [1])
        quota = self.get_quota(action)
        self.pop, self.harvest = self.harvest_draw(quota)

        self.quadratic_coeff = 1
        self.reward = self.harvest - self.quadratic_coeff * (action ** 2)

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
        # if any(self.pop[i] <= thresh_arr[i] for i in range(3)) and self.training:
        #     done = True
        #     self.reward -= 50/self.years_passed
        #     # self.reward -= (
        #     #     50 / (self.years_passed ** (1) )
        #     #     + 50 / (self.years_passed ** (0.5) )
        #     # )

        return self.state, self.reward, done, {}
