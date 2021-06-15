from gym_fishing.envs.base_fishing_env import BaseFishingEnv
import numpy as np
np.random.seed(1)

class FishingCtsEnv(BaseFishingEnv):
    def __init__(
        self, init = "notrandom" , r=0.3, K=1, sigma=0.0, init_state= 0.75, Tmax=100, file=None
    ):
        self.init_state = 0.75 + ( np.random.rand() - 0.75) * (init == "random")
        super().__init__(
            params={"r": 0.3, "K": 1, "sigma": 0.0, "x0": 0.75, "init" : "random"},
            Tmax=Tmax,
            file=file,
        )
