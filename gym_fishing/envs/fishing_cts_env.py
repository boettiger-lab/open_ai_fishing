from gym_fishing.envs.base_fishing_env import BaseFishingEnv
import numpy as np
np.random.seed()

class FishingCtsEnv(BaseFishingEnv):
    def __init__(
        self, init = "notrandom" , Seed = 1, r=0.3, K=1, sigma=0.0, init_state= 0.75, Tmax=100, file=None
    ):
        np.random.seed(Seed)
        self.init_state = 0.75 + ( np.random.rand() - 0.75) * (init == "random")
        super().__init__(
            params={"r": 0.3, "K": 1, "sigma": 0.0, "x0": 0.75, "Seed":1, "init" : "random"},
            Tmax=Tmax,
            file=file,
        )
