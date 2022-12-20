""" The gym.env class forageVVHcont which is the same as forageVVH
but with a continuous action space """

from gym_fishing.envs.forage_VVH_env import forageVVH
import numpy as np
from gym import spaces
from pandas import DataFrame

class forageVVHcont(forageVVH):
    """ continuous action space """
    def __init__(
        self,
        params={
            "dt": 1.,
        },
        Tmax=200,
        file=None,
    ):
        self.training=True # manually set to false for non-training (e.g. for controlled_dynamics)
        
        self.v_ind = {"V1": 0, "V2": 1, "H": 2}
        self.ind_v = {0: "V1", 1: "V2", 2: "H"}

        self.Tmax = Tmax
        self.dt = params["dt"]
        self.init_pop = np.array([0.7921041547384046, 0.18899622296518745, 0.2281360434110448], dtype=np.float32) # a fixed point for current param values
        self.pop = self.init_pop
        self.set_dynamics()
        # self.set_May_dynamics()
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
    
    def get_quota(self, action):
        """
        now action and quota are the same thing
        """
        if isinstance(action, np.ndarray):
            action = action[0]
        return action

    def get_action(self, quota):
        """
        as above
        """
        return np.array([quota], dtype=np.float32)
    
    def controlled_dynamics(self, T, verbose=False, ctrl="simple", reps=1):
        row = []
        for rep in range(reps):
            self.reset()
            self.tot_reward = 0.0
            #
            if ctrl == "simple":
                ctrl_fn = self.simple_control
            if ctrl == "escapement":
                ctrl_fn = self.escapement_control
            if ctrl == "msy":
                ctrl_fn = self.msy_control
            if ctrl == "tac":
                ctrl_fn = self.tac_control
            # else:
            #     print("ctrl variable can only be = 'simple', 'escapement', 'msy' or 'tac'")
            #     return None
            #
            for t in range(T):
                act = ctrl_fn()
                self.tot_reward += self.reward
                row.append([
                    t, 
                    self.pop[0], 
                    self.pop[1], 
                    self.pop[2],
                    act,
                    self.reward,
                    int(rep),
                ])
                self.step(act)
                if verbose:
                    self.print_pop()
                    time.sleep(0.03)
        self.reset()
        df = DataFrame(
            row, 
            columns=["time", "pop0", "pop1", "pop2", "action", "reward", "rep"],
            # index=[i for i in range(index_start, index_start+T)],
        )
        return df