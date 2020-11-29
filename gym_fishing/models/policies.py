import numpy as np

class msy:
    def __init__(self, env, **kwargs):
      self.env = env
      self.S = BMSY(env)
      ## use the built-in method to determine MSY
      env.fish_population = self.S
      sigma = env.sigma
      env.sigma = 0
      self.msy = env.population_draw() - self.S
      env.sigma = sigma
      env.reset()

    def predict(self, obs, **kwargs):
      msy = self.msy
      action = self.env.get_action(msy)
      return action, obs

class escapement:
    def __init__(self, env, **kwargs):
      self.env = env
      self.S = BMSY(env)

    def predict(self, obs, **kwargs):
      fish_population = self.env.get_fish_population(obs)
      quota = max(fish_population - self.S, 0.) 
      action = self.env.get_action(quota)
      return action, obs

class user_action:
    def __init__(self, env, **kwargs):
      self.env = env

    def predict(self, obs, **kwargs):
      fish_population = self.env.get_fish_population(obs)
      prompt = "fish population: " + str(fish_population) + ". Your harvest quota: "
      quota = input(prompt)
      action = self.env.get_action(float(quota))
      return action, obs


## Note, this resets the environment
def BMSY(env):
    n = 10001 # ick should  be cts
    state_range = np.linspace(env.observation_space.low, 
                            env.observation_space.high, 
                            num=n, 
                            dtype=env.observation_space.dtype)
    x_0 = np.asarray(list(map(env.get_fish_population, state_range)))
    env.fish_population = x_0
    sigma = env.sigma
    env.sigma = 0
    growth = env.population_draw() - x_0
    S = x_0[np.argmax(growth)]
    env.sigma = sigma
    env.reset()
    return(S)


