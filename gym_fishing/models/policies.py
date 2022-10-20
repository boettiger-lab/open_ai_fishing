import numpy as np


class msy:
    def __init__(self, env, **kwargs):
        self.env = env
        self.S = altBMSY(env)
        # use the built-in method to determine MSY
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


class multiSpecies_singleHarvest_msy:
    def __init__(self, env, **kwargs):
        self.env = env
        self.S = multiSpecies_singleHarvestBMSY(env)
        # use the built-in method to determine MSY
        env.fish_population = self.S
        if hasattr(env, "sigma"):
            sigmaArr = env.turn_noise_off()
        # sigma = env.sigma
        # env.sigma = 0

        self.msy = env.population_draw() - self.S
        if hasattr(env, "sigma"):
            env.turn_noise_on(sigmaArr)
        # env.sigma = sigma
        env.reset()

    def predict(self, obs, **kwargs):
        msy = self.msy
        action = self.env.get_action(msy)
        return action, obs


class escapement:
    def __init__(self, env, **kwargs):
        self.env = env
        self.S = altBMSY(env)

    def predict(self, obs, **kwargs):
        fish_population = self.env.get_fish_population(obs)
        quota = max(fish_population - self.S, 0.0)
        action = self.env.get_action(quota)
        return action, obs


class user_action:
    def __init__(self, env, **kwargs):
        self.env = env

    def predict(self, obs, **kwargs):
        fish_population = self.env.get_fish_population(obs)
        prompt = (
            "fish population: "
            + str(fish_population)
            + ". Your harvest quota: "
        )
        quota = input(prompt)
        action = self.env.get_action(float(quota))
        return action, obs


# Line marked with ### gave an error
#
# Note, this resets the environment
# def BMSY(env):
#     n = 10001  # ick should  be cts
#     state_range = np.linspace(
#         env.observation_space.low,
#         env.observation_space.high,
#         num=n,
#         dtype=env.observation_space.dtype,
#     )
#     x_0 = np.asarray(list(map(env.get_fish_population, state_range)))
#     env.fish_population = x_0
#     sigma = env.sigma
#     env.sigma = 0
#     growth = env.population_draw() - x_0 ###
#     S = x_0[np.argmax(growth)]
#     env.sigma = sigma
#     env.reset()
#     return S

# Note, this resets the environment
def altBMSY(env):
    n = 10001  # ick should  be cts
    state_range = np.linspace(
        env.observation_space.low,
        env.observation_space.high,
        num=n,
        dtype=env.observation_space.dtype,
    )
    growth = []
    x_0 = np.asarray(list(map(env.get_fish_population, state_range)))
    for xx in x_0:
        sigma = env.sigma
        env.sigma = 0
        env.fish_population = xx
        growth.append(env.population_draw() - xx)
    S = x_0[np.argmax(growth)]
    env.sigma = sigma
    env.reset()
    return S

def multiSpecies_singleHarvestBMSY(env):
    n = 1001  # ick should  be cts
    env.reset()
    varnames = env.variable_names()
    # TBD: can we do this for general n species nicely?
    
    # manual for now:
    Alow = env.observation_space.low[0]
    Ahigh = env.observation_space.high[0]
    Flow = env.observation_space.low[1]
    Fhigh = env.observation_space.high[1]
    Jlow = env.observation_space.low[2]
    Jhigh = env.observation_space.high[2]
    state1_space, state2_space, state3_space = np.mgrid(Alow:Ahigh:n*j, Flow:eFhigh:n*j, Jlow:Jhigh:n*j)
    
    state_range = [np.array(
        state1_space[i],
        state2_space[i],
        state3_space[i],
    ) for i in range(len(state1_space))]

    # state_range = np.linspace(
    #     env.observation_space.low[0],
    #     env.observation_space.high[0],
    #     num=n,
    #     dtype=env.observation_space.dtype,
    # )
    growth = []
    x_0 = np.asarray(list(map(env.get_fish_population, state_range)))
    for xx in x_0:
        sigmaArr = []
        if hasattr(env, "sigma"):
            sigmaArr = env.turn_noise_off()
            # sigma = env.sigma
            # env.sigma = 0
        env.fish_population = xx
        growth.append(env.population_draw() - xx)
    S = x_0[np.argmax(growth)]
    if hasattr(env, "sigma"):
        env.turn_noise_on(sigmaArr)
        # env.sigma = sigma
    env.reset()
    return S


# Note, this resets the environment
# Two-species BMSY function (not sure why the single-species one gives no problems though)
# def TS_BMSY(env):
#     n = 10001  # ick should  be cts
#     state_range = np.linspace(
#         env.observation_space.low,
#         env.observation_space.high,
#         num=n,
#         dtype=env.observation_space.dtype,
#     )
#     x_0 = np.asarray(list(map(env.get_fish_population, state_range)))
#     env.fish_population = x_0
#     sigma = env.sigma
#     env.sigma = 0
#     growth = env.population_draw() - x_0
#     S = x_0[np.argmax(growth)]
#     env.sigma = sigma
#     env.reset()
#     return S
