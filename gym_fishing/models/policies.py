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


class periodic_full_harvest:
    def __init__(self, env):
        self.env = env
        env.fish_population[0] = 0
        self.total_harvest = 0

    def harvest(self, env, t_harvest, deterministic=True):
        if deterministic and hasattr(env, "sigma"):
            sigmaArr = env.turn_noise_off()
        for t in range(t_harvest):
            env.population_draw()
        self.total_harvest += env.fish_population[0]
        env.fish_population[0] = 0.0
        env.turn_noise_on(sigmaArr)
        # return the harvest per unit time
        return self.total_harvest / t_harvest

    def opt_harvest_timing(self, env, t_scale):
        max_t_harvest = 1
        max_norm_harvest = 0
        for t_harvest in range(1, t_scale + 1):
            norm_harvest = self.harvest(env, t_harvest)
            if norm_harvest > max_norm_harvest:
                max_norm_harvest = norm_harvest
                max_t_harvest = t_harvest
        return max_t_harvest, max_norm_harvest

    def norm_yields(self, env, t_scale):
        harvest_times = []
        unit_time_yields = []
        for t_harvest in range(t_scale):
            harvest_times.append(t_harvest)
            unit_time_yields.append()
        return harvest_times, unit_time_yields


# Obsolete
#
# class multiSpecies_singleHarvest_msy:
#     def __init__(self, env, **kwargs):
#         self.env = env
#         self.S = multiSpecies_singleHarvestBMSY(env)
#         # use the built-in method to determine MSY
#         env.fish_population = self.S
#         if hasattr(env, "sigma"):
#             sigmaArr = env.turn_noise_off()
#         # sigma = env.sigma
#         # env.sigma = 0
#
#         self.msy = env.population_draw() - self.S
#         if hasattr(env, "sigma"):
#             env.turn_noise_on(sigmaArr)
#         # env.sigma = sigma
#         env.reset()
#
#     def predict(self, obs, **kwargs):
#         msy = self.msy
#         action = self.env.get_action(msy)
#         return action, obs


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
    n = 101  # (replacing it manually in grid :( ) ick should  be cts
    env.reset()
    varnames = env.variable_names()
    # TBD: can we do this for general n species nicely?

    # manual for now:
    state_space = np.mgrid[-1:1.1:101j, -1:1.1:101j, -1:1.1:101j]
    states_A = np.array(
        state_space[0], dtype=np.float32
    ).flatten()  # used to be just state_space[0]
    states_F = np.array(state_space[1], dtype=np.float32).flatten()
    states_J = np.array(state_space[2], dtype=np.float32).flatten()
    # Flatten them for a single loop
    state_range = [
        np.array([states_A[i], states_F[i], states_J[i]], dtype=np.float32)
        for i in range(n)
    ]

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
