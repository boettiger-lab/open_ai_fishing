import gym
import numpy as np
from stable_baselines3.common.env_checker import check_env

import gym_fishing
from gym_fishing.models.policies import (
    escapement,
    msy,
    periodic_full_harvest,
    user_action,
)

np.random.seed(0)


def test_trophicTriangleJConstEnv():
    env = gym.make("trophictriangle-v2")
    check_env(env)
    model = periodic_full_harvest(env)
    opt_harvest_t, opt_harvest = model.opt_harvest_timing(env, t_scale=1000)
    print(
        "opt harvest time = {}, yields {} per unit time".format(
            opt_harvest_t, opt_harvest
        )
    )


def test_trophicTriangleRandEnv():
    env = gym.make("trophictriangle-v1")
    check_env(env)
    # model = multiSpecies_singleHarvest_msy(env)
    # df = env.simulate(model)
    # env.plot(df, "trophictriangle-v1_msy-test.png")


def test_twospecies_fishing():
    env = gym.make("tsfishing-v0")
    check_env(env)
    # model = msy(env)
    # df = env.simulate(model)
    # env.plot(df, "tsfishing-v0_msy-test.png")
    # model = escapement(env)
    # df = env.simulate(model)
    # env.plot(df, "tsfishing-v0_escapement-test.png")


def test_discrete():
    env = gym.make("fishing-v0")
    check_env(env)
    model = msy(env)
    df = env.simulate(model)
    env.plot(df, "v0_msy-test.png")
    model = escapement(env)
    df = env.simulate(model)
    env.plot(df, "v0_escapement-test.png")


def test_cts():
    env = gym.make("fishing-v1")
    check_env(env)
    model = msy(env)
    df = env.simulate(model)
    env.plot(df, "v1_msy-test.png")
    model = escapement(env)
    df = env.simulate(model)
    env.plot(df, "v1_escapement-test.png")


def test_allen():
    env = gym.make("fishing-v5", sigma=0)
    check_env(env)
    # model = user_action(env)
    model = msy(env)
    df = env.simulate(model)
    env.plot(df, "allen_msy-test.png")
    model = escapement(env)
    df = env.simulate(model)
    env.plot(df, "allen_escapement-test.png")


def test_beverton_holt():
    env = gym.make("fishing-v6", sigma=0)
    check_env(env)
    # model = user_action(env)
    model = msy(env)
    df = env.simulate(model)
    env.plot(df, "bh_msy-test.png")
    model = escapement(env)
    df = env.simulate(model)
    env.plot(df, "bh_escapement-test.png")


def test_may():
    env = gym.make("fishing-v7", sigma=0)
    check_env(env)
    #  model = user_action(env)
    model = msy(env)
    df = env.simulate(model)
    env.plot(df, "may_msy-test.png")
    model = escapement(env)
    df = env.simulate(model)
    env.plot(df, "may_escapement-test.png")


def test_myers():
    env = gym.make("fishing-v8", sigma=0)
    check_env(env)
    model = user_action(env)
    model = msy(env)
    df = env.simulate(model)
    env.plot(df, "myers_msy-test.png")
    model = escapement(env)
    df = env.simulate(model)
    env.plot(df, "myers_escapement-test.png")


def test_ricker():
    env = gym.make("fishing-v9", sigma=0)
    check_env(env)
    # model = user_action(env)
    model = msy(env)
    df = env.simulate(model)
    env.plot(df, "ricker_msy-test.png")
    model = escapement(env)
    df = env.simulate(model)
    env.plot(df, "ricker_escapement-test.png")


def test_tipping():
    np.random.seed(0)
    env = gym.make("fishing-v2", sigma=0, init_state=0.75)
    check_env(env)
    env.reset()
    # increases above tipping point
    obs, reward, done, info = env.step(env.get_action(0))
    assert env.get_fish_population(obs) >= 0.75

    # Decreases below the tipping point
    env.init_state = 0.3
    env.reset()
    obs, reward, done, info = env.step(env.get_action(0))
    assert env.get_fish_population(obs) <= 0.3

    # model = user_action(env)
    model = msy(env)
    df = env.simulate(model)
    env.plot(df, "tip_msy-test.png")
    model = escapement(env)
    df = env.simulate(model)
    env.plot(df, "tip_escapement-test.png")


def test_nonstationary():
    env = gym.make("fishing-v10", sigma=0, alpha=-0.007)
    check_env(env)
    # model = user_action(env)
    model = msy(env)
    df = env.simulate(model)
    env.plot(df, "ns_msy-test.png")
    model = escapement(env)
    df = env.simulate(model)
    env.plot(df, "ns_escapement-test.png")


def test_model_uncertainty():
    np.random.seed(0)
    env = gym.make("fishing-v11")
    check_env(env)
    model = user_action(env)
    model = msy(env)
    df = env.simulate(model, reps=10)
    env.plot(df, "mu_msy-test.png")
    model = escapement(env)
    df = env.simulate(model, reps=10)
    env.plot(df, "mu_escapement-test.png")
