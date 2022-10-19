import gym
import numpy as np
from stable_baselines3.common.env_checker import check_env

import gym_fishing
from gym_fishing.models.policies import escapement, msy, user_action, multiSpecies_singleHarvest_msy

np.random.seed(0)

def test_trophicTriangleRandEnv():
    env = gym.make("trophic-triangle-v1")
    check_env(env)
    model = multiSpecies_singleHarvest_msy(env)
    df = env.simulate(model)
    env.plot(df, "trophic-triangle-v1_msy-test.png")

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
