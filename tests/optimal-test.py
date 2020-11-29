import gym
import gym_fishing
import numpy as np
from gym_fishing.models.policies import msy, escapement, user_action


def run_optimal(env_name, r=0.1, K=1, sigma=0.01):
    '''
    :param env_name: 'v0','v1', 'v2','v4'
    :param r:
    :param K:
    :param sigma:
    :return:
    '''
    if env_name != 'v4':
        env = gym.make('fishing-' + env_name, r=r, K=K, sigma=sigma)
    else:
        env = gym.make('fishing-' + env_name, sigma=sigma)

    model = msy(env)
    df = env.simulate(model)
    env.plot(df, "msy-" + env_name + ".png")

    model = escapement(env)
    df = env.simulate(model)
    env.plot(df, "escapement-" + env_name + ".png")
    return


run_optimal('v0')
run_optimal('v1')
run_optimal('v2')
run_optimal('v4')
# model = user_action(env)
# Not run, require user input to test
# df = env.simulate(model)
'''
env = gym.make('fishing-v1', r=0.1, K=1, sigma=0.01)

model = msy(env)
df = env.simulate(model)
env.plot(df, "msy.png")

model = escapement(env)
df = env.simulate(model)
env.plot(df, "escapement.png")




env = gym.make('fishing-v2', r=0.1, K=1, sigma=0.01)
model = msy(env)
df = env.simulate(model)
env.plot(df, "msy-v2.png")

model = escapement(env)
df = env.simulate(model)
env.plot(df, "escapement-v2.png")
'''

