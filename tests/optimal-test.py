import gym
import gym_fishing
import numpy as np
from gym_fishing.models.policies import msy, escapement, user_action


env = gym.make('fishing-v1', r = 0.1, K = 1, sigma = 0.05)
model = msy(env)
df = env.simulate(model)
env.plot(df, "msy.png")


model = escapement(env)
df = env.simulate(model)
env.plot(df, "escapement.png")


model = user_action(env)
## Not run, require user input to test
# df = env.simulate(model)

env = gym.make('fishing-v0', r = 0.1, K = 1, sigma = 0.05)
model = msy(env)
df = env.simulate(model)
env.plot(df, "msy.png")


