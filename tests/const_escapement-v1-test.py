import pandas as pd
import gym
import gym_fishing
import math

r = 0.3
K = 1
env = gym.make('fishing-v1', r = r, K = K, sigma = 0.)

## inits
row = []
rep = 0
## Simulate under MSY
env.reset()
obs = env.state

## Rewrite this as a custom `model` class that has a `predict` method for an obs

for t in range(env.Tmax):
  fish_population = env.get_fish_population(obs)
  Q = max(fish_population - K/2,0) 
  action = env.get_action(Q)
  quota =  env.get_quota(action)
  row.append([t, fish_population, quota, reward, int(rep)])
  obs, reward, done, info = env.step(action)
  
  
df = pd.DataFrame(row, columns=['time', 'state', 'action', 'reward', "rep"])
df


env.plot(df, "const_escapement.png")

