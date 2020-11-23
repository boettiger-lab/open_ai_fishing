import numpy as np
import gym
import gym_fishing
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

env = gym.make('fishing-v0')
model = PPO('MlpPolicy', env, verbose=0)
model.learn(total_timesteps=100000)

## simulate and plot results
df = env.simulate(model, reps=10)
env.plot(df, "results/ppo.png")


## Evaluate model
mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=50)
print("mean reward:", mean_reward, "std:", std_reward)


model.save("results/ppo")
