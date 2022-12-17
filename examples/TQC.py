import numpy as np
import gym
import gym_fishing
from stable_baselines3 import PPO
from sb3_contrib import TQC
from stable_baselines3.common.evaluation import evaluate_policy

env = gym.make("fishing-v1")
model = TQC("MlpPolicy", env, verbose=0)
model.learn(total_timesteps=100000)

## simulate and plot results
df = env.simulate(model, reps=10)
env.plot(df, "tqc.png")
#env.plot_policyfn(df, "tqc_policy.png")
## Evaluate model
mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=50)
print("mean reward:", mean_reward, "std:", std_reward)

# save trained agent for future use, if desired
model.save("tqc")
