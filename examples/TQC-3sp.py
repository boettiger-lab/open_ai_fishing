import numpy as np
import pandas as pd
import gym
import gym_fishing
from sb3_contrib import TQC
from stable_baselines3.common.evaluation import evaluate_policy

# Intialize
env = gym.make("threeFishing-v2")
model = TQC("MlpPolicy", env, verbose=0)

# Train
model.learn(total_timesteps=200)
model.save("test-tqc-3sp")

# Evaluate
mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=50)
print("mean reward:", mean_reward, "std:", std_reward)

# Load a saved model
model.load("test-tqc-3sp")

# simulate
cols = ["t", "sp1", "sp2", "sp3"]
df = pd.DataFrame(None, columns = cols)
observation = env.reset()
for t in range(10):
  action, _ = model.predict(observation)
  observation, reward, terminated, info = env.step(action)
  df.loc[t+1] = np.append(t, observation)

df = df.melt(id_vars="t")

# Plot :-)
from plotnine import ggplot, geom_point, aes, geom_line
(ggplot(df, aes("t", "value", color="variable")) + geom_line())

