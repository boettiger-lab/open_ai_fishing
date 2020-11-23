import gym
import gym_fishing
from stable_baselines3.common.env_checker import check_env

env0 = gym.make('fishing-v0')
env1 = gym.make('fishing-v1')
env2 = gym.make('fishing-v2')
#env3 = gym.make('fishing-v3')
env4 = gym.make('fishing-v4')


check_env(env0)
check_env(env1)
check_env(env2)
#check_env(env3)
check_env(env4)

env0.close()
