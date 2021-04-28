import gym
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env

import gym_fishing
from gym_fishing.envs.shared_env import (
    plot_mdp,
    plot_policyfn,
    simulate_mdp_vec,
)


def test_vectorized():
    env = make_vec_env("fishing-v1", n_envs=4)

    model = A2C("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=200)

    # Simulate a run with the trained model, visualize result
    eval_env = gym.make("fishing-v1")
    df = simulate_mdp_vec(env, eval_env, model, 5)
    plot_mdp(eval_env, df, "dqn-test.png")

    df = eval_env.policyfn(model)
    plot_policyfn(eval_env, df, "policy-vectorized-test.png")
