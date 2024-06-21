import gym
from stable_baselines3 import DQN,PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy

import gym_fishing


def test_dqn():
    env = gym.make("fishing-v1",init="random")
    check_env(env)
    
    model = PPO("MlpPolicy", env, verbose=0)
    #model = DQN("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=200)

    # Simulate a run with the trained model, visualize result
    df = env.simulate(model)
    env.plot(df, "PPO-test.png")
    #env.plot(df, "dqn-test.png")

    # Evaluate model
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=5)

    #df = env.policyfn(model)
    #env.plot_policy(df, "policy-test.png")
