from csv import writer

import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame


def csv_entry(self):
    row_contents = [self.years_passed, self.state[0], self.action, self.reward]
    csv_writer = writer(self.write_obj)
    csv_writer.writerow(row_contents)
    return row_contents


def df_entry_vec(df, env, rep, obs, action, reward, t):
    for i, e in enumerate(range(len(df.index), len(df.index) + env.num_envs)):
        fish_population = env.env_method(
            "get_fish_population", (obs[i],), indices=i
        )[0][0]
        df.loc[e] = [
            t,
            fish_population,
            action[i][0],
            reward[i],
            (rep * env.num_envs) + i,
        ]


def simulate_mdp(env, model, reps=1):
    row = []
    for rep in range(reps):
        obs = env.reset()
        quota = 0.0
        reward = 0.0
        for t in range(env.Tmax):
            # record
            fish_population = env.get_fish_population(obs)
            row.append([t, fish_population, quota, reward, int(rep)])

            # Predict and implement action
            action, _state = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)

            # discrete actions are not arrays, but cts actions are
            if isinstance(action, np.ndarray):
                action = action[0]
            if isinstance(reward, np.ndarray):
                reward = reward[0]
            quota = env.get_quota(action)

            if done:
                break
    df = DataFrame(row, columns=["time", "state", "action", "reward", "rep"])
    return df


def simulate_mdp_vec(env, model, n_eval_episodes):
    assert (
        n_eval_episodes % env.num_envs == 0
    ), "Error: number of \
                     evaluations needs to be divisible by the number \
                     of parallel environments"

    # Since performed in parallel, need to account for number of envs
    reps = int(n_eval_episodes / env.num_envs)
    df = DataFrame(columns=["time", "state", "action", "reward", "rep"])
    for rep in range(reps):
        obs = env.reset()
        state = None
        done = [False for _ in range(env.num_envs)]
        action = [[env.action_space.low[0]] for _ in range(env.num_envs)]
        reward = [0 for _ in range(env.num_envs)]
        for t in range(env.get_attr("Tmax")[0]):
            df_entry_vec(df, env, rep, obs, action, reward, t)
            action, state = model.predict(
                obs, state=state, mask=done, deterministic=True
            )
            obs, reward, done, info = env.step(action)
        df_entry_vec(df, env, rep, obs, action, reward, t + 1)

    return df


def estimate_policyfn(env, model, reps=1, n=50):
    row = []
    state_range = np.linspace(
        env.observation_space.low,
        env.observation_space.high,
        num=n,
        dtype=env.observation_space.dtype,
    )
    for rep in range(reps):
        for obs in state_range:
            action, _state = model.predict(obs, deterministic=True)
            if isinstance(action, np.ndarray):
                action = action[0]

            fish_population = env.get_fish_population(obs)
            quota = env.get_quota(action)

            row.append([fish_population, quota, rep])

    df = DataFrame(row, columns=["state", "action", "rep"])
    return df


def plot_mdp(self, df, output="results.png"):
    fig, axs = plt.subplots(3, 1)
    for i in np.unique(df.rep):
        results = df[df.rep == i]
        episode_reward = np.cumsum(results.reward)
        axs[0].plot(results.time, results.state, color="blue", alpha=0.3)
        axs[1].plot(results.time, results.action, color="blue", alpha=0.3)
        axs[2].plot(results.time, episode_reward, color="blue", alpha=0.3)

    axs[0].set_ylabel("state")
    axs[1].set_ylabel("action")
    axs[2].set_ylabel("reward")
    fig.tight_layout()
    plt.savefig(output)
    plt.close("all")


def plot_policyfn(self, df, output="policy.png"):
    for i in np.unique(df.rep):
        results = df[df.rep == i]
        plt.plot(results.state, results.state - results.action, color="blue")
    plt.savefig(output)
    plt.close("all")
