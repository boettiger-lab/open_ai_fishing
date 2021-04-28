from csv import writer

import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame, Series


def csv_entry(env):
    row_contents = [env.years_passed, env.state[0], env.action, env.reward]
    csv_writer = writer(env.write_obj)
    csv_writer.writerow(row_contents)
    return row_contents


def df_entry_vec(df, rep, obs, action, reward, t):
    # Appending entry to the dataframe
    series = Series(
        [t, obs[0][0], action[0][0], reward[0], rep], index=df.columns
    )
    return df.append(series, ignore_index=True)


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
            action, _state = model.predict(obs)
            obs, reward, done, info = env.step(action)

            # discrete actions are not arrays, but cts actions are
            if isinstance(action, np.ndarray):
                action = action[0]
            if isinstance(reward, np.ndarray):
                reward = reward[0]
            quota = env.get_quota(action)

            if done:
                break
        # Recording the last step
        fish_population = env.get_fish_population(obs)
        row.append([t + 1, fish_population, quota, reward, int(rep)])
    df = DataFrame(row, columns=["time", "state", "action", "reward", "rep"])
    return df


def simulate_mdp_vec(env, eval_env, model, n_eval_episodes):
    # A big issue with evaluating a vectorized environment is that
    # SB automatically resets an environment after a done flag.
    # By automatic resetting, you lose the final state.
    # To workaround this I have a single evaluation environment
    # that I run in parallel to the vectorized env.
    reps = int(n_eval_episodes)
    df = DataFrame(columns=["time", "state", "action", "reward", "rep"])
    for rep in range(reps):
        # Creating the 2 environments
        e_obs = eval_env.reset()
        obs = env.reset()
        # Passing first obs from eval env into first index
        obs[0] = e_obs
        # Initializing variables
        state = None
        done = [False for _ in range(env.num_envs)]
        action = [[env.action_space.low[0]] for _ in range(env.num_envs)]
        reward = [0 for _ in range(env.num_envs)]
        t = 0
        while True:
            df = df_entry_vec(df, rep, obs, action, reward, t)
            t += 1
            # Using the vec env to do predictions
            action, state = model.predict(obs, state=state, mask=done)
            obs, reward, done, info = env.step(action)
            # Stepping the eval env along with the vec env
            e_obs, e_reward, e_done, e_info = eval_env.step(action[0])
            # Passing the evaluation env in for the first vec env's
            # observations. This is to avoid automatic resetting when
            # `done=True` which is a constraint of vectorized environments.
            # Unfortunately, a recurrent trained agent must be evaluated on
            # the number of vectorized envs it was trained on.
            obs[0] = e_obs
            if e_done:
                break
        df = df_entry_vec(df, rep, obs, action, reward, t)

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
            action, _state = model.predict(obs)

            fish_population = env.get_fish_population(obs)
            quota = env.get_quota(action)

            row.append([fish_population, quota, rep])

    df = DataFrame(row, columns=["state", "action", "rep"])
    return df


def plot_mdp(env, df, output="results.png"):
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


def plot_policyfn(env, df, output="policy.png"):
    for i in np.unique(df.rep):
        results = df[df.rep == i]
        plt.plot(results.state, results.state - results.action, color="blue")
    plt.savefig(output)
    plt.close("all")
