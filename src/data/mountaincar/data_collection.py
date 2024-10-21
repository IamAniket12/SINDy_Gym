import gym
import numpy as np
import pandas as pd
import random
import os
import sys
from stable_baselines3 import SAC

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
)
from environment.mountain_car import (
    Continuous_MountainCarEnv,
    Continuous_MountainCarEnvWB,
    Continuous_MountainCarEnv_Sindy,
)


def initialize_environment(env_name="MountainCarBB-v0"):
    """
    Initialize the specified gym environment.

    Args:
        env_name (str): The environment name to initialize (default is 'MountainCarBB-v0').

    Returns:
        env (gym.Env): The initialized gym environment.
    """
    return gym.make(env_name)


def load_sac_agent(model_path):
    """
    Load the SAC agent from a given path.

    Args:
        model_path (str): Path to the saved SAC model.

    Returns:
        agent (SAC): The loaded SAC agent.
    """
    return SAC.load(model_path)


def epsilon_greedy_action(env, agent, current_obs, epsilon=0.2):
    """
    Choose an action based on an ε-greedy policy.

    Args:
        env (gym.Env): The environment object.
        agent (SAC): The SAC agent for deterministic action prediction.
        current_obs (array): The current observation state.
        epsilon (float): The probability of taking a random action (default is 0.2).

    Returns:
        action (array): The action chosen by the agent or randomly.
    """
    if random.random() < epsilon:
        # 20% chance of taking a random action
        return env.action_space.sample()
    else:
        # 80% chance of taking the best action (SAC agent)
        return agent.predict(current_obs, deterministic=True)[0]


def collect_episode_data(env, agent, num_steps=10000, epsilon=0.2):
    """
    Collect data from one episode in the environment using SAC and ε-greedy policy.

    Args:
        env (gym.Env): The environment object.
        agent (SAC): The SAC agent for deterministic action prediction.
        num_steps (int): The maximum number of steps to run in one episode.
        epsilon (float): The probability of taking a random action (default is 0.2).

    Returns:
        episode_data (list): A list of collected episode data.
    """
    episode_data = []
    current_obs = env.reset()[0]
    done = False
    num_steps_taken = 0

    while not done and num_steps_taken < num_steps:
        action = epsilon_greedy_action(env, agent, current_obs, epsilon)
        next_obs, reward, done, info, _ = env.step(action)

        # Append current observation, action, and next observation to the data list
        episode_data.append(
            (current_obs[0], current_obs[1], action[0], next_obs[0], next_obs[1], done)
        )

        # Update the current observation for the next step
        current_obs = next_obs
        num_steps_taken += 1

    return episode_data


def save_data_to_csv(data, file_path):
    """
    Save the collected data to a CSV file.

    Args:
        data (list): The collected episode data.
        file_path (str): The file path to save the CSV file.
    """
    df = pd.DataFrame(
        data,
        columns=[
            "current_pos",
            "current_vel",
            "action",
            "next_pos",
            "next_vel",
            "done",
        ],
    )
    df.to_csv(file_path, index=False)
    print(f"Data saved to {file_path}")


def main():
    # Initialize environment
    env = initialize_environment()

    # Load SAC agent
    agent = load_sac_agent("results/models/sac_mountaincar_core.zip")

    # Number of episodes to run
    num_episodes = 1

    # Initialize an empty list to store the data
    all_data = []

    # Collect data from the episodes
    for episode in range(num_episodes):
        episode_data = collect_episode_data(env, agent)
        all_data.extend(episode_data)

    # Define the file path to save the data
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_dir, "mountain_car_data.csv")

    # Save data to CSV
    save_data_to_csv(all_data, csv_path)

    # Close the environment
    env.close()


if __name__ == "__main__":
    main()
