import os
import sys
import gym
from stable_baselines3 import SAC
from pysindy.feature_library import *
from pysindy.differentiation import *
from environment.mountain_car import (
    Continuous_MountainCarEnv,
    Continuous_MountainCarEnvWB,
    Continuous_MountainCarEnv_Sindy,
)
from src.sindy.mountaincar.transition_model_mc import create_transition_function


def run_trained_model_with_sindy(
    file_path, model_path, env_name="MountainCarBB-v0", render_mode="human"
):
    """
    Run a trained SAC model on a Gym environment with a transition model created using SINDy.

    Args:
        file_path (str): Path to the data file used for generating the transition model.
        model_path (str): Path to the trained SAC model.
        env_name (str): The name of the gym environment (default is 'MountainCarBB-v0').
        render_mode (str): The mode in which to render the environment (default is 'human').

    Returns:
        total_reward (float): The total reward achieved in the episode.
        step (int): The number of steps taken in the episode.
    """
    # Function to create the SINDy transition model
    model = create_transition_function(
        file_path=file_path, agent=None, plot_predictions=False, interpret=False
    )

    # Create the environment
    env = gym.make(env_name, render_mode=render_mode)

    # Load the trained model
    model = SAC.load(model_path)

    # Run the trained model
    obs = env.reset()[0]
    done = False
    total_reward = 0
    step = 0

    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        step += 1
        env.render()

    print(f"Episode finished after {step} steps")
    print(f"Total reward: {total_reward:.4f}")

    # Close the environment
    env.close()

    return total_reward, step


if __name__ == "__main__":
    file_path = "src/data/mountaincar/mountain_car_data.csv"
    model_path = "results/models/mountain_car/sac_mountaincar_sindy_v1.zip"

    run_trained_model_with_sindy(file_path, model_path)
