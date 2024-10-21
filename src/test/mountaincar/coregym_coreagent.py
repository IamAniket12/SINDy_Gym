import os
import gym
import sys
from stable_baselines3 import SAC
from pysindy.feature_library import *
from pysindy.differentiation import *

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
)
# Now you can import from src.environment.custom_gym_envs
from environment.mountain_car import (
    Continuous_MountainCarEnv,
    Continuous_MountainCarEnvWB,
    Continuous_MountainCarEnv_Sindy,
)


def run_trained_model(env_name, model_path, render_mode="human"):
    """summary: Run the trained model on the specified environment.
    Args:
        env_name (str): The Name of the gym environment
        model_path (str): Path to the trained model
         render_mode (str): The mode in which to render the environment (default is 'human').

    Returns:
        total_reward (float): The total reward achieved in the episode.
        step (int): The number of steps taken in the episode.
    """
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
    env_name = "MountainCarBB-v0"  # Example environment
    model_path = "results/models/mountain_car/sac_mountaincar_core.zip"

    run_trained_model(env_name, model_path)
