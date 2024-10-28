import sys
import os
import gym
from stable_baselines3 import SAC, PPO
from pysindy.feature_library import *
from pysindy.differentiation import *
from src.sindy.lunarlander.transition_model_ll import create_transition_function


def run_sindy_trained_model(
    file_path, model_path, env_name="LunarLander-v2", render_mode="human"
):
    """
    Run a trained SAC model on a Gym environment using a transition model created with SINDy.

    Args:
        file_path (str): Path to the CSV file used to generate the SINDy transition model.
        model_path (str): Path to the trained SAC model.
        env_name (str): The name of the Gym environment to be used (default is 'MountainCar_Sindy-v0').
        render_mode (str): The mode in which to render the environment (default is 'human').

    Returns:
        total_reward (float): The total reward achieved during the episode.
        step (int): The number of steps taken in the episode.
    """
    # Create the transition model using SINDy
    sindy_model = create_transition_function(
        file_path=file_path, agent=None, plot_predictions=False, interpret=False
    )

    # Create the environment with the SINDy model
    env = gym.make(env_name, render_mode=render_mode)

    # Load the trained model
    model = PPO.load(model_path)

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
    file_path = "src/data/lunarlander/lunar_lander_data.csv"
    model_path = "results/models/lunar_lander/ppo-LunarLander-sindy_v2.zip"

    run_sindy_trained_model(file_path, model_path)
