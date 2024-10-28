import os
import sys
import gym
import wandb
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.evaluation import evaluate_policy
from src.sindy.lunarlander.transition_model_ll import create_transition_function
from environment.lunar_lander import LunarLander


def initialize_wandb():
    """
    Initializes the wandb run for model training and logs hyperparameters.

    Returns:
        run (wandb.run): The initialized wandb run object.
    """
    config = {
        "n_envs": 16,
        "n_timesteps": 1000000,
        "policy": "MlpPolicy",
        "n_steps": 1024,
        "batch_size": 64,
        "gae_lambda": 0.98,
        "gamma": 0.999,
        "n_epochs": 4,
        "ent_coef": 0.01,
    }

    return wandb.init(
        project="lunar-lander-ppo-sindy",
        config=config,
        sync_tensorboard=True,
        monitor_gym=True,
    )


def train_ppo_model(file_path, model_save_path, total_timesteps=1000000):
    """
    Train a PPO model using a custom SINDy model for LunarLander environment.

    Args:
        file_path (str): Path to the CSV file used for creating the SINDy transition model.
        model_save_path (str): Path to save the trained model.
        total_timesteps (int): Total number of timesteps for training.
    """
    # Create the SINDy model
    sindy_model = create_transition_function(
        file_path=file_path, agent=None, plot_predictions=False, interpret=False
    )

    # Create the environment with the SINDy model
    env = gym.make("LunarLander-Sindy-v0", sindy_model=sindy_model)

    # Initialize the PPO model with wandb config
    model = PPO(
        wandb.config.policy,
        env,
        n_steps=wandb.config.n_steps,
        batch_size=wandb.config.batch_size,
        gae_lambda=wandb.config.gae_lambda,
        gamma=wandb.config.gamma,
        n_epochs=wandb.config.n_epochs,
        ent_coef=wandb.config.ent_coef,
        verbose=1,
        tensorboard_log=f"runs/{wandb.run.id}",
    )

    # Set up the WandbCallback
    wandb_callback = WandbCallback(
        model_save_path=f"models/{wandb.run.id}",
        verbose=2,
        model_save_freq=10000,
    )

    # Train the model
    model.learn(
        total_timesteps=int(wandb.config.n_timesteps),
        callback=wandb_callback,
        tb_log_name="PPO",
    )

    # Save the final model
    model_name = os.path.join(model_save_path, "ppo-LunarLander-sindy_v2")
    model.save(model_name)

    # Evaluate the model
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

    # Log the final evaluation metrics
    wandb.log({"eval/mean_reward": mean_reward, "eval/std_reward": std_reward})

    # Close the environment
    env.close()


if __name__ == "__main__":
    # Ensure the logs and models folders exist
    os.makedirs("./logs", exist_ok=True)
    os.makedirs("./models", exist_ok=True)

    # Initialize wandb
    run = initialize_wandb()

    # Define file paths
    file_path = "src/data/lunarlander/lunar_lander_data.csv"
    model_save_path = "./results/models/lunar_lander"

    # Train the PPO model
    train_ppo_model(file_path, model_save_path)

    # Finish the wandb run
    wandb.finish()
