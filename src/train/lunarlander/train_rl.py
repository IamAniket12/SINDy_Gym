import os
import sys
import gym
import wandb
import numpy as np
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.callbacks import BaseCallback
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
from src.sindy.lunarlander.transition_model_ll import create_transition_function

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
)


class CustomWandbCallback(WandbCallback):
    """
    Custom callback to log episode rewards, lengths, and training loss using wandb.

    Args:
        verbose (int): Verbosity level for the callback (default is 0).

    Methods:
        _on_step(): Logs the training loss and episode rewards/lengths.
        _on_rollout_end(): Records the episode reward and length after each rollout.
    """

    def __init__(self, verbose=0):
        super(CustomWandbCallback, self).__init__(verbose=verbose, model_save_path=None)
        self.episode_rewards = []
        self.episode_lengths = []

    def _on_step(self) -> bool:
        """
        Log training loss and episode statistics (reward/length) during training.

        Returns:
            bool: True if logging was successful.
        """
        if len(self.model.logger.name_to_value) > 0:
            for key, value in self.model.logger.name_to_value.items():
                wandb.log({key: value}, step=self.num_timesteps)

        if len(self.episode_rewards) > 0:
            wandb.log(
                {
                    "episode_reward": sum(self.episode_rewards)
                    / len(self.episode_rewards),
                    "episode_length": sum(self.episode_lengths)
                    / len(self.episode_lengths),
                },
                step=self.num_timesteps,
            )
            self.episode_rewards = []
            self.episode_lengths = []

        return True

    def _on_rollout_end(self):
        if len(self.model.ep_info_buffer) > 0:
            self.episode_rewards.append(self.model.ep_info_buffer[-1]["r"])
            self.episode_lengths.append(self.model.ep_info_buffer[-1]["l"])


def initialize_wandb():
    """
    Initializes the wandb run for model training and logs hyperparameters.

    Returns:
        run (wandb.run): The initialized wandb run object.
    """
    return wandb.init(
        project="LunarLander_PPO_Sindy",
        config={
            "algorithm": "PPO",
            "n_envs": 16,
            "n_timesteps": 1000000,
            "policy": "MlpPolicy",
            "n_steps": 1024,
            "batch_size": 64,
            "gae_lambda": 0.98,
            "gamma": 0.999,
            "n_epochs": 4,
            "ent_coef": 0.01,
        },
        name="LunarLander_PPO_Sindy",
    )


def train_ppo_model(file_path, model_save_path, total_timesteps=100000):
    """
    Train an SAC model using a custom SINDy model for a MountainCar environment.

    Args:
        file_path (str): Path to the CSV file used for creating the SINDy transition model.
        model_save_path (str): Path to save the trained model.
        total_timesteps (int): Total number of timesteps for training (default is 100000).
    """
    # Create the SINDy model
    sindy_model = create_transition_function(
        file_path=file_path, agent=None, plot_predictions=False, interpret=False
    )

    # Create the environment with the SINDy model
    env = gym.make("LunarLander-Sindy-v0", sindy_model=sindy_model)

    # Initialize the SAC model
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

    # Create the custom wandb callback
    custom_wandb_callback = CustomWandbCallback()

    # Train the model
    model.learn(total_timesteps=total_timesteps, callback=custom_wandb_callback)

    # Save the model
    model.save(os.path.join(model_save_path, "ppo_lunarlander_sindy_v1"))

    # Close the environment
    env.close()


if __name__ == "__main__":
    # Ensure the logs folders exist
    if not os.path.exists("./logs"):
        os.makedirs("./logs")

    # Initialize wandb
    run = initialize_wandb()

    # Define file paths
    file_path = "src/data/lunarlander/lunar_lander_data.csv"
    model_save_path = "./results/models/mountain_car/"

    # Train the SAC model
    train_ppo_model(file_path, model_save_path)

    # Finish the wandb run
    wandb.finish()
