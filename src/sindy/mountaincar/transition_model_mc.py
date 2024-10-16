import pandas as pd
import numpy as np
import gym
from pysindy import SINDy
from pysindy.feature_library import CustomLibrary
from pysindy.differentiation import SmoothedFiniteDifference
from pysindy.optimizers import STLSQ
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


def create_transition_function(
    file_path, agent=None, plot_predictions=False, interpret=False
):
    """
    Create a SINDy model for predicting the next state of the Mountain Car environment.

    Args:
        file_path (str): Path to the CSV file containing the training data.
        agent (optional): Agent for interaction with the environment (default is None).
        plot_predictions (bool): If True, plot the predicted vs actual states (default is False).
        interpret (bool): If True, interpret the SINDy model and print the equations (default is False).

    Returns:
        model (SINDy): The trained SINDy model.
    """
    print("Creating the transition function")

    # Load the data
    data = pd.read_csv(file_path)

    # Prepare input (X) and output (u) data
    X = data[
        ["current_pos", "current_vel"]
    ].values  # State variables: position and velocity
    u = data["action"].values.reshape(-1, 1)  # Action taken
    episode_end = data["done"].values  # Indicates the end of an episode

    # Define the environment (for possible future use)
    temp = gym.make("MountainCarBB-v0")

    # Define SINDy library functions
    functions = [
        lambda x: 1,  # Constant
        lambda x: x,  # Linear (e.g., position)
        lambda x: np.cos(3 * x),  # Nonlinear term
    ]

    # Create a custom library for the SINDy model
    lib = CustomLibrary(library_functions=functions)

    # Define the SINDy model and the optimizer
    optimizer = STLSQ(threshold=0.0019, alpha=0.01)
    der = SmoothedFiniteDifference()
    model = SINDy(
        discrete_time=True,
        feature_library=lib,
        differentiation_method=der,
        optimizer=optimizer,
    )

    # Fit the model to the data
    model.fit(X, u=u, t=0.0001)
    print("SINDy model fitted")

    # Predict the next state
    X_pred = model.predict(X, u=u)

    # Update the positions based on predicted velocities
    predicted_positions = X_pred[:, 0]  # Predicted positions
    predicted_velocities = X_pred[:, 1]  # Predicted velocities
    for i in range(1, len(predicted_positions)):
        if episode_end[i - 1]:  # Reset position at the end of an episode
            predicted_positions[i] = X[i, 0]  # Use original starting position
        else:
            predicted_positions[i] = (
                predicted_positions[i - 1] + predicted_velocities[i - 1]
            )

    # Combine the predicted positions and velocities into a DataFrame
    x_pred = pd.DataFrame(
        {"current_pos": predicted_positions, "current_vel": predicted_velocities}
    )

    # Plot predictions if requested
    if plot_predictions:
        plot_predictions("mountain_car_data.csv", x_pred)

    # Interpret the SINDy model if requested
    if interpret:
        interpret_sindy_model(model)

    # Calculate Mean Squared Error (MSE) between predicted and actual states
    mse = mean_squared_error(X[1:], X_pred[:-1])
    print(f"Mean Squared Error: {mse}")

    # Return the trained SINDy model
    return model


def plot_predictions(file_path, x_pred):
    """
    Plot actual vs. predicted positions and velocities from a CSV file.

    Args:
        file_path (str): Path to the CSV file with actual data.
        x_pred (DataFrame): DataFrame containing predicted positions and velocities.
    """
    # Load the actual data
    data = pd.read_csv(file_path)

    # Check if relevant columns exist
    if "current_pos" in data.columns and "current_vel" in data.columns:
        plt.figure(figsize=(10, 6))
        # Scatter plot for actual data
        plt.scatter(
            data["current_pos"],
            data["current_vel"],
            alpha=0.5,
            c="blue",
            label="Actual Data",
            edgecolor="k",
        )

        # Overlay predicted data
        plt.scatter(
            x_pred["current_pos"],
            x_pred["current_vel"],
            alpha=0.7,
            c="red",
            label="Predicted Data",
            edgecolor="k",
        )

        # Customize the plot
        plt.title("Scatter Plot of Current Position vs. Current Velocity")
        plt.xlabel("Current Position")
        plt.ylabel("Current Velocity")
        plt.grid(True)
        plt.axhline(
            0, color="black", linewidth=0.8, linestyle="--"
        )  # Add horizontal line at y=0
        plt.axvline(
            0, color="black", linewidth=0.8, linestyle="--"
        )  # Add vertical line at x=0
        plt.legend()
        plt.show()
    else:
        print("Columns 'current_pos' or 'current_vel' not found in the CSV file.")


def interpret_sindy_model(model):
    """
    Interpret and print the equations of the SINDy model in human-readable format.

    Args:
        model (SINDy): The fitted SINDy model to interpret.
    """
    functions = [
        "1",  # f0: constant term
        "x",  # f1: linear term (position or velocity)
        "cos(3 * x)",  # f2: nonlinear term
    ]

    # Extract and interpret the model equations
    equations = model.equations()
    for i, eq in enumerate(equations):
        state_var = "position" if i == 0 else "velocity"
        for j, func in enumerate(functions):
            eq = eq.replace(f"f{j}(x0[k])", f"{func}(position_t)")
            eq = eq.replace(f"f{j}(x1[k])", f"{func}(velocity_t)")
            eq = eq.replace(f"f{j}(u0[k])", "action_t")

        print(f"Equation for {state_var}_t+1:\n{eq}\n")


if __name__ == "__main__":
    # Example usage
    file_path = "src/data/mountaincar/mountain_car_data.csv"
    model = create_transition_function(file_path, plot_predictions=True, interpret=True)
