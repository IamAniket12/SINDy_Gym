import pandas as pd
import numpy as np
import gym
import seaborn as sns
from pysindy import SINDy
from pysindy.feature_library import CustomLibrary
from pysindy.differentiation import SmoothedFiniteDifference, FiniteDifference
from pysindy.optimizers import STLSQ
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from itertools import product
from tqdm import tqdm


def create_transition_function(
    file_path,
    agent=None,
    plot_predictions=False,
    interpret=False,
    tune_hyperparameters=False,
):
    """
    Create a SINDy model for predicting the next state of the Mountain Car environment.

    Args:
        file_path (str): Path to the CSV file containing the training data.
        agent (optional): Agent for interaction with the environment (default is None).
        plot_predictions (bool): If True, plot the predicted vs actual states (default is False).
        interpret (bool): If True, interpret the SINDy model and print the equations (default is False).
        tune_hyperparameters (bool): If True, perform hyperparameter tuning (default is False).

    Returns:
        model (SINDy): The trained SINDy model.
    """
    print("Creating the transition function")

    # Load the data
    data = pd.read_csv(file_path)

    # Prepare input (X) and output (u) data
    X = data[["x_pos", "y_pos", "x_vel", "y_vel", "angle", "angular_vel"]].values
    u = data["action"].values.reshape(-1, 1)
    episode_end = data["done"].values

    if tune_hyperparameters:
        # Define parameter grid for tuning
        param_grid = {
            "threshold": [0.001, 0.0015, 0.002, 0.0025, 0.003],
            "alpha": [0.001, 0.005, 0.01, 0.05, 0.1],
        }

        best_params, best_mse, model = hyperparameter_tuning(X, u, param_grid)
        print("\nUsing best parameters from hyperparameter tuning")

    else:
        # Use default parameters
        functions = [lambda x: 1, lambda x: x, lambda x: x**2]
        lib = CustomLibrary(library_functions=functions)
        optimizer = STLSQ(threshold=0.01, alpha=0.1)
        der = SmoothedFiniteDifference()
        model = SINDy(
            discrete_time=True,
            feature_library=lib,
            differentiation_method=der,
            optimizer=optimizer,
        )
        model.fit(X, u=u, t=0.0001)
        print("SINDy model fitted with default parameters")

    # Predict the next state
    X_pred = model.predict(X, u=u)

    # Update the positions based on predicted velocities
    x_pos_pred = X_pred[:, 0]
    y_pos_pred = X_pred[:, 1]
    x_vel_pred = X_pred[:, 2]
    y_vel_pred = X_pred[:, 3]
    angle_pred = X_pred[:, 4]
    angular_vel_pred = X_pred[:, 5]

    # Combine the predicted positions and velocities into a DataFrame
    x_pred = pd.DataFrame(
        {
            "x_pos_pred": x_pos_pred,
            "y_pos_pred": y_pos_pred,
            "x_vel_pred": x_vel_pred,
            "y_vel_pred": y_vel_pred,
            "angle_pred": angle_pred,
            "angular_vel_pred": angular_vel_pred,
        }
    )

    if plot_predictions:
        plot_state_transitions(file_path, x_pred)

    if interpret:
        interpret_sindy_model(model)

    # Calculate Mean Squared Error (MSE) between predicted and actual states
    mse = mean_squared_error(X[1:], X_pred[:-1])
    print(f"Mean Squared Error: {mse}")

    return model


def hyperparameter_tuning(X, u, param_grid):
    """
    Perform grid search to find optimal hyperparameters for the SINDy model.

    Args:
        X (numpy.ndarray): Input state variables
        u (numpy.ndarray): Action variables
        param_grid (dict): Dictionary containing parameter ranges to search
            Example: {
                'threshold': [0.001, 0.005, 0.01],
                'alpha': [0.001, 0.01, 0.1]
            }

    Returns:
        tuple: (best_params, best_mse, best_model)
            - best_params (dict): Dictionary of best parameters
            - best_mse (float): MSE score with best parameters
            - best_model (SINDy): Best performing model
    """
    best_mse = float("inf")
    best_params = None
    best_model = None

    # Create parameter combinations
    param_combinations = [
        dict(zip(param_grid.keys(), v)) for v in product(*param_grid.values())
    ]

    print(f"Testing {len(param_combinations)} parameter combinations...")

    # Progress bar for parameter search
    for params in tqdm(param_combinations):
        # Define the model with current parameters
        functions = [lambda x: 1, lambda x: x, lambda x: x**2, lambda x: x**3]
        lib = CustomLibrary(library_functions=functions)
        optimizer = STLSQ(threshold=params["threshold"], alpha=params["alpha"])
        der = FiniteDifference()
        model = SINDy(
            discrete_time=True,
            feature_library=lib,
            differentiation_method=der,
            optimizer=optimizer,
        )

        # Fit the model
        try:
            model.fit(X, u=u, t=0.0001)

            # Make predictions
            X_pred = model.predict(X, u=u)

            # Calculate MSE
            mse = mean_squared_error(X[1:], X_pred[:-1])

            # Update best parameters if current MSE is lower
            if mse < best_mse:
                best_mse = mse
                best_params = params
                best_model = model

        except Exception as e:
            print(f"Error with parameters {params}: {str(e)}")
            continue

    print("\nBest parameters found:")
    for param, value in best_params.items():
        print(f"{param}: {value}")
    print(f"Best MSE: {best_mse}")

    return best_params, best_mse, best_model


def plot_state_transitions(file_path, x_pred):
    """
    Plot actual vs. predicted distributions for all state variables with improved layout.

    Args:
        file_path (str): Path to the CSV file with actual data.
        x_pred (DataFrame): DataFrame containing predicted state variables.
    """
    # Load the actual data
    data = pd.read_csv(file_path)

    # Define the variables to plot
    variables = {
        "x_pos": ("x_pos_pred", "Position X"),
        "y_pos": ("y_pos_pred", "Position Y"),
        "x_vel": ("x_vel_pred", "Velocity X"),
        "y_vel": ("y_vel_pred", "Velocity Y"),
        "angle": ("angle_pred", "Angle"),
        "angular_vel": ("angular_vel_pred", "Angular Velocity"),
    }

    # Create figure with more height to accommodate subplots
    fig = plt.figure(figsize=(22, 24))
    plt.suptitle(
        "Distribution and Scatter Plots of Actual vs Predicted States",
        fontsize=20,
        y=0.95,
    )

    # Plot each variable with adjusted spacing
    for idx, (actual_col, (pred_col, title)) in enumerate(variables.items(), 1):
        if actual_col in data.columns and pred_col in x_pred.columns:
            # 1. Distribution plot
            ax1 = plt.subplot(len(variables), 3, (idx - 1) * 3 + 1)
            sns.kdeplot(data=data[actual_col], label="Actual", alpha=0.6)
            sns.kdeplot(data=x_pred[pred_col], label="Predicted", alpha=0.6)
            plt.title(f"{title} Distribution", pad=20, fontsize=14)
            plt.xlabel("Value", fontsize=12)
            plt.ylabel("Density", fontsize=12)
            plt.legend(fontsize=10)
            plt.xticks(fontsize=10)
            plt.yticks(fontsize=10)

            # 2. Scatter plot
            ax2 = plt.subplot(len(variables), 3, (idx - 1) * 3 + 2)
            plt.scatter(data[actual_col], x_pred[pred_col], alpha=0.1, s=1)
            plt.plot(
                [data[actual_col].min(), data[actual_col].max()],
                [data[actual_col].min(), data[actual_col].max()],
                "r--",
                alpha=0.8,
            )
            plt.title(f"{title} Actual vs Predicted", pad=20, fontsize=14)
            plt.xlabel("Actual", fontsize=12)
            plt.ylabel("Predicted", fontsize=12)
            plt.xticks(fontsize=10)
            plt.yticks(fontsize=10)

            # 3. Error distribution
            ax3 = plt.subplot(len(variables), 3, (idx - 1) * 3 + 3)
            errors = x_pred[pred_col] - data[actual_col]
            sns.histplot(errors, kde=True, bins=30)
            plt.axvline(x=0, color="r", linestyle="--", alpha=0.8)
            plt.title(f"{title} Prediction Error Distribution", pad=20, fontsize=14)
            plt.xlabel("Error (Predicted - Actual)", fontsize=12)
            plt.ylabel("Count", fontsize=12)
            plt.xticks(fontsize=10)
            plt.yticks(fontsize=10)

            # Calculate and display metrics in a better position
            mse = mean_squared_error(data[actual_col], x_pred[pred_col])
            mae = np.mean(np.abs(errors))
            stats_text = f"MSE: {mse:.2e}\nMAE: {mae:.2e}"
            plt.text(
                0.98,
                0.98,
                stats_text,
                transform=ax3.transAxes,
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
                fontsize=10,
                verticalalignment="top",
                horizontalalignment="right",
            )

            # Add gridlines for better readability
            ax1.grid(True, alpha=0.3)
            ax2.grid(True, alpha=0.3)
            ax3.grid(True, alpha=0.3)

    # Adjust layout with more space between subplots
    plt.tight_layout(rect=[0, 0.03, 1, 0.95], h_pad=4.0, w_pad=3.0)
    plt.show()

    # Create correlation heatmap with improved layout
    plt.figure(figsize=(12, 8))
    actual_data = data[[col for col in variables.keys()]]
    predicted_data = x_pred[[col[0] for col in variables.values()]]

    # Rename predicted columns to match actual for correlation
    predicted_data.columns = actual_data.columns

    # Calculate correlations between actual and predicted
    correlations = pd.DataFrame(index=actual_data.columns, columns=["Correlation"])
    for col in actual_data.columns:
        correlations.loc[col, "Correlation"] = actual_data[col].corr(
            predicted_data[col]
        )

    # Plot correlation heatmap with improved formatting
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        correlations,
        annot=True,
        cmap="RdYlBu",
        center=0,
        vmin=-1,
        vmax=1,
        fmt=".3f",
        annot_kws={"size": 12},
    )
    plt.title("Correlation between Actual and Predicted Values", pad=20, fontsize=16)
    plt.ylabel("State Variables", fontsize=12)
    plt.tick_params(axis="both", which="major", labelsize=10)
    plt.tight_layout()
    plt.show()

    # Print summary statistics with improved formatting
    print("\nSummary Statistics:")
    print("=" * 50)
    stats_df = pd.DataFrame(columns=["MSE", "MAE", "Correlation"])

    for actual_col, (pred_col, title) in variables.items():
        if actual_col in data.columns and pred_col in x_pred.columns:
            mse = mean_squared_error(data[actual_col], x_pred[pred_col])
            mae = np.mean(np.abs(x_pred[pred_col] - data[actual_col]))
            correlation = data[actual_col].corr(x_pred[pred_col])
            stats_df.loc[title] = [mse, mae, correlation]

    # Format the statistics DataFrame
    pd.set_option(
        "display.float_format",
        lambda x: "{:.2e}".format(x) if abs(x) < 0.01 else "{:.3f}".format(x),
    )
    print(stats_df)
    print("=" * 50)


# [Rest of the code remains the same]


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
    file_path = "src/data/lunarlander/lunar_lander_data.csv"
    model = create_transition_function(
        file_path, plot_predictions=True, interpret=True, tune_hyperparameters=True
    )
