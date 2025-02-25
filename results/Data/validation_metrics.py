import numpy as np
from sklearn.metrics import mean_squared_error
import pandas as pd
import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
        )
    )
)
from src.sindy.lunarlander.transition_model_ll import create_transition_function

# from src.sindy.mountaincar.transition_model_mc import create_transition_function


def calculate_validation_metrics(actual_data, model, env_type="mountain_car"):
    """
    Calculate MSE and state-wise correlations between actual and predicted states.
    """
    metrics = {}

    if env_type == "mountain_car":
        # Get states and actions
        states = actual_data[["current_pos", "current_vel"]].values
        actions = actual_data["action"].values.reshape(-1, 1)

        # Get predictions
        predicted_states = model.predict(states, u=actions)
        predicted_data = pd.DataFrame(
            predicted_states, columns=["current_pos_pred", "current_vel_pred"]
        )

        state_vars = {
            "Position": ["current_pos", "current_pos_pred"],
            "Velocity": ["current_vel", "current_vel_pred"],
        }
    else:  # lunar_lander
        # Get states and actions
        states = actual_data[
            ["x_pos", "y_pos", "x_vel", "y_vel", "angle", "angular_vel"]
        ].values
        actions = actual_data["action"].values.reshape(-1, 1)

        # Get predictions
        predicted_states = model.predict(states, u=actions)
        predicted_data = pd.DataFrame(
            predicted_states,
            columns=[
                "x_pos_pred",
                "y_pos_pred",
                "x_vel_pred",
                "y_vel_pred",
                "angle_pred",
                "angular_vel_pred",
            ],
        )

        state_vars = {
            "x_position": ["x_pos", "x_pos_pred"],
            "y_position": ["y_pos", "y_pos_pred"],
            "x_velocity": ["x_vel", "x_vel_pred"],
            "y_velocity": ["y_vel", "y_vel_pred"],
            "angle": ["angle", "angle_pred"],
            "angular_velocity": ["angular_vel", "angular_vel_pred"],
        }

    print(f"\nValidation Metrics for {env_type}:")
    print("-" * 50)
    print(f"{'State Variable':<20} {'MSE':<15} {'Correlation':<15}")
    print("-" * 50)

    for state_name, (actual_col, pred_col) in state_vars.items():
        # Calculate MSE
        mse = mean_squared_error(actual_data[actual_col], predicted_data[pred_col])

        # Calculate correlation
        correlation = np.corrcoef(actual_data[actual_col], predicted_data[pred_col])[
            0, 1
        ]

        metrics[state_name] = {"mse": mse, "correlation": correlation}

        # Fixed formatting
        print(f"{state_name:<20} {mse:<15.2e} {correlation:<15.3f}")

    # Generate LaTeX table format
    print("\nLaTeX Table Format:")
    print(r"\begin{tabular}{lcc}")
    print(r"\hline")
    print(r"\textbf{State Variable} & \textbf{MSE} & \textbf{Correlation} \\")
    print(r"\hline")
    for state_name, values in metrics.items():
        print(f"{state_name} & {values['mse']:.2e} & {values['correlation']:.3f} \\\\")
    print(r"\hline")
    print(r"\end{tabular}")

    return metrics


# Example usage for Mountain Car:

# First, get predictions from your SINDy model
# model = create_transition_function("../../src/data/mountaincar/mountain_car_data.csv")
# actual_data = pd.read_csv("../../src/data/mountaincar/mountain_car_data.csv")
# metrics_mc = calculate_validation_metrics(actual_data, model, "mountain_car")


# Example usage for Lunar Lander:
model = create_transition_function("../../src/data/lunarlander/lunar_lander_data.csv")
actual_data = pd.read_csv("../../src/data/lunarlander/lunar_lander_data.csv")
metrics_ll = calculate_validation_metrics(actual_data, model, "lunar_lander")
