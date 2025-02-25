import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from stable_baselines3 import SAC


def visualize_mountain_car_policy(model, title="Mountain Car Policy"):
    # Use LaTeX fonts for publication quality
    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
        }
    )

    # Create position and velocity meshgrid
    pos = np.linspace(-1.2, 0.6, 100)
    vel = np.linspace(-0.07, 0.07, 100)
    pos_grid, vel_grid = np.meshgrid(pos, vel)

    # Reshape for model prediction
    states = np.column_stack((pos_grid.ravel(), vel_grid.ravel()))

    # Get actions for each state
    actions = []
    for state in states:
        action = model.predict(state.reshape(1, -1), deterministic=True)[0]
        actions.append(action[0])

    actions = np.array(actions).reshape(pos_grid.shape)

    # Create figure with better proportions
    fig, ax = plt.subplots(figsize=(6, 5))

    # Create custom colormap for better visualization
    colors = ["#e41a1c", "#377eb8"]  # Red to Blue (colorblind-friendly)
    custom_cmap = LinearSegmentedColormap.from_list("custom", colors)

    # Plot contours with improved aesthetics
    contour = plt.contourf(pos_grid, vel_grid, actions, levels=20, cmap=custom_cmap)
    cbar = plt.colorbar(contour)
    cbar.set_label("Action Force (N)", labelpad=10)

    # Add arrows with better visibility
    skip = 8  # Increased skip for clearer arrows
    for i in range(0, len(pos), skip):
        for j in range(0, len(vel), skip):
            action = actions[j, i]
            plt.arrow(
                pos[i],
                vel[j],
                action * 0.05,
                0,  # Scale arrow length
                head_width=0.003,
                head_length=0.02,
                fc="black",
                ec="black",
                alpha=0.6,
            )

    # Improve axes labels and title
    plt.xlabel("Position (m)")
    plt.ylabel("Velocity (m/s)")
    plt.title(title)

    # Add grid with better styling
    plt.grid(True, linestyle="--", alpha=0.3)

    # Add key points annotation
    plt.plot(-1.2, 0, "k*", markersize=10, label="Start")
    plt.plot(0.5, 0, "k^", markersize=10, label="Goal")
    plt.legend()

    # Adjust layout
    plt.tight_layout()

    return plt


def main():
    # Load models
    original_model = SAC.load("../../models/mountain_car/sac_mountaincar_core")
    surrogate_model = SAC.load("../../models/mountain_car/sac_mountaincar_sindy_v1")

    # Create and save visualizations
    visualize_mountain_car_policy(original_model, "Original Environment Policy")
    plt.savefig(
        "original_mc_policy.pdf",
        dpi=300,
        bbox_inches="tight",
        format="pdf",
        transparent=True,
    )
    plt.close()

    visualize_mountain_car_policy(surrogate_model, "Surrogate Environment Policy")
    plt.savefig(
        "surrogate_mc_policy.pdf",
        dpi=300,
        bbox_inches="tight",
        format="pdf",
        transparent=True,
    )
    plt.close()


if __name__ == "__main__":
    main()
