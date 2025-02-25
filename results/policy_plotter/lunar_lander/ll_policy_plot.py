import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from matplotlib.colors import ListedColormap
import seaborn as sns


def visualize_lunar_lander_policy(model, title="Lunar Lander Policy"):
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

    # Create figure with better proportions for paper
    fig = plt.figure(figsize=(8, 6))

    # Define colors suitable for publication (colorblind-friendly)
    action_colors = [
        "#CCCCCC",
        "#377eb8",
        "#fdb462",
        "#e41a1c",
    ]  # Gray, Blue, Orange, Red
    action_names = ["No Action", "Left Engine", "Main Engine", "Right Engine"]

    # Create subplots with better spacing
    gs = plt.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

    # 1. Position-based Policy
    ax1 = fig.add_subplot(gs[0, 0])
    decisions = np.zeros((20, 20))
    x_pos = np.linspace(-1, 1, 20)
    y_pos = np.linspace(-1, 1, 20)

    for i, x in enumerate(x_pos):
        for j, y in enumerate(y_pos):
            state = np.array([x, y, 0, 0, 0, 0, 0, 0])
            action = model.predict(state.reshape(1, -1), deterministic=True)[0]
            decisions[i, j] = action[0]

    im1 = ax1.imshow(
        decisions,
        extent=[-1, 1, -1, 1],
        origin="lower",
        cmap=ListedColormap(action_colors),
        aspect="auto",
    )
    ax1.set_title("(a) Position-based Policy")
    ax1.set_xlabel("Horizontal Position")
    ax1.set_ylabel("Vertical Position")
    ax1.grid(True, alpha=0.2)

    # 2. Velocity-based Policy
    ax2 = fig.add_subplot(gs[0, 1])
    velocities = np.linspace(-1, 1, 20)
    velocity_decisions = np.zeros((20, 20))

    for i, vx in enumerate(velocities):
        for j, vy in enumerate(velocities):
            state = np.array([0, 0, vx, vy, 0, 0, 0, 0])
            action = model.predict(state.reshape(1, -1), deterministic=True)[0]
            velocity_decisions[i, j] = action[0]

    im2 = ax2.imshow(
        velocity_decisions,
        extent=[-1, 1, -1, 1],
        origin="lower",
        cmap=ListedColormap(action_colors),
        aspect="auto",
    )
    ax2.set_title("(b) Velocity-based Policy")
    ax2.set_xlabel("Horizontal Velocity")
    ax2.set_ylabel("Vertical Velocity")
    ax2.grid(True, alpha=0.2)

    # 3. Angle-based Policy
    ax3 = fig.add_subplot(gs[1, 0])
    angles = np.linspace(-np.pi / 2, np.pi / 2, 20)
    ang_velocities = np.linspace(-1, 1, 20)
    angle_decisions = np.zeros((20, 20))

    for i, angle in enumerate(angles):
        for j, ang_vel in enumerate(ang_velocities):
            state = np.array([0, 0, 0, 0, angle, ang_vel, 0, 0])
            action = model.predict(state.reshape(1, -1), deterministic=True)[0]
            angle_decisions[i, j] = action[0]

    im3 = ax3.imshow(
        angle_decisions,
        extent=[-np.pi / 2, np.pi / 2, -1, 1],
        origin="lower",
        cmap=ListedColormap(action_colors),
        aspect="auto",
    )
    ax3.set_title("(c) Attitude Control Policy")
    ax3.set_xlabel("Angle (radians)")
    ax3.set_ylabel("Angular Velocity")
    ax3.grid(True, alpha=0.2)

    # 4. Legend with better formatting
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis("off")
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor=color, label=name)
        for color, name in zip(action_colors, action_names)
    ]
    ax4.legend(
        handles=legend_elements,
        loc="center",
        title="Control Actions",
        frameon=True,
        bbox_to_anchor=(0.5, 0.5),
    )

    plt.suptitle(title, y=1.02, fontsize=12)
    return plt


def main():
    original_model = PPO.load("../../models/lunar_lander/ppo-LunarLander-v1-core")
    surrogate_model = PPO.load("../../models/lunar_lander/ppo-LunarLander-sindy_v2")

    # Create visualizations with publication-quality settings
    visualize_lunar_lander_policy(original_model, "Original Environment Policy")
    plt.savefig(
        "original_lunar_policy.pdf",
        dpi=300,
        bbox_inches="tight",
        format="pdf",
        transparent=True,
    )
    plt.close()

    visualize_lunar_lander_policy(surrogate_model, "Surrogate Environment Policy")
    plt.savefig(
        "surrogate_lunar_policy.pdf",
        dpi=300,
        bbox_inches="tight",
        format="pdf",
        transparent=True,
    )
    plt.close()


if __name__ == "__main__":
    main()
