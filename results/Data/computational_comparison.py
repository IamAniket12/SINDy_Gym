import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def create_computational_comparison():
    plt.style.use("seaborn-v0_8-paper")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
        }
    )

    # Real data with data collection + training steps
    env_steps = {
        "Mountain Car": {
            "Original": 100000,
            "Surrogate": 65075,  # 65000 + 75 (training + data collection)
        },
        "Lunar Lander": {
            "Original": 1000000,
            "Surrogate": 801000,  # 800000 + 1000 (training + data collection)
        },
    }

    # Create figure
    fig, ax = plt.subplots(figsize=(3, 4))

    environments = list(env_steps.keys())
    x = np.arange(len(environments))
    width = 0.20

    # Plot data in log scale
    original_steps = [env_steps[env]["Original"] for env in environments]
    surrogate_steps = [env_steps[env]["Surrogate"] for env in environments]

    ax.set_yscale("log")
    bars1 = ax.bar(
        x - width / 2,
        original_steps,
        width,
        label="Original Environment",
        color="#1f77b4",
    )
    bars2 = ax.bar(
        x + width / 2, surrogate_steps, width, label="SINDy Surrogate", color="#2ca02c"
    )

    # Customize the plot
    ax.set_ylabel("Total Environment Steps (log scale)")
    ax.set_title("Total Steps Required (Data Collection + Training)")
    ax.set_xticks(x)
    ax.set_xticklabels(environments)
    ax.legend()

    # Add value labels on the bars
    def format_value(value):
        if value >= 1000000:
            return f"{value/1000000:.1f}M"
        elif value >= 1000:
            return f"{value/1000:.1f}K"
        return str(value)

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                format_value(height),
                ha="center",
                va="bottom",
            )

    plt.tight_layout()
    plt.savefig(
        "computational_comparison.pdf", format="pdf", dpi=300, bbox_inches="tight"
    )
    plt.close()


if __name__ == "__main__":
    create_computational_comparison()
