import json
from pathlib import Path
import matplotlib.pyplot as plt


def plot_dummy():
    # Placeholder Pareto curve
    x = [0.2, 0.3, 0.4, 0.5]
    y = [0.55, 0.6, 0.62, 0.63]
    plt.figure(figsize=(4, 3))
    plt.plot(x, y, marker="o")
    plt.xlabel("Interaction Cost (lower is better)")
    plt.ylabel("Task Success (higher is better)")
    plt.title("Pareto Curve (Dummy)")
    out = Path(__file__).resolve().parent / "pareto.png"
    plt.tight_layout()
    plt.savefig(out)
    print("Saved:", out)


if __name__ == "__main__":
    plot_dummy()


