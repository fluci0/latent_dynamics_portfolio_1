from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_history(train: list[float], validation: list[float], destination: Path) -> None:
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.plot(train, label="Training")
    axis.plot(validation, label="Validation")
    axis.set(xlabel="Epoch", ylabel="MSE", title="Learning curve")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(destination, dpi=160)
    plt.close(figure)


def plot_latent(target: np.ndarray, destination: Path, predicted: np.ndarray | None = None) -> None:
    if target.shape[1] != 2:
        return
    figure, axis = plt.subplots(figsize=(6, 6))
    axis.plot(target[:, 0], target[:, 1], label="Target", linewidth=1.4)
    if predicted is not None:
        axis.plot(predicted[:, 0], predicted[:, 1], "--", label="Predicted", linewidth=1.1)
    axis.set(xlabel="z1", ylabel="z2", title="Validation latent trajectory")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(destination, dpi=160)
    plt.close(figure)

