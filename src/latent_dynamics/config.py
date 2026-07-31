from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ExperimentConfig:
    """Configuration intentionally scoped to the portfolio's single dataset."""

    csv_path: Path = PROJECT_ROOT / "data" / "raw" / "pendulum_random_torque.csv"
    output_dir: Path = PROJECT_ROOT / "outputs"
    state_columns: tuple[str, ...] = (
        "theta_rad",
        "omega_rad_s",
        "alpha_rad_s2",
        "torque_Nm",
    )
    prediction_columns: tuple[str, ...] = ("theta_rad", "omega_rad_s")
    train_fraction: float = 0.8
    seed: int = 22
    batch_size: int = 64
    epochs: int = 700
    patience: int = 25
    learning_rate: float = 1e-3
    hidden_dim: int = 32
    latent_dim: int = 2
    ema_momentum: float = 0.99

    def __post_init__(self) -> None:
        if not 0.0 < self.train_fraction < 1.0:
            raise ValueError("train_fraction must be between 0 and 1")
        if self.latent_dim < 1 or self.epochs < 1 or self.patience < 1:
            raise ValueError("latent_dim, epochs, and patience must be positive")

