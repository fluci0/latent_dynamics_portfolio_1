import json
import random
from dataclasses import asdict

import numpy as np
import pandas as pd
import torch

from .config import ExperimentConfig
from .data import load_pendulum, prepare_autoencoder, prepare_temporal
from .evaluation import evaluate_jepa, evaluate_physical
from .models import Autoencoder, DirectPredictor, MinimalTabularJEPA
from .plots import plot_history, plot_latent
from .training import train_model


EXPERIMENTS = ("autoencoder", "direct-predictor", "jepa")


def set_reproducible(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _jsonable_config(config: ExperimentConfig) -> dict:
    values = asdict(config)
    values["csv_path"] = str(config.csv_path)
    values["output_dir"] = str(config.output_dir)
    return values


def run_experiment(name: str, config: ExperimentConfig) -> dict:
    if name not in EXPERIMENTS:
        raise ValueError(f"Unknown experiment: {name}")
    set_reproducible(config.seed)
    frame = load_pendulum(config.csv_path, config.state_columns)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    experiment_dir = config.output_dir / name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    if name == "autoencoder":
        data = prepare_autoencoder(frame, config)
        model = Autoencoder(len(config.state_columns), config.hidden_dim, config.latent_dim)
        metric_columns = config.state_columns
    elif name == "direct-predictor":
        data = prepare_temporal(frame, config, jepa=False)
        model = DirectPredictor(len(config.state_columns), config.hidden_dim, len(config.prediction_columns))
        metric_columns = config.prediction_columns
    else:
        data = prepare_temporal(frame, config, jepa=True)
        model = MinimalTabularJEPA(len(config.state_columns), config.hidden_dim, config.latent_dim)
        metric_columns = ()

    history = train_model(
        model,
        data.train_loader,
        data.validation_loader,
        device=device,
        epochs=config.epochs,
        patience=config.patience,
        learning_rate=config.learning_rate,
        ema_momentum=config.ema_momentum if name == "jepa" else None,
    )
    if name == "jepa":
        metrics, predicted_latent, target_latent = evaluate_jepa(model, data, device)
        plot_latent(target_latent, experiment_dir / "latent_trajectory.png", predicted_latent)
    else:
        metrics, _ = evaluate_physical(model, data, device, metric_columns)
        if name == "autoencoder":
            with torch.no_grad():
                latent = model.encoder(data.validation_inputs.to(device)).cpu().numpy()
            plot_latent(latent, experiment_dir / "latent_trajectory.png")

    metrics.update({
        "best_epoch": history.best_epoch,
        "best_validation_loss": history.best_validation_loss,
        "device": str(device),
        "training_samples": len(data.train_loader.dataset),
        "validation_samples": len(data.validation_loader.dataset),
    })
    torch.save({"model_state": model.state_dict(), "config": _jsonable_config(config)}, experiment_dir / "model.pt")
    pd.DataFrame({"train_loss": history.train_loss, "validation_loss": history.validation_loss}).to_csv(
        experiment_dir / "history.csv", index=False
    )
    plot_history(history.train_loss, history.validation_loss, experiment_dir / "learning_curve.png")
    (experiment_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def run_many(names: tuple[str, ...], config: ExperimentConfig) -> dict[str, dict]:
    return {name: run_experiment(name, config) for name in names}
