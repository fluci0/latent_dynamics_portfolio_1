import numpy as np
import torch
from torch import nn

from .data import PreparedData
from .models import MinimalTabularJEPA


def evaluate_physical(model: nn.Module, data: PreparedData, device: torch.device, columns: tuple[str, ...]) -> tuple[dict, np.ndarray]:
    model.eval()
    with torch.no_grad():
        predictions_scaled = model(data.validation_inputs.to(device)).cpu().numpy()
    targets_scaled = data.validation_targets.numpy()
    predictions_raw = data.target_scaler.inverse_transform(predictions_scaled)
    error_scaled = (predictions_scaled - targets_scaled) ** 2
    error_raw = (predictions_raw - data.validation_targets_raw) ** 2
    metrics = {
        "standardized_mse": float(error_scaled.mean()),
        "physical_mse": float(error_raw.mean()),
        "standardized_mse_by_variable": dict(zip(columns, error_scaled.mean(axis=0).tolist())),
        "physical_mse_by_variable": dict(zip(columns, error_raw.mean(axis=0).tolist())),
    }
    return metrics, predictions_raw


def evaluate_jepa(model: MinimalTabularJEPA, data: PreparedData, device: torch.device) -> tuple[dict, np.ndarray, np.ndarray]:
    model.eval()
    with torch.no_grad():
        context = model.context_encoder(data.validation_inputs.to(device))
        predicted = model.predictor(context)
        target = model.target_encoder(data.validation_targets.to(device))
    context_np, predicted_np, target_np = (item.cpu().numpy() for item in (context, predicted, target))
    prediction_mse = float(np.mean((predicted_np - target_np) ** 2))
    persistence_mse = float(np.mean((context_np - target_np) ** 2))
    improvement = None if persistence_mse <= np.finfo(float).eps else 1.0 - prediction_mse / persistence_mse
    target_std = target_np.std(axis=0)
    correlations = []
    for index in range(target_np.shape[1]):
        if target_np[:, index].std() == 0 or predicted_np[:, index].std() == 0:
            correlations.append(None)
        else:
            correlations.append(float(np.corrcoef(target_np[:, index], predicted_np[:, index])[0, 1]))
    metrics = {
        "latent_prediction_mse": prediction_mse,
        "persistence_mse": persistence_mse,
        "relative_improvement_over_persistence": improvement,
        "target_latent_std": target_std.tolist(),
        "predicted_latent_std": predicted_np.std(axis=0).tolist(),
        "latent_correlation": correlations,
        "collapsed": bool(np.all(target_std < 1e-3)),
    }
    return metrics, predicted_np, target_np

