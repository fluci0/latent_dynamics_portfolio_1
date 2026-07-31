from copy import deepcopy
from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader

from .models import MinimalTabularJEPA


@dataclass
class TrainingResult:
    train_loss: list[float]
    validation_loss: list[float]
    best_epoch: int
    best_validation_loss: float


def _mean_loss(model: nn.Module, loader: DataLoader, loss_function: nn.Module, device: torch.device, *, jepa: bool) -> float:
    model.eval()
    total = 0.0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            predictions = model(inputs)
            expected = model.target(targets) if jepa else targets
            total += loss_function(predictions, expected).item() * inputs.size(0)
    return total / len(loader.dataset)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    *,
    device: torch.device,
    epochs: int,
    patience: int,
    learning_rate: float,
    ema_momentum: float | None = None,
) -> TrainingResult:
    model.to(device)
    jepa = isinstance(model, MinimalTabularJEPA)
    parameters = (
        list(model.context_encoder.parameters()) + list(model.predictor.parameters())
        if jepa
        else model.parameters()
    )
    optimizer = torch.optim.Adam(parameters, lr=learning_rate)
    loss_function = nn.MSELoss()
    train_history: list[float] = []
    validation_history: list[float] = []
    best_loss = float("inf")
    best_epoch = 0
    best_state = None
    stale_epochs = 0

    for epoch in range(1, epochs + 1):
        model.train()
        if jepa:
            model.target_encoder.eval()
        total = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            predictions = model(inputs)
            expected = model.target(targets) if jepa else targets
            loss = loss_function(predictions, expected)
            loss.backward()
            optimizer.step()
            if jepa:
                model.update_target(ema_momentum if ema_momentum is not None else 0.99)
            total += loss.item() * inputs.size(0)

        train_history.append(total / len(train_loader.dataset))
        validation_loss = _mean_loss(model, validation_loader, loss_function, device, jepa=jepa)
        validation_history.append(validation_loss)
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= patience:
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a valid checkpoint")
    model.load_state_dict(best_state)
    return TrainingResult(train_history, validation_history, best_epoch, best_loss)

