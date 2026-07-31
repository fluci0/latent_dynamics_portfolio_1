from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from .config import ExperimentConfig


@dataclass
class PreparedData:
    train_loader: DataLoader
    validation_loader: DataLoader
    train_inputs: torch.Tensor
    validation_inputs: torch.Tensor
    train_targets: torch.Tensor
    validation_targets: torch.Tensor
    input_scaler: StandardScaler
    target_scaler: StandardScaler
    validation_targets_raw: np.ndarray


def load_pendulum(path: Path, required_columns: tuple[str, ...]) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")
    frame = pd.read_csv(path)
    missing = sorted(set(required_columns) - set(frame.columns))
    if missing:
        raise ValueError(f"Dataset is missing columns: {', '.join(missing)}")
    if frame.empty:
        raise ValueError("Dataset is empty")
    if not np.isfinite(frame.loc[:, required_columns].to_numpy(dtype=float)).all():
        raise ValueError("State columns contain missing or non-finite values")
    return frame


def _tensor(values: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(values, dtype=torch.float32)


def _loader(inputs: torch.Tensor, targets: torch.Tensor, config: ExperimentConfig, shuffle: bool) -> DataLoader:
    generator = torch.Generator().manual_seed(config.seed)
    return DataLoader(
        TensorDataset(inputs, targets),
        batch_size=config.batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
    )


def prepare_autoencoder(frame: pd.DataFrame, config: ExperimentConfig) -> PreparedData:
    values = frame.loc[:, config.state_columns].to_numpy(dtype=np.float64)
    split = int(len(values) * config.train_fraction)
    train_raw, validation_raw = values[:split], values[split:]
    scaler = StandardScaler().fit(train_raw)
    train = _tensor(scaler.transform(train_raw))
    validation = _tensor(scaler.transform(validation_raw))
    return PreparedData(
        _loader(train, train, config, True),
        _loader(validation, validation, config, False),
        train,
        validation,
        train,
        validation,
        scaler,
        scaler,
        validation_raw,
    )


def prepare_temporal(frame: pd.DataFrame, config: ExperimentConfig, *, jepa: bool) -> PreparedData:
    """Split the timeline first, then form t -> t+1 pairs inside each partition."""
    split = int(len(frame) * config.train_fraction)
    train_frame, validation_frame = frame.iloc[:split], frame.iloc[split:]
    input_columns = config.state_columns
    target_columns = config.state_columns if jepa else config.prediction_columns

    def pairs(partition: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        inputs = partition.loc[:, input_columns].iloc[:-1].to_numpy(dtype=np.float64)
        targets = partition.loc[:, target_columns].iloc[1:].to_numpy(dtype=np.float64)
        return inputs, targets

    train_inputs_raw, train_targets_raw = pairs(train_frame)
    validation_inputs_raw, validation_targets_raw = pairs(validation_frame)
    input_scaler = StandardScaler().fit(train_inputs_raw)
    target_scaler = input_scaler if jepa else StandardScaler().fit(train_targets_raw)
    train_inputs = _tensor(input_scaler.transform(train_inputs_raw))
    validation_inputs = _tensor(input_scaler.transform(validation_inputs_raw))
    train_targets = _tensor(target_scaler.transform(train_targets_raw))
    validation_targets = _tensor(target_scaler.transform(validation_targets_raw))
    return PreparedData(
        _loader(train_inputs, train_targets, config, True),
        _loader(validation_inputs, validation_targets, config, False),
        train_inputs,
        validation_inputs,
        train_targets,
        validation_targets,
        input_scaler,
        target_scaler,
        validation_targets_raw,
    )

