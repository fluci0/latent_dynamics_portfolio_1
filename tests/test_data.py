from dataclasses import replace

import numpy as np
import pandas as pd

from latent_dynamics.config import ExperimentConfig
from latent_dynamics.data import prepare_temporal


def test_temporal_split_has_no_boundary_pair() -> None:
    frame = pd.DataFrame({
        "theta_rad": np.arange(10),
        "omega_rad_s": np.arange(10),
        "alpha_rad_s2": np.arange(10),
        "torque_Nm": np.arange(10),
    })
    config = replace(ExperimentConfig(), train_fraction=0.8, batch_size=32)
    prepared = prepare_temporal(frame, config, jepa=False)
    assert len(prepared.train_loader.dataset) == 7
    assert len(prepared.validation_loader.dataset) == 1
    assert prepared.validation_targets_raw[0, 0] == 9

