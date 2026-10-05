# Latent Dynamics Portfolio

A compact, reproducible comparison of three ways to learn from the same simulated pendulum trajectory:

1. **Autoencoder** — compresses and reconstructs the instantaneous physical state.
2. **Direct predictor** — predicts the next angular position and velocity directly.
3. **Minimal Tabular JEPA** — predicts the next representation using a momentum-updated target encoder.

The project is intentionally specific to one dataset and one physical system. Its purpose is to demonstrate model design, temporal validation, representation analysis, and clean ML engineering—not to provide a generic CSV framework. This work started with initial exploratory scripts (for understanding) and then evolutionated into a structured experimental pipeline (assited by AI)

## Research question

How do reconstruction-based, state-prediction, and representation-prediction objectives encode the dynamics of a pendulum under random torque?

## Dataset

`data/raw/pendulum_random_torque.csv` contains 18,961 time-ordered observations generated in Isaac Sim. The models use:

| Variable | Meaning | Role |
|---|---|---|
| `theta_rad` | Angular position | State and prediction target |
| `omega_rad_s` | Angular velocity | State and prediction target |
| `alpha_rad_s2` | Angular acceleration | Auxiliary state |
| `torque_Nm` | Applied torque | Control input/state context |

All experiments use the first 80% of the timeline for training and the final 20% for validation. Scalers are fit on the training partition only. Temporal models split the timeline **before** constructing one-step pairs, so no pair crosses the training/validation boundary.

## Architecture

```text
data/raw/pendulum_random_torque.csv
                 |
       chronological split + scaling
                 |
       +---------+----------+
       |         |          |
  Autoencoder  Direct     Minimal JEPA
  x -> z -> x  x_t->y_t+1  x_t->z->z_t+1
       |         |          |
       +---------+----------+
                 |
     checkpoints, metrics, and plots
```

Shared concerns—data validation, deterministic seeding, training, early stopping, evaluation, and artifact generation—live in reusable modules under `src/latent_dynamics`. Model-specific behavior remains explicit.

## Quick start

Requires Python 3.10 or newer.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
python -m pip install -e ".[dev]"

latent-dynamics all
```

For a fast smoke run:

```bash
latent-dynamics all --epochs 2 --patience 1 --output-dir outputs/smoke
```

Individual experiments are `autoencoder`, `direct-predictor`, and `jepa`.

## Outputs

Each experiment writes to `outputs/<experiment>/`:

- `model.pt` — best early-stopped checkpoint and configuration
- `metrics.json` — validation metrics and run metadata
- `history.csv` — epoch-level training and validation losses
- `learning_curve.png` — loss curves
- `latent_trajectory.png` — two-dimensional latent trajectory where applicable

Metrics are reported in standardized space for model comparison and in physical units for the autoencoder and direct predictor. JEPA evaluation includes a persistence baseline, relative improvement, latent variance, correlation, and a collapse diagnostic.

## Model choices

### Autoencoder

The four-dimensional state is compressed to a two-dimensional latent vector and reconstructed with a symmetric MLP. Including torque makes the comparison consistent with the predictive models and asks the embedding to preserve both state and control context.

### Direct predictor

The full state at time `t` predicts `theta_rad` and `omega_rad_s` at `t+1`. Acceleration remains an informative input but is not forced into the output objective; this preserves the strongest result from the original auxiliary-predictor experiment.

### Minimal Tabular JEPA

A context encoder and predictor learn through latent-space MSE. The target encoder receives no gradients and follows the context encoder through an exponential moving average. Performance is compared with the persistence assumption `z(t+1) = z(t)`; latent standard deviation is reported to detect trivial collapse.

## Corrections made during refactoring (refactoring latent-dynamics-portfolio)

- Replaced repeated monolithic scripts with importable models and a shared training pipeline. 
- Renamed the held-out partition from “test” to “validation,” because it controls early stopping.
- Removed the direct predictor's cross-boundary temporal pair by splitting before pairing.
- Ensured the target encoder stays frozen and is updated only by EMA.
- Restored the complete best JEPA checkpoint, including context encoder, target encoder, and predictor.
- Added an epsilon guard for persistence-relative improvement and safe handling for undefined correlations.
- Saved plots instead of opening blocking interactive windows.
- Added dataset/schema checks, deterministic data-loader shuffling, physical-unit metrics, and automated tests.

## Repository layout

```text
.
├── data/raw/                 # The single portfolio dataset
├── src/latent_dynamics/      # Data, models, training, evaluation, CLI
├── tests/                    # Shape and temporal-boundary regression tests
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Scope and interpretation

This is a portfolio study, not a production forecasting service. One trajectory is used, the forecast horizon is one simulation step, and the final 20% remains a validation segment rather than an untouched benchmark test set. Results demonstrate learning behavior on this system; they should not be interpreted as generalization to other pendulums or operating regimes.

