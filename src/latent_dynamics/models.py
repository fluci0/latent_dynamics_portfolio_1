from copy import deepcopy

import torch
from torch import nn


class Autoencoder(nn.Module):
    def __init__(self, input_dim: int = 4, hidden_dim: int = 32, latent_dim: int = 2) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(inputs))


class DirectPredictor(nn.Module):
    def __init__(self, input_dim: int = 4, hidden_dim: int = 32, output_dim: int = 2) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


def _encoder(input_dim: int, hidden_dim: int, latent_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim // 2),
        nn.ReLU(),
        nn.Linear(hidden_dim // 2, latent_dim),
    )


class MinimalTabularJEPA(nn.Module):
    def __init__(self, input_dim: int = 4, hidden_dim: int = 32, latent_dim: int = 2) -> None:
        super().__init__()
        self.context_encoder = _encoder(input_dim, hidden_dim, latent_dim)
        self.target_encoder = deepcopy(self.context_encoder)
        self.target_encoder.requires_grad_(False)
        self.predictor = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        return self.predictor(self.context_encoder(context))

    @torch.no_grad()
    def target(self, future: torch.Tensor) -> torch.Tensor:
        return self.target_encoder(future)

    @torch.no_grad()
    def update_target(self, momentum: float) -> None:
        for context_parameter, target_parameter in zip(
            self.context_encoder.parameters(), self.target_encoder.parameters()
        ):
            target_parameter.mul_(momentum).add_(context_parameter, alpha=1.0 - momentum)

