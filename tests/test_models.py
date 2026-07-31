import torch

from latent_dynamics.models import Autoencoder, DirectPredictor, MinimalTabularJEPA


def test_model_shapes() -> None:
    inputs = torch.randn(8, 4)
    assert Autoencoder()(inputs).shape == (8, 4)
    assert DirectPredictor()(inputs).shape == (8, 2)
    jepa = MinimalTabularJEPA()
    assert jepa(inputs).shape == (8, 2)
    assert jepa.target(inputs).shape == (8, 2)
    assert all(not parameter.requires_grad for parameter in jepa.target_encoder.parameters())

