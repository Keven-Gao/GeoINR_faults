"""
geoinr_faults/nn/models.py
───────────────────
Built-in network architectures.  Every model takes `input_dim` as its first
constructor argument and returns a (N, 1) scalar field.

Use get_model() to retrieve a model class by name, or subclass BaseGeoNet
to plug in a custom architecture.
"""

from __future__ import annotations
from typing import List, Optional

import torch
import torch.nn as nn

from geoinr_faults.inr.activations import get_activation


# ──────────────────────────────────────────────────────────────────────────────
# Base class
# ──────────────────────────────────────────────────────────────────────────────

class BaseGeoNet(nn.Module):
    """Minimal contract: __init__(input_dim, **kwargs) → forward(x) → (N,1)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


# ──────────────────────────────────────────────────────────────────────────────
# MLP  (the primary architecture used in the notebook)
# ──────────────────────────────────────────────────────────────────────────────

class MLP(BaseGeoNet):
    """
    Fully-connected MLP with configurable depth, width, and activation.

    Parameters
    ----------
    input_dim  : Number of input features (3 xyz + n_faults).
    hidden_dim : Width of every hidden layer.
    n_layers   : Number of hidden layers.
    activation : Activation name / class / instance (see nn.activations).
    output_dim : Usually 1 (scalar field).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        n_layers: int = 3,
        activation: str | nn.Module = "softplus",
        output_dim: int = 1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        layers: List[nn.Module] = []
        in_features = input_dim
        for _ in range(n_layers):
            layers.append(nn.Linear(in_features, hidden_dim))
            layers.append(get_activation(activation))
            in_features = hidden_dim
        layers.append(nn.Linear(in_features, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ──────────────────────────────────────────────────────────────────────────────
# SIREN  (sinusoidal representation network)
# ──────────────────────────────────────────────────────────────────────────────

class SIREN(BaseGeoNet):
    """
    Sinusoidal Representation Network (Sitzmann et al., 2020).

    Parameters
    ----------
    input_dim    : Same as MLP.
    hidden_dim   : Width of every hidden layer.
    n_layers     : Number of hidden layers.
    omega_0      : Frequency of the first-layer sine.
    omega_hidden : Frequency of subsequent layers.
    output_dim   : Usually 1.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        n_layers: int = 3,
        omega_0: float = 30.0,
        omega_hidden: float = 30.0,
        output_dim: int = 1,
    ):
        super().__init__()
        self.input_dim = input_dim

        def _siren_linear(in_f: int, out_f: int, is_first: bool) -> nn.Linear:
            lin = nn.Linear(in_f, out_f)
            # weight initialisation per the SIREN paper
            with torch.no_grad():
                if is_first:
                    lin.weight.uniform_(-1 / in_f, 1 / in_f)
                else:
                    bound = (6.0 / in_f) ** 0.5 / omega_hidden
                    lin.weight.uniform_(-bound, bound)
            return lin

        layers: List[nn.Module] = []
        in_f = input_dim
        for i in range(n_layers):
            layers.append(_siren_linear(in_f, hidden_dim, is_first=(i == 0)))
            omega = omega_0 if i == 0 else omega_hidden
            from geoinr_faults.inr.activations import Sine
            layers.append(Sine(omega))
            in_f = hidden_dim
        layers.append(nn.Linear(in_f, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ──────────────────────────────────────────────────────────────────────────────
# ResNet-style MLP  (skip connections)
# ──────────────────────────────────────────────────────────────────────────────

class ResNetMLP(BaseGeoNet):
    """
    MLP with residual (skip) connections every `skip_every` layers.

    Parameters
    ----------
    input_dim  : Number of input features.
    hidden_dim : Width of all hidden layers.
    n_layers   : Total number of hidden layers.
    skip_every : Add a skip connection every N layers (default 2).
    activation : Activation name / class / instance.
    output_dim : Usually 1.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        n_layers: int = 4,
        skip_every: int = 2,
        activation: str | nn.Module = "softplus",
        output_dim: int = 1,
    ):
        super().__init__()
        self.skip_every = skip_every
        self.activation = get_activation(activation)

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(n_layers)]
        )
        # 1×1 projection used to add the skip when dims differ (not needed here)
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.activation(self.input_proj(x))
        for i, layer in enumerate(self.hidden_layers):
            residual = h
            h = self.activation(layer(h))
            if (i + 1) % self.skip_every == 0:
                h = h + residual
        return self.output_layer(h)


# ──────────────────────────────────────────────────────────────────────────────
# Registry
# ──────────────────────────────────────────────────────────────────────────────

_MODEL_REGISTRY: dict[str, type] = {
    "mlp":       MLP,
    "siren":     SIREN,
    "resnet":    ResNetMLP,
    "resnetmlp": ResNetMLP,
}


def get_model(name: str | type) -> type:
    """
    Return a model *class* by name (not an instance).

    Examples
    --------
    >>> ModelClass = get_model('mlp')
    >>> model = ModelClass(input_dim=8, hidden_dim=128, n_layers=4)
    """
    if isinstance(name, type):
        return name
    key = name.lower().strip()
    if key not in _MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{name}'. Available: {sorted(_MODEL_REGISTRY)}"
        )
    return _MODEL_REGISTRY[key]


def list_models() -> list[str]:
    return sorted(_MODEL_REGISTRY)


def build_model(
    name: str | type,
    input_dim: int,
    **kwargs,
) -> BaseGeoNet:
    """
    Convenience: get class and instantiate in one call.

    Examples
    --------
    >>> model = build_model('mlp', input_dim=8, hidden_dim=256, n_layers=3)
    >>> model = build_model('siren', input_dim=8)
    """
    ModelClass = get_model(name)
    return ModelClass(input_dim=input_dim, **kwargs)
