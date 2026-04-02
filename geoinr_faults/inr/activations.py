"""
geoinr_faults/nn/activations.py
────────────────────────
Named registry of activation functions.
Pass the name string to get_activation() or use the class directly.

Supported names (case-insensitive)
───────────────────────────────────
  'softplus'    – nn.Softplus(beta=30)   [default, best for implicit surfaces]
  'relu'        – nn.ReLU()
  'tanh'        – nn.Tanh()
  'silu'        – nn.SiLU()  (Swish)
  'gelu'        – nn.GELU()
  'sine'        – custom Sine activation  (SIREN-style)
  'leaky_relu'  – nn.LeakyReLU(0.2)
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn


# ─────────────────────────── custom activations ───────────────────────────────

class Sine(nn.Module):
    """SIREN-style sinusoidal activation.  out = sin(ω · x)"""

    def __init__(self, omega: float = 30.0):
        super().__init__()
        self.omega = omega

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega * x)

    def extra_repr(self) -> str:
        return f"omega={self.omega}"


# ─────────────────────────── registry ────────────────────────────────────────

_REGISTRY: dict[str, type | callable] = {
    "softplus":   lambda: nn.Softplus(beta=30),
    "relu":       nn.ReLU,
    "tanh":       nn.Tanh,
    "silu":       nn.SiLU,
    "swish":      nn.SiLU,       # alias
    "gelu":       nn.GELU,
    "sine":       Sine,
    "siren":      Sine,          # alias
    "leaky_relu": lambda: nn.LeakyReLU(0.2),
    "elu":        nn.ELU,
    "sigmoid":    nn.Sigmoid,
    "identity":   nn.Identity,
}


def get_activation(name: str | nn.Module | type, **kwargs) -> nn.Module:
    """
    Return an instantiated activation module.

    Parameters
    ----------
    name   : str key (see module docstring), an nn.Module subclass, or an
             already-instantiated nn.Module (returned as-is).
    kwargs : Forwarded to the constructor when *name* is a string or class.

    Examples
    --------
    >>> act = get_activation('softplus')
    >>> act = get_activation('sine', omega=10.0)
    >>> act = get_activation(nn.Tanh)
    >>> act = get_activation(nn.Tanh())   # returned unchanged
    """
    if isinstance(name, nn.Module):
        return name
    if isinstance(name, type) and issubclass(name, nn.Module):
        return name(**kwargs)
    if isinstance(name, str):
        key = name.lower().strip()
        if key not in _REGISTRY:
            raise ValueError(
                f"Unknown activation '{name}'. "
                f"Available: {sorted(_REGISTRY)}"
            )
        factory = _REGISTRY[key]
        return factory(**kwargs) if kwargs else factory()
    raise TypeError(f"Expected str, nn.Module subclass, or nn.Module instance; got {type(name)}")


def list_activations() -> list[str]:
    """Return sorted list of registered activation names."""
    return sorted(_REGISTRY)
