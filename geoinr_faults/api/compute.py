"""
geoinr_faults/api/compute.py
─────────────────────
One call trains the network, runs inference, and populates the GeoModel.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Type

import torch
import torch.nn as nn

from geoinr_faults.core.geomodel import GeoModel
from geoinr_faults.solvers.nn_solver import (
    build_input_tensors,
    build_network,
    train,
    enforce_output_polarity,
    predict,
)


# ──────────────────────────────────────────────────────────────────────────────
# compute_model
# ──────────────────────────────────────────────────────────────────────────────

def compute_model(
    model: GeoModel,
    # ── network ────────────────────────────────────────────────────────────────
    nn_model: str | type | nn.Module = "mlp",
    hidden_dim: int = 256,
    n_layers: int = 3,
    activation: str | nn.Module = "softplus",
    softplus_beta: Optional[float] = None,
    # ── training ───────────────────────────────────────────────────────────────
    epochs: int = 2000,
    lr: float = 1e-3,
    optimizer_cls: Type[torch.optim.Optimizer] = torch.optim.AdamW,
    loss_weights: Optional[Dict[str, float]] = None,
    orientation_tensor: Optional[torch.Tensor] = None,
    # ── misc ───────────────────────────────────────────────────────────────────
    seed: Optional[int] = None,
    device: Optional[torch.device] = None,
    fault_scale: Optional[float] = None,
    print_every: int = 500,
    verbose: bool = True,
    use_best: bool = True,
    enforce_polarity: bool = True,
    **model_kwargs: Any,
) -> GeoModel:
    """
    Full pipeline: build tensors → build network → train → predict.

    Parameters
    ----------
    model              : GeoModel from create_geomodel (+ add_faults).
    nn_model           : 'mlp' | 'siren' | 'resnet' | custom class/instance.
    hidden_dim         : Hidden layer width.
    n_layers           : Number of hidden layers.
    activation         : Activation function name or nn.Module.
    softplus_beta      : Beta for Softplus when activation='softplus'.
                         If None, uses activation registry default.
    epochs             : Training iterations.
    lr                 : Learning rate.
    optimizer_cls      : Torch optimiser class (default AdamW).
    loss_weights       : Dict with keys 'var', 'on', 'above', 'below', 'orie'.
    orientation_tensor : (M, 3) orientation measurements for orie_loss.
                         If None, model.orientations is used automatically
                         when available.
    seed               : Random seed. If None, uses random initialization.
    device             : torch.device override (auto-detects GPU by default).
    fault_scale        : Override the fault feature scaling factor.
    print_every        : Print interval (0 = silent).
    verbose            : Print summary.
    use_best           : Load best checkpoint before final inference.
    enforce_polarity   : Detect and correct global sign-flip across seeds.
    model_kwargs       : Extra kwargs forwarded to the network constructor.

    Returns
    -------
    The same GeoModel with predictions, iso_surfaces, and loss_history set.

    Example
    -------
    >>> geo_data = mg.compute_model(data)

    >>> geo_data = mg.compute_model(
    ...     data,
    ...     nn_model='mlp',
    ...     hidden_dim=256,
    ...     n_layers=3,
    ...     activation='softplus',
    ...     softplus_beta=30.0,
    ...     epochs=3000,
    ...     lr=5e-4,
    ...     loss_weights={'var': 1.0, 'on': 1.0, 'above': 1.0, 'below': 1.0},
    ... )
    """
    # 1. Build augmented input tensors (xyz + fault features)
    build_input_tensors(model, device=device, fault_scale=fault_scale)

    # 2. Instantiate network + optimiser
    build_network(
        model,
        nn_model=nn_model,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        activation=activation,
        softplus_beta=softplus_beta,
        optimizer_cls=optimizer_cls,
        lr=lr,
        seed=seed,
        **model_kwargs,
    )

    # 3. Training loop
    if orientation_tensor is None and model.orientations is not None:
        orientation_tensor = torch.tensor(model.orientations, dtype=torch.float32)

    train(
        model,
        epochs=epochs,
        loss_weights=loss_weights,
        orientation_tensor=orientation_tensor,
        print_every=print_every,
        verbose=verbose,
    )

    # 4. Optional polarity stabilization (before final inference)
    if enforce_polarity:
        enforce_output_polarity(model, use_best=use_best)

    # 5. Inference over full grid
    predict(model, use_best=use_best)

    return model

