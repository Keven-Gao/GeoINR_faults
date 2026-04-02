"""
geoinr_faults/nn/losses.py
───────────────────
All loss functions used during implicit-surface training.

Each function is pure (no side-effects) and operates on torch tensors.
They are composed by solvers/nn_solver.py.

Loss taxonomy (from the notebook)
──────────────────────────────────
  var_loss   – minimise within-horizon variance  → flat iso-surface
  on_loss    – force points to lie on their iso-value  → surface fitting
  order_loss – enforce stratigraphic order  → correct layer stacking
  orie_loss  – match gradient direction to orientation measurements

An additional registry allows users to register custom loss functions by name.
"""

from __future__ import annotations
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────────
# helpers shared across loss functions
# ──────────────────────────────────────────────────────────────────────────────

def compute_preds_means_norms(
    y_pred: torch.Tensor,
    norm_grad: torch.Tensor,
    masks: List[torch.Tensor | list],
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
    """
    For each horizon mask return its predictions, mean prediction, and
    gradient-norm values.

    Parameters
    ----------
    y_pred     : (N, 1) network output.
    norm_grad  : (N,)   ‖∇f‖ at each training point.
    masks      : list of index arrays – one per horizon.

    Returns
    -------
    preds, means, norms  – each a list aligned with masks.
    """
    preds, means, norms = [], [], []
    for mask in masks:
        p = y_pred[mask].squeeze()
        preds.append(p)
        means.append(p.mean())
        norms.append(norm_grad[mask])
    return preds, means, norms


# ──────────────────────────────────────────────────────────────────────────────
# individual loss terms
# ──────────────────────────────────────────────────────────────────────────────

def var_loss(preds: List[torch.Tensor]) -> torch.Tensor:
    """
    Variance loss: penalise spread of scalar values within each horizon.

    L_var = Σ_i  Var(f(x_i))
    """
    return sum(p.var() for p in preds)


def on_loss(
    preds: List[torch.Tensor],
    means: List[torch.Tensor],
    norms: List[torch.Tensor],
) -> torch.Tensor:
    """
    On-surface loss: push each point towards the horizon mean,
    normalised by the local gradient norm.

    L_on = Σ_i  mean( |(p - mean_p) / ‖∇f‖| )
    """
    return sum(
        ((p - m) / n).abs().mean()
        for p, m, n in zip(preds, means, norms)
    )


def orie_loss(
    scalar_grad: torch.Tensor,
    orie_tensor: torch.Tensor,
) -> torch.Tensor:
    """
    Orientation loss: cosine similarity between predicted gradient and
    measured normals / orientations.

    L_or = mean( 1 - cos(∇f, n̂) )

    Note: only the first 3 components of scalar_grad are used (xyz partial
    derivatives); fault features are excluded.
    """
    cosine = F.cosine_similarity(
        scalar_grad[-orie_tensor.shape[0]:, :3],
        orie_tensor,
        dim=1,
        eps=1e-8,
    )
    return torch.mean(1 - cosine)


def order_loss(
    preds: List[torch.Tensor],
    means: List[torch.Tensor],
    norms: List[torch.Tensor],
    groups: Optional[List[List[int]]] = None,
    above_w: float = 1.0,
    below_w: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Stratigraphic ordering loss.

    For each pair (i < j) – horizon i is above horizon j:
      above: ReLU( -(p_i - mean_j) / ‖∇f_i‖ )  → enforce p_i > mean_j
      below: ReLU(  (p_j - mean_i) / ‖∇f_j‖ )  → enforce p_j < mean_i

    Returns
    -------
    (L_above, L_below)  – already weighted.
    """
    loss_above = torch.tensor(0.0)
    loss_below = torch.tensor(0.0)

    if groups is None:
        groups = [list(range(len(preds)))]

    for group in groups:
        if len(group) < 2:
            continue
        for a in range(len(group)):
            i = group[a]
            for b in range(a + 1, len(group)):
                j = group[b]

                p_i, m_j, n_i = preds[i], means[j], norms[i]
                loss_above = loss_above + torch.relu(-(p_i - m_j) / n_i).mean()

                p_j, m_i, n_j = preds[j], means[i], norms[j]
                loss_below = loss_below + torch.relu((p_j - m_i) / n_j).mean()

    return above_w * loss_above, below_w * loss_below


def eikonal_loss(norm_grad: torch.Tensor) -> torch.Tensor:
    """
    Eikonal regularisation: encourage ‖∇f‖ ≈ 1.

    L_eik = mean( (‖∇f‖ - 1)² )
    """
    return torch.mean((norm_grad - 1.0) ** 2)


def laplacian_loss(
    y_pred: torch.Tensor,
    x_input: torch.Tensor,
) -> torch.Tensor:
    """
    Laplacian regularisation (smoothness): penalise large second-order
    derivatives.  Expensive – use sparingly.

    L_lap = mean( (∇²f)² )
    """
    grad1 = torch.autograd.grad(
        y_pred, x_input,
        grad_outputs=torch.ones_like(y_pred),
        create_graph=True,
    )[0]
    # sum of diagonal Hessian entries (trace)
    lap = sum(
        torch.autograd.grad(
            grad1[:, i].sum(), x_input, create_graph=True
        )[0][:, i]
        for i in range(3)
    )
    return torch.mean(lap ** 2)


# ──────────────────────────────────────────────────────────────────────────────
# Loss builder / registry
# ──────────────────────────────────────────────────────────────────────────────

_LOSS_REGISTRY: Dict[str, Callable] = {
    "var":      var_loss,
    "on":       on_loss,
    "orie":     orie_loss,
    "order":    order_loss,
    "eikonal":  eikonal_loss,
    "laplacian": laplacian_loss,
}


def register_loss(name: str, fn: Callable) -> None:
    """Register a custom loss function by name."""
    _LOSS_REGISTRY[name.lower()] = fn


def get_loss(name: str) -> Callable:
    key = name.lower()
    if key not in _LOSS_REGISTRY:
        raise ValueError(
            f"Unknown loss '{name}'. Available: {sorted(_LOSS_REGISTRY)}"
        )
    return _LOSS_REGISTRY[key]


def list_losses() -> list[str]:
    return sorted(_LOSS_REGISTRY)
