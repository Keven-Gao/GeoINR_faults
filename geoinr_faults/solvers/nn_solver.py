"""
geoinr_faults/solvers/nn_solver.py
──────────────────────────
Builds the augmented input tensor (xyz + fault features), trains the
implicit-surface network, runs inference, and extracts iso-surface values.

All hyper-parameters are exposed as keyword arguments so users can override
them without touching library code.
"""

from __future__ import annotations
import time
from typing import Callable, Dict, List, Optional, Tuple, Type

import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F

from geoinr_faults.core.geomodel import GeoModel
from geoinr_faults.data.importer import normalize_xyz
from geoinr_faults.inr.activations import get_activation
from geoinr_faults.inr.losses import (
    compute_preds_means_norms,
    var_loss, on_loss, order_loss, orie_loss,
)
from geoinr_faults.inr.models import build_model, BaseGeoNet


# ──────────────────────────────────────────────────────────────────────────────
# Step 1 – assemble augmented input tensors
# ──────────────────────────────────────────────────────────────────────────────

def build_input_tensors(
    model: GeoModel,
    device: Optional[torch.device] = None,
    fault_scale: Optional[float] = None,
) -> GeoModel:
    """
    Concatenate normalised XYZ coordinates with scaled fault features to form
    the full input tensors for training and inference.

    Requires
    --------
    model.surface_points   (raw coordinates)
    model.test_xyz         (raw coordinates)
    model.train_fault_feats, model.domain_fault_feats
    model.fault_names

    Side-effects
    ------------
    Sets model.x_tensor (requires_grad=True) and model.test_x_tensor.
    """
    if model.surface_points is None or model.test_xyz is None or not model.surf_masks:
        raise ValueError(
            "Stratigraphic horizons are not set. Call add_stratis() after create_geomodel() "
            "to populate model.surface_points/model.surf_masks before compute_model()."
        )

    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    scale = fault_scale or model.extra.get("fault_scale", 0.1)
    xyz_train = normalize_xyz(model.surface_points, model.extent)
    xyz_test = normalize_xyz(model.test_xyz, model.extent)
    has_orientations = (
        model.orientation_points is not None
        and model.orientations is not None
        and model.orientation_points.shape[0] > 0
    )

    # --- training tensor ---
    parts_train = [torch.tensor(xyz_train, dtype=torch.float32)]
    for name in model.fault_names:
        feat = model.train_fault_feats[name] * scale
        parts_train.append(
            torch.tensor(feat, dtype=torch.float32).unsqueeze(1)
        )
    x_train_surface = torch.cat(parts_train, dim=1)

    if has_orientations:
        xyz_orie = normalize_xyz(model.orientation_points, model.extent)
        parts_orie = [torch.tensor(xyz_orie, dtype=torch.float32)]
        for name in model.fault_names:
            feat = model.orientation_fault_feats.get(name)
            if feat is None:
                raise ValueError(
                    f"Missing orientation fault features for '{name}'. "
                    "Call add_faults() after loading orientations."
                )
            parts_orie.append(torch.tensor(feat * scale, dtype=torch.float32).unsqueeze(1))
        x_train_orie = torch.cat(parts_orie, dim=1)
        x_train = torch.cat([x_train_surface, x_train_orie], dim=0).to(device).requires_grad_(True)
        model.extra["n_orientation_points"] = int(x_train_orie.shape[0])
    else:
        x_train = x_train_surface.to(device).requires_grad_(True)
        model.extra["n_orientation_points"] = 0

    model.extra["n_surface_points"] = int(x_train_surface.shape[0])

    # --- inference tensor ---
    parts_test = [torch.tensor(xyz_test, dtype=torch.float32)]
    for name in model.fault_names:
        feat = model.domain_fault_feats[name] * scale
        parts_test.append(
            torch.tensor(feat, dtype=torch.float32).unsqueeze(1)
        )
    x_test = torch.cat(parts_test, dim=1).to(device)

    model.x_tensor = x_train
    model.test_x_tensor = x_test
    model.extra["device"] = device
    return model


# ──────────────────────────────────────────────────────────────────────────────
# Step 2 – initialise network and optimiser
# ──────────────────────────────────────────────────────────────────────────────

def build_network(
    model: GeoModel,
    nn_model: str | type | BaseGeoNet = "mlp",
    hidden_dim: int = 256,
    n_layers: int = 3,
    activation: str | nn.Module = "softplus",
    softplus_beta: Optional[float] = None,
    optimizer_cls: type = torch.optim.AdamW,
    lr: float = 1e-3,
    seed: Optional[int] = None,
    **model_kwargs,
) -> GeoModel:
    """
    Instantiate the neural network and optimiser, store both in model.extra.

    Parameters
    ----------
    model        : GeoModel (must have x_tensor set).
    nn_model     : 'mlp' | 'siren' | 'resnet' | custom class | instance.
    hidden_dim   : Width of hidden layers.
    n_layers     : Number of hidden layers.
    activation   : Activation name or module (ignored when nn_model is SIREN).
    softplus_beta: Softplus beta when activation='softplus'. If None, use
                   activation registry default.
    optimizer_cls: torch optimiser class (default AdamW).
    lr           : Learning rate.
    seed         : Random seed for reproducibility. If None, do not force seed.
    model_kwargs : Extra kwargs forwarded to the model constructor.
    """
    device = model.extra.get("device",
                              torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    input_dim = model.x_tensor.shape[1]

    if seed is not None:
        torch.manual_seed(int(seed))

    if isinstance(activation, str) and activation.lower().strip() == "softplus" and softplus_beta is not None:
        activation = nn.Softplus(beta=float(softplus_beta))

    if isinstance(nn_model, nn.Module):
        net = nn_model.to(device)
    else:
        net = build_model(
            nn_model,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            activation=activation,
            **model_kwargs,
        ).to(device)

    opt = optimizer_cls(net.parameters(), lr=lr)

    model.model = net
    model.extra["optimizer"] = opt
    return model


# ──────────────────────────────────────────────────────────────────────────────
# Step 3 – training loop
# ──────────────────────────────────────────────────────────────────────────────

def train(
    model: GeoModel,
    epochs: int = 2000,
    loss_weights: Optional[Dict[str, float]] = None,
    orientation_tensor: Optional[torch.Tensor] = None,
    print_every: int = 500,
    verbose: bool = True,
) -> GeoModel:
    """
    Run the training loop.

    Parameters
    ----------
    model              : GeoModel with x_tensor, model, optimizer set.
    epochs             : Number of gradient steps.
    loss_weights       : Dict with keys 'var', 'on', 'above', 'below', 'orie'.
                         Default: all 1.0, orie 0.0 (disabled unless provided).
    orientation_tensor : (M, 3) gradient / normal measurements; if provided,
                         orientation loss is added automatically.
    print_every        : Log interval (0 = silent).
    verbose            : Whether to print progress.

    Side-effects
    ------------
    Populates model.loss_history, model.best_params.
    """
    weights = {"var": 1.0, "on": 1.0, "above": 1.0, "below": 1.0, "orie": 0.0}
    if loss_weights:
        weights.update(loss_weights)

    device = model.extra.get("device", torch.device("cpu"))
    net = model.model
    opt = model.extra["optimizer"]
    x_tensor = model.x_tensor
    masks = model.surf_masks

    if orientation_tensor is not None:
        n_orie_points = int(model.extra.get("n_orientation_points", 0))
        if n_orie_points == 0:
            raise ValueError(
                "orientation_tensor was provided but no orientation points are present in training input. "
                "Load orientations via create_geomodel(path_to_orientations=...) first."
            )
        if int(orientation_tensor.shape[0]) != n_orie_points:
            raise ValueError(
                f"orientation_tensor has {orientation_tensor.shape[0]} rows, "
                f"but training input contains {n_orie_points} orientation points."
            )
        orie_t = orientation_tensor.to(device)
        weights["orie"] = weights.get("orie", 1.0) or 1.0
    else:
        orie_t = None

    history: Dict[str, List[float]] = {
        "total": [], "var": [], "on": [], "above": [], "below": [], "orie": []
    }
    min_loss = float("inf")
    best_params = None

    t_start = time.time()

    for epoch in range(epochs):
        y_pred = net(x_tensor)

        # gradient of scalar field w.r.t. inputs
        scalar_grad = autograd.grad(
            outputs=y_pred,
            inputs=x_tensor,
            grad_outputs=torch.ones_like(y_pred),
            create_graph=True,
        )[0]

        # only xyz components for norm (first 3 dims)
        norm_grad = torch.norm(scalar_grad[:, :3], p=2, dim=1) + 1e-8

        preds, means, norms = compute_preds_means_norms(y_pred, norm_grad, masks)

        L_var = var_loss(preds)
        L_on  = on_loss(preds, means, norms)
        groups = model.strati_groups if getattr(model, "strati_groups", None) else None
        L_above, L_below = order_loss(
            preds, means, norms,
            groups=groups,
            above_w=weights["above"],
            below_w=weights["below"],
        )

        total = (
            weights["var"] * L_var
            + weights["on"] * L_on
            + L_above
            + L_below
        )

        L_orie = torch.tensor(0.0, device=total.device)
        if orie_t is not None and weights["orie"] > 0:
            L_orie = orie_loss(scalar_grad, orie_t)
            total = total + weights["orie"] * L_orie
            history["orie"].append(L_orie.item())
        else:
            history["orie"].append(0.0)

        if total.item() < min_loss:
            min_loss = total.item()
            best_params = {k: v.clone() for k, v in net.state_dict().items()}

        history["total"].append(total.item())
        history["var"].append(L_var.item())
        history["on"].append(L_on.item())
        history["above"].append(L_above.item())
        history["below"].append(L_below.item())

        opt.zero_grad()
        total.backward()
        opt.step()

        if verbose and print_every > 0 and (epoch + 1) % print_every == 0:
            print(
                f"Epoch [{epoch+1}/{epochs}]  "
                f"Loss: {total.item():.4f}  "
                f"Var: {L_var.item():.4f}  "
                f"On: {L_on.item():.4f}  "
                f"Above: {L_above.item():.4f}  "
                f"Below: {L_below.item():.4f}  "
                f"Orie: {L_orie.item():.4f}  "
                f"Best: {min_loss:.4f}"
            )

    elapsed = time.time() - t_start
    if verbose:
        print(f"\nTraining finished in {elapsed:.1f} s  |  Best loss: {min_loss:.4f}")

    model.loss_history = history
    model.best_params = best_params
    return model


def _flip_last_linear_sign(net: nn.Module) -> bool:
    """Flip output sign by negating the last linear layer weights/bias."""
    last_linear = None
    for module in net.modules():
        if isinstance(module, nn.Linear):
            last_linear = module

    if last_linear is None:
        return False

    with torch.no_grad():
        last_linear.weight.mul_(-1.0)
        if last_linear.bias is not None:
            last_linear.bias.mul_(-1.0)
    return True


def enforce_output_polarity(model: GeoModel, use_best: bool = True) -> GeoModel:
    """
    Stabilize scalar-field polarity across random initializations.

    The loss formulation can converge to globally sign-flipped solutions under
    some seeds. This helper detects a flipped polarity and corrects it by
    negating the final linear layer.
    """
    net = model.model
    if net is None or model.x_tensor is None or not model.surf_masks:
        return model

    if use_best and model.best_params is not None:
        net.load_state_dict(model.best_params)

    should_flip = False
    n_orie = int(model.extra.get("n_orientation_points", 0))

    # 1) If orientation constraints exist, use average cosine alignment first.
    if n_orie > 0 and model.orientations is not None and int(model.orientations.shape[0]) == n_orie:
        x_orie = model.x_tensor[-n_orie:].detach().clone().requires_grad_(True)
        y_orie = net(x_orie)
        grad_orie = autograd.grad(
            outputs=y_orie,
            inputs=x_orie,
            grad_outputs=torch.ones_like(y_orie),
            create_graph=False,
        )[0][:, :3]
        target_orie = torch.tensor(model.orientations, dtype=grad_orie.dtype, device=grad_orie.device)
        cos_mean = F.cosine_similarity(grad_orie, target_orie, dim=1, eps=1e-8).mean().item()
        model.extra["orientation_cosine_mean"] = float(cos_mean)
        if cos_mean < 0.0:
            should_flip = True

    # 2) Fallback: check stratigraphic mean ordering consistency.
    if not should_flip:
        n_surface = int(model.extra.get("n_surface_points", 0))
        if n_surface <= 0:
            n_surface = model.x_tensor.shape[0] - n_orie if n_orie > 0 else model.x_tensor.shape[0]

        with torch.no_grad():
            y_surface = net(model.x_tensor[:n_surface]).squeeze()

        means = [float(y_surface[mask].mean().item()) for mask in model.surf_masks]
        groups = model.strati_groups if model.strati_groups else [list(range(len(model.surf_masks)))]
        satisfied = 0
        violated = 0
        for group in groups:
            for a in range(len(group)):
                i = group[a]
                for b in range(a + 1, len(group)):
                    j = group[b]
                    if means[i] > means[j]:
                        satisfied += 1
                    else:
                        violated += 1
        model.extra["polarity_satisfied_pairs"] = int(satisfied)
        model.extra["polarity_violated_pairs"] = int(violated)
        if violated > satisfied:
            should_flip = True

    flipped = False
    if should_flip:
        flipped = _flip_last_linear_sign(net)

    if use_best and model.best_params is not None:
        model.best_params = {k: v.detach().clone() for k, v in net.state_dict().items()}

    model.extra["polarity_applied_to_best"] = bool(use_best and model.best_params is not None)
    model.extra["polarity_flipped"] = bool(flipped)
    return model


# ──────────────────────────────────────────────────────────────────────────────
# Step 4 – inference
# ──────────────────────────────────────────────────────────────────────────────

def predict(model: GeoModel, use_best: bool = True) -> GeoModel:
    """
    Run the trained network over the full inference grid.

    Parameters
    ----------
    model    : GeoModel with trained model and test_x_tensor.
    use_best : If True, load best_params before inference.

    Side-effects
    ------------
    Sets model.predictions (numpy), model.iso_surfaces,
    model.grid_mesh.point_data['scalar'].
    """
    net = model.model
    device = model.extra.get("device", torch.device("cpu"))

    if use_best and model.best_params is not None:
        net.load_state_dict(model.best_params)

    net.eval()
    with torch.no_grad():
        t0 = time.time()
        preds = net(model.test_x_tensor)
        print(f"Inference completed in {time.time() - t0:.4f} s")

    model.predictions = preds.cpu().numpy()

    # iso-surface values = mean scalar at each horizon's training points
    net.train()  # restore train mode
    with torch.no_grad():
        y_train = net(model.x_tensor)
    iso_vals = []
    for mask in model.surf_masks:
        iso_vals.append(float(y_train[mask].squeeze().mean().cpu()))
    model.iso_surfaces = iso_vals

    if model.grid_mesh is not None:
        model.grid_mesh.point_data["scalar"] = model.predictions.ravel()

    return model

