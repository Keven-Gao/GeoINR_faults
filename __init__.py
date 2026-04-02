"""
geoinr_faults
══════
Implicit-surface geological modelling with finite-fault features
and neural-network solvers.

Quick-start
───────────
    import mygeo as mg

    # 1. Create model + load data
    data = mg.create_geomodel(
        project_name='Amoco',
        extent=[0, 2, 0, 1, 0, 1],
        resolution=[200, 100, 150],
        path_to_surface_points='Amoco_model.csv',
        horizon_formations=['horizon1', 'horizon2', 'horizon3', 'horizon4'],
    )

    # 2. Add faults  (omit if no faults)
    data = mg.add_faults(data, fault_configs=[
        mg.FaultConfig(name='fault4', lx=10, ly=10, lz=10),
        mg.FaultConfig(name='fault1', lx=10, ly=10, lz=10,
                       truncation=True, trunc_fault_name='fault4',
                       hanging_trunc=True),
    ])

    # 3. Compute / train
    geo_data = mg.compute_model(data)

    # 4. Visualise
    mg.plot_model(geo_data)
    mg.plot_loss(geo_data)
"""

# ── core ──────────────────────────────────────────────────────────────────────
from geoinr_faults.core.geomodel import GeoModel

# ── data helpers ──────────────────────────────────────────────────────────────
from geoinr_faults.data.importer import (
    load_surface_points,
    load_orientations,
    build_inference_grid,
    normalize_xyz,
)

# ── fault features ────────────────────────────────────────────────────────────
from geoinr_faults.features.fault_features import FaultConfig, compute_fault_features

# ── nn building blocks (for advanced users) ───────────────────────────────────
from geoinr_faults.inr.activations import get_activation, list_activations
from geoinr_faults.inr.losses import (
    get_loss, register_loss, list_losses,
    var_loss, on_loss, order_loss, orie_loss, eikonal_loss,
)
from geoinr_faults.inr.models import get_model, build_model, list_models, MLP, SIREN, ResNetMLP

# ── solver building blocks (for advanced users) ───────────────────────────────
from geoinr_faults.solvers.nn_solver import (
    build_input_tensors,
    build_network,
    train,
    predict,
)

# ── top-level API  (GemPy-style) ──────────────────────────────────────────────
from geoinr_faults.api.create import create_geomodel, add_faults
from geoinr_faults.api.compute import compute_model

# ── visualisation ─────────────────────────────────────────────────────────────
from geoinr_faults.visualization.visualize import (
    plot_model,
    plot_fault_features,
    plot_loss,
    plot_section,
    plot_summary,
)

__all__ = [
    # core
    "GeoModel",
    # data
    "load_surface_points", "load_orientations", "build_inference_grid", "normalize_xyz",
    # faults
    "FaultConfig", "compute_fault_features",
    # nn
    "get_activation", "list_activations",
    "get_loss", "register_loss", "list_losses",
    "var_loss", "on_loss", "order_loss", "orie_loss", "eikonal_loss",
    "get_model", "build_model", "list_models",
    "MLP", "SIREN", "ResNetMLP",
    # solver
    "build_input_tensors", "build_network", "train", "predict",
    # top-level API
    "create_geomodel", "add_faults", "compute_model",
    # visualisation
    "plot_model", "plot_fault_features", "plot_loss", "plot_section", "plot_summary",
]

__version__ = "0.1.0"

