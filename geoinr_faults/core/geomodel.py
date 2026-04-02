"""
geoinr_faults/core/geomodel.py
─────────────────────
Central data container that travels through the entire pipeline.
Every stage of the workflow reads from and writes into a GeoModel instance.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class GeoModel:
    """
    Holds all state for one geological modelling run.

    Attributes
    ----------
    project_name : str
        Human-readable name shown in plots / saved files.
    extent : list[float]
        [xmin, xmax, ymin, ymax, zmin, zmax]
    resolution : list[int]
        [nx, ny, nz] – grid resolution for inference mesh.
    refinement : int
        Reserved for multi-resolution workflows.

    # --- populated by data.importer ---
    surface_points_df : pd.DataFrame | None
    surface_points     : np.ndarray | None   shape (N, 3)
    surf_masks         : list[np.ndarray]    index arrays, one per horizon
    orientations_df    : pd.DataFrame | None
    orientations       : np.ndarray | None   shape (M, 3)  (normals / gradients)

    # --- populated by api.create (grid) ---
    test_xyz           : np.ndarray | None   shape (Ng, 3)  full grid points
    grid_mesh          : pv.ImageData | None

    # --- populated by features.fault_features ---
    fault_names        : list[str]
    fault_configs      : dict  {name: FaultConfig}
    domain_fault_feats : dict  {name: np.ndarray}   for grid
    train_fault_feats  : dict  {name: np.ndarray}   for train points
    fault_meshes       : dict  {name: pv.PolyData}

    # --- populated by solvers.nn_solver ---
    x_tensor           : torch.Tensor | None
    test_x_tensor      : torch.Tensor | None
    model              : nn.Module | None
    best_params        : dict | None
    predictions        : np.ndarray | None
    iso_surfaces       : list[float]
    loss_history       : dict  {metric_name: list[float]}
    """

    # ── identity ──────────────────────────────────────────────────────────────
    project_name: str = "unnamed"
    extent: List[float] = field(default_factory=lambda: [0, 1, 0, 1, 0, 1])
    resolution: List[int] = field(default_factory=lambda: [50, 50, 50])
    refinement: int = 1

    # ── raw input data ─────────────────────────────────────────────────────────
    surface_points_df: Optional[Any] = None   # pd.DataFrame
    surface_points: Optional[np.ndarray] = None
    surf_masks: List[np.ndarray] = field(default_factory=list)
    # ordered horizon/formation names aligned with surf_masks (flat list)
    horizon_formations: List[str] = field(default_factory=list)
    # list of groups of horizon indices; order loss is applied within each group
    strati_groups: List[List[int]] = field(default_factory=list)
    orientations_df: Optional[Any] = None
    orientation_points: Optional[np.ndarray] = None
    orientations: Optional[np.ndarray] = None

    # ── inference grid ─────────────────────────────────────────────────────────
    test_xyz: Optional[np.ndarray] = None
    grid_mesh: Optional[Any] = None           # pv.ImageData

    # ── fault features ─────────────────────────────────────────────────────────
    fault_names: List[str] = field(default_factory=list)
    fault_configs: Dict[str, Any] = field(default_factory=dict)
    domain_fault_feats: Dict[str, np.ndarray] = field(default_factory=dict)
    train_fault_feats: Dict[str, np.ndarray] = field(default_factory=dict)
    orientation_fault_feats: Dict[str, np.ndarray] = field(default_factory=dict)
    fault_meshes: Dict[str, Any] = field(default_factory=dict)

    # ── nn / solver state ──────────────────────────────────────────────────────
    x_tensor: Optional[Any] = None            # torch.Tensor
    test_x_tensor: Optional[Any] = None
    model: Optional[Any] = None
    best_params: Optional[dict] = None
    predictions: Optional[np.ndarray] = None
    iso_surfaces: List[float] = field(default_factory=list)
    loss_history: Dict[str, List[float]] = field(default_factory=dict)

    # ── misc ──────────────────────────────────────────────────────────────────
    extra: Dict[str, Any] = field(default_factory=dict)   # user-defined extras

    # ─────────────────────────────── helpers ──────────────────────────────────
    @property
    def n_faults(self) -> int:
        return len(self.fault_names)

    @property
    def n_horizons(self) -> int:
        return len(self.surf_masks)

    @property
    def n_stratis(self) -> int:
        if self.strati_groups:
            return len(self.strati_groups)
        return 1 if self.n_horizons > 0 else 0

    @property
    def horizons_per_strati(self) -> List[int]:
        if self.strati_groups:
            return [len(group) for group in self.strati_groups]
        if self.n_horizons > 0:
            return [self.n_horizons]
        return []

    @property
    def input_dim(self) -> int:
        """Feature dimension fed to the network: 3 (xyz) + n_faults."""
        return 3 + self.n_faults

    def __repr__(self) -> str:
        return (
            f"GeoModel('{self.project_name}', "
            f"extent={self.extent}, "
            f"resolution={self.resolution}, "
            f"stratis={self.n_stratis}, "
            f"horizons={self.horizons_per_strati}, "
            f"faults={self.n_faults})"
        )
