"""
geoinr_faults/api/create.py
────────────────────
Top-level factory functions exposed under the `geoinr_faults` namespace.
These mirror GemPy's `gp.create_geomodel` / `gp.map_stack_to_surfaces` style.
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Optional, Sequence, Union

from geoinr_faults.core.geomodel import GeoModel
from geoinr_faults.data.importer import (
    load_surface_points_df,
    build_surface_points,
    load_orientations,
    build_inference_grid,
)
from geoinr_faults.features.fault_features import FaultConfig, compute_fault_features
from geoinr_faults.features.strati import StratiConfig


# ──────────────────────────────────────────────────────────────────────────────
# create_geomodel
# ──────────────────────────────────────────────────────────────────────────────

def create_geomodel(
    project_name: str,
    extent: List[float],
    resolution: List[int] = [100, 100, 100],
    path_to_surface_points: Optional[str | Path] = None,
    path_to_orientations: Optional[str | Path] = None,
    horizon_formations: Optional[Sequence[str]] = None,
    refinement: int = 1,
) -> GeoModel:
    """
    Create and initialise a GeoModel.

    Parameters
    ----------
    project_name           : Name shown in plots / saved files.
    extent                 : [xmin, xmax, ymin, ymax, zmin, zmax]
    resolution             : [nx, ny, nz] inference grid.
    path_to_surface_points : CSV with X, Y, Z, formation columns.
    path_to_orientations   : CSV with orientation / gradient data.
    horizon_formations     : Ordered formation names (top → bottom).
                             If None, all non-fault formations are used.
    refinement             : Reserved for multi-scale workflows.

    Returns
    -------
    GeoModel  – ready for add_faults / compute_model.

    Example
    -------
    >>> data = mg.create_geomodel(
    ...     project_name='Amoco',
    ...     extent=[0, 2, 0, 1, 0, 1],
    ...     resolution=[200, 100, 150],
    ...     path_to_surface_points='Amoco_model.csv',
    ...     horizon_formations=['horizon1', 'horizon2', 'horizon3', 'horizon4'],
    ... )
    """
    model = GeoModel(
        project_name=project_name,
        extent=extent,
        resolution=resolution,
        refinement=refinement,
    )

    # load and build grid
    if path_to_surface_points is not None:
        load_surface_points_df(model, csv_path=path_to_surface_points)
    if path_to_orientations is not None:
        load_orientations(model, csv_path=path_to_orientations)

    build_inference_grid(model)

    # keep coordinates in original extent units; normalization is done lazily
    # when building NN input tensors.

    # Backwards-compatible path: if horizon_formations is provided, build a
    # single strati immediately. New code should call add_stratis() explicitly.
    if horizon_formations is not None and path_to_surface_points is not None:
        add_stratis(
            model,
            strati_configs=[StratiConfig(name="legacy", formations=list(horizon_formations))],
        )

    return model


# ──────────────────────────────────────────────────────────────────────────────
# add_faults
# ──────────────────────────────────────────────────────────────────────────────

# ---------------------------------------------------------------------------
# add_stratis
# ---------------------------------------------------------------------------

def add_stratis(
    model: GeoModel,
    strati_configs: Optional[List[Union[StratiConfig, dict]]] = None,
    *,
    x_col: str = "X",
    y_col: str = "Y",
    z_col: str = "Z",
    formation_col: str = "formation",
) -> GeoModel:
    """
    Define one or more independent stratigraphic stacks ("strati").

    Each strati contains an ordered list of formations (top -> bottom). Ordering
    constraints are applied within each strati, but not across strati.

    Side-effects
    ------------
    Populates model.surface_points, model.surf_masks, model.horizon_formations,
    model.strati_groups.

    Notes
    -----
    Call this before add_faults()/compute_model(), because they rely on
    model.surface_points and model.surf_masks.
    """
    if model.surface_points_df is None:
        raise ValueError(
            "surface_points_df is not set; pass path_to_surface_points to create_geomodel first."
        )

    df = model.surface_points_df
    all_formations = df[formation_col].unique().tolist()

    if strati_configs is None:
        formations = [f for f in all_formations if not str(f).lower().startswith("fault")]
        strati_configs = [StratiConfig(name="default", formations=list(formations))]

    configs = [
        StratiConfig.from_dict(c) if isinstance(c, dict) else c
        for c in strati_configs
    ]

    known = set(all_formations)
    for cfg in configs:
        if not cfg.formations:
            raise ValueError("Each StratiConfig must contain at least one formation.")
        missing = [f for f in cfg.formations if f not in known]
        if missing:
            raise ValueError(
                f"Unknown formations in strati '{cfg.name}': {missing}. Available: {sorted(known)}"
            )

    import numpy as np

    points_all = []
    masks_all = []
    horizon_formations: List[str] = []
    strati_groups: List[List[int]] = []

    offset = 0
    for cfg in configs:
        group: List[int] = []
        points_list, masks_list, offset = build_surface_points(
            model,
            cfg.formations,
            x_col=x_col,
            y_col=y_col,
            z_col=z_col,
            formation_col=formation_col,
            start_index=offset,
        )
        points_all.extend(points_list)
        for formation_name, mask in zip(cfg.formations, masks_list):
            masks_all.append(mask)
            horizon_formations.append(formation_name)
            group.append(len(masks_all) - 1)
        strati_groups.append(group)

    model.surface_points = np.concatenate(points_all, axis=0).astype(np.float32)
    model.surf_masks = masks_all
    model.horizon_formations = horizon_formations
    model.strati_groups = strati_groups
    model.extra["stratis"] = [{"name": c.name, "formations": list(c.formations)} for c in configs]

    return model


def add_faults(
    model: GeoModel,
    fault_configs: List[Union[FaultConfig, dict]],
    scale_value: float = 0.1,
) -> GeoModel:
    """
    Compute finite-fault features and attach them to the model.

    Parameters
    ----------
    model         : GeoModel created by create_geomodel.
    fault_configs : Ordered list of FaultConfig objects or plain dicts.
                    Faults that truncate others must come first.
    scale_value   : Feature scaling factor (default 0.1 per paper).

    Returns
    -------
    The same GeoModel with fault data populated.

    Example
    -------
    >>> data = mg.add_faults(data, fault_configs=[
    ...     mg.FaultConfig(name='fault4', lx=10, ly=10, lz=10),
    ...     mg.FaultConfig(name='fault1', lx=10, ly=10, lz=10,
    ...                    truncation=True, trunc_fault_name='fault4',
    ...                    hanging_trunc=True),
    ... ])
    """
    # accept dicts in addition to FaultConfig objects
    configs = [
        FaultConfig.from_dict(c) if isinstance(c, dict) else c
        for c in fault_configs
    ]
    return compute_fault_features(model, configs, scale_value=scale_value)

