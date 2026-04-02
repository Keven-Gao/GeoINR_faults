"""
geoinr_faults/data/importer.py
──────────────────────
Reads surface-point / orientation CSV files and builds the PyVista grid.
All results are stored in the GeoModel object.
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import pyvista as pv

from geoinr_faults.core.geomodel import GeoModel



# ──────────────────────────────────────────────────────────────────────────────
# public helpers used by api.create
# ──────────────────────────────────────────────────────────────────────────────

def load_surface_points_df(
    model: GeoModel,
    csv_path: str | Path,
) -> GeoModel:
    """
    Load the surface-point CSV into ``model.surface_points_df`` only.

    Higher-level API helpers (e.g. ``api.create.add_stratis``) can then decide
    how to group/order formations into independent stratigraphic stacks.
    """
    df = pd.read_csv(csv_path)
    model.surface_points_df = df
    return model


def build_surface_points(
    model: GeoModel,
    formations: Sequence[str],
    *,
    x_col: str = "X",
    y_col: str = "Y",
    z_col: str = "Z",
    formation_col: str = "formation",
    start_index: int = 0,
) -> Tuple[List[np.ndarray], List[np.ndarray], int]:
    """
    Build surface-point arrays + masks for an ordered list of *formations*.

    Masks returned here index into the concatenated order, so downstream code
    can safely do ``y_pred[mask]``.
    """
    if model.surface_points_df is None:
        raise ValueError("surface_points_df is not set; call load_surface_points_df first.")

    df = model.surface_points_df

    points_list: List[np.ndarray] = []
    masks_list: List[np.ndarray] = []
    offset = int(start_index)

    for formation in formations:
        mask = df[formation_col] == formation
        coords = df.loc[mask, [x_col, y_col, z_col]].values.astype(np.float32)
        if coords.shape[0] == 0:
            raise ValueError(f"Formation '{formation}' has no rows in surface_points_df.")

        points_list.append(coords)
        masks_list.append(np.arange(offset, offset + coords.shape[0], dtype=np.int64))
        offset += coords.shape[0]

    return points_list, masks_list, offset


def load_surface_points(
    model: GeoModel,
    csv_path: str | Path,
    horizon_formations: Optional[Sequence[str]] = None,
    x_col: str = "X",
    y_col: str = "Y",
    z_col: str = "Z",
    formation_col: str = "formation",
) -> GeoModel:
    """
    Load surface-point CSV and split rows by formation.

    Parameters
    ----------
    model              : GeoModel to populate.
    csv_path           : Path to the CSV file.
    horizon_formations : Ordered list of formation names that define stratigraphic
                         horizons (top-to-bottom).  If None, all non-fault
                         formations are used in the order they appear.
    x_col, y_col, z_col, formation_col : column names in the CSV.

    Side-effects
    ------------
    Sets model.surface_points_df, model.surface_points, model.surf_masks.
    """
    load_surface_points_df(model, csv_path=csv_path)
    df = model.surface_points_df
    assert df is not None

    all_formations = df[formation_col].unique().tolist()

    if horizon_formations is None:
        # auto-detect: everything that is NOT named 'fault*'
        horizon_formations = [f for f in all_formations
                               if not f.lower().startswith("fault")]

    points_list, masks_list, _ = build_surface_points(
        model,
        horizon_formations,
        x_col=x_col,
        y_col=y_col,
        z_col=z_col,
        formation_col=formation_col,
        start_index=0,
    )

    model.surface_points = np.concatenate(points_list, axis=0).astype(np.float32)
    model.surf_masks = masks_list
    model.horizon_formations = list(horizon_formations)
    model.strati_groups = [list(range(len(horizon_formations)))]
    return model


def load_orientations(
    model: GeoModel,
    csv_path: str | Path,
    x_col: str = "X",
    y_col: str = "Y",
    z_col: str = "Z",
    gx_col: str = "G_x",
    gy_col: str = "G_y",
    gz_col: str = "G_z",
) -> GeoModel:
    """
    Load orientation / gradient data from CSV.

    Side-effects
    ------------
    Sets model.orientations_df, model.orientation_points, model.orientations.
    """
    df = pd.read_csv(csv_path)
    model.orientations_df = df
    model.orientation_points = df[[x_col, y_col, z_col]].values.astype(np.float32)
    model.orientations = df[[gx_col, gy_col, gz_col]].values.astype(np.float32)
    return model


def build_inference_grid(
    model: GeoModel,
    resolution: Optional[List[int]] = None,
) -> GeoModel:
    """
    Build the regular 3-D inference grid and store as PyVista ImageData.

    Parameters
    ----------
    model      : GeoModel (uses model.extent and model.resolution).
    resolution : Override model.resolution if provided.

    Side-effects
    ------------
    Sets model.test_xyz, model.grid_mesh.
    """
    if resolution is not None:
        model.resolution = resolution

    ext = model.extent          # [xmin,xmax, ymin,ymax, zmin,zmax]
    res = model.resolution      # [nx, ny, nz]

    grid = pv.ImageData()
    grid.dimensions = res
    grid.origin = [ext[0], ext[2], ext[4]]
    grid.spacing = [
        (ext[1] - ext[0]) / (res[0] - 1),
        (ext[3] - ext[2]) / (res[1] - 1),
        (ext[5] - ext[4]) / (res[2] - 1),
    ]

    model.test_xyz = grid.points.copy()
    model.grid_mesh = grid
    return model


# ──────────────────────────────────────────────────────────────────────────────
# coordinate normalisation
# ──────────────────────────────────────────────────────────────────────────────

def normalize_xyz(
    data: np.ndarray,
    extent: List[float],
) -> np.ndarray:
    """
    Normalise XYZ coordinates to [-1, 1] per axis using the model extent.

    Parameters
    ----------
    data   : (N, 3) array.
    extent : [xmin, xmax, ymin, ymax, zmin, zmax]

    Returns
    -------
    Normalised copy of data.
    """
    out = data.copy().astype(np.float64)
    bounds = extent
    for i, (lo, hi) in enumerate([(bounds[0], bounds[1]),
                                   (bounds[2], bounds[3]),
                                   (bounds[4], bounds[5])]):
        mean = (lo + hi) / 2.0
        delta = hi - lo
        out[:, i] = (out[:, i] - mean) / delta * 2.0
    return out.astype(np.float32)



