"""
geoinr_faults/visualization/visualize.py
──────────────────────────────────
All visualisation helpers.  Each function accepts a GeoModel and optional
keyword overrides; none of them modify the model.

Backends
────────
  PyVista  – 3-D interactive rendering  (primary)
  Matplotlib – 2-D cross-sections and loss curves  (lightweight fallback)
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# helpers
# ──────────────────────────────────────────────────────────────────────────────

def _require_pyvista():
    try:
        import pyvista as pv
        return pv
    except ImportError:
        raise ImportError("PyVista is required for 3-D visualisation.  "
                          "Install with:  pip install pyvista")


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        raise ImportError("Matplotlib is required for 2-D plots.  "
                          "Install with:  pip install matplotlib")


def _build_rock_unit_ids(
    scalar_values: np.ndarray,
    iso_values: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert continuous scalar values into discrete rock-unit IDs.

    Each interval between neighbouring iso-values is assigned one integer unit.
    The full scalar range becomes N+1 units for N iso-values.
    """
    thresholds = np.unique(np.sort(np.asarray(iso_values, dtype=np.float64)))
    if thresholds.size == 0:
        raise ValueError("iso_values is empty; cannot build rock units.")

    values = np.asarray(scalar_values, dtype=np.float64).reshape(-1)
    unit_ids = np.digitize(values, bins=thresholds, right=False).astype(np.int32)
    return unit_ids, thresholds


def _resolve_rock_unit_annotations(
    model,
    n_units: int,
    rock_unit_labels: Optional[Dict[int, str] | Sequence[str]] = None,
) -> Dict[float, str]:
    """
    Build scalar-bar annotations for rock units.

    Priority:
    1) explicit `rock_unit_labels`
    2) model.extra['rock_unit_labels']
    3) model.horizon_formations when length matches n_units
    4) fallback: unit_0, unit_1, ...
    """
    if n_units <= 0:
        return {}

    labels_source = rock_unit_labels
    if labels_source is None and getattr(model, "extra", None) is not None:
        labels_source = model.extra.get("rock_unit_labels")

    labels: List[str] = []
    if isinstance(labels_source, dict):
        labels = [str(labels_source.get(i, f"unit_{i}")) for i in range(n_units)]
    elif labels_source is not None:
        labels = [str(x) for x in labels_source]
        if len(labels) != n_units:
            print(
                f"[warn] rock_unit_labels has {len(labels)} names, but {n_units} rock units are present; "
                "fallback to default labels."
            )
            labels = []

    if not labels:
        horizon_names = list(getattr(model, "horizon_formations", []) or [])
        if len(horizon_names) == n_units:
            labels = [str(x) for x in horizon_names]
        else:
            labels = [f"unit_{i}" for i in range(n_units)]

    return {float(i): labels[i] for i in range(n_units)}


# ──────────────────────────────────────────────────────────────────────────────
# 3-D volume + iso-surfaces  (main result)
# ──────────────────────────────────────────────────────────────────────────────

def plot_model(
    model,                                          # GeoModel
    show_volume: bool = False,
    show_scalar_field: Optional[bool] = True,
    show_rock_unit: bool = False,
    show_isosurfaces: bool = False,
    show_faults: bool = False,
    show_fault_meshes: Optional[bool] = None,
    show_points: bool = False,
    show_interface_points: Optional[bool] = None,
    show_fault_points: Optional[bool] = None,
    show_box: bool = True,
    scalar_cmap: str = "viridis",
    rock_unit_cmap: str = "tab20",
    rock_unit_field_name: str = "rock_unit",
    rock_unit_labels: Optional[Dict[int, str] | Sequence[str]] = None,
    rock_unit_opacity: Optional[float] = None,
    scalar_opacity: float = 1.0,
    fault_color: str = "white",
    fault_opacity: float = 1.0,
    point_size: float = 10.0,
    point_color: str = "black",
    fault_point_color: Optional[str] = None,
    camera_position: Optional[str] = None,
    parallel_projection: bool = False,
    background: str = "white",
    window_size: Tuple[int, int] = (1000, 700),
    interactive: bool = True,
    screenshot: Optional[str] = None,
    notebook: bool = False,
) -> Any:
    """
    Interactive 3-D visualisation of the geological model.

    Parameters
    ----------
    model              : Trained GeoModel.
    show_volume        : Render the scalar field as a volume slice.
    show_scalar_field  : Alias of show_volume. If set, overrides show_volume.
    show_rock_unit     : Render discretized rock units from scalar + iso-values.
    show_isosurfaces   : Extract and render iso-surfaces per horizon.
    show_faults        : Render fault surface meshes.
    show_fault_meshes  : Alias of show_faults. If set, overrides show_faults.
    show_points        : Global toggle for training points.
    show_interface_points: Interface-point visibility. If None, follows show_points.
    show_fault_points  : Fault-point visibility. If None, follows show_points.
    show_box           : Draw model bounding box.
    scalar_cmap        : Matplotlib colormap name for the scalar field.
    rock_unit_cmap     : Colormap used for rock-unit categories.
    rock_unit_field_name: Name of the generated grid scalar field.
    rock_unit_labels   : Rock-unit legend labels. Either {unit_id: name} or
                         ordered names with length equal to unit count.
    rock_unit_opacity  : Opacity of rock-unit rendering (defaults to scalar_opacity).
    scalar_opacity     : Opacity of the volume mesh.
    fault_color        : Colour string for fault surfaces.
    fault_opacity      : Opacity of fault surfaces.
    point_size         : Size of training-point spheres.
    point_color        : Colour of training points.
    fault_point_color  : Colour of fault points (defaults to point_color).
    camera_position    : Initial camera ('xz', 'xy', 'yz' or list).
    parallel_projection: Use orthographic projection.
    background         : Background colour.
    window_size        : (width, height) in pixels.
    interactive        : Block until window is closed.
    screenshot         : If a path string, save PNG before showing.

    Returns
    -------
    pv.Plotter instance (closed if interactive=True).
    """
    pv = _require_pyvista()

    plotter = pv.Plotter(window_size=list(window_size), notebook=notebook)
    plotter.background_color = background

    ext = model.extent
    show_volume = bool(show_scalar_field) if show_scalar_field is not None else show_volume
    show_faults = bool(show_fault_meshes) if show_fault_meshes is not None else show_faults
    split_toggles_set = (show_interface_points is not None) or (show_fault_points is not None)
    show_interface_points = show_points if show_interface_points is None else show_interface_points
    show_fault_points = show_points if show_fault_points is None else show_fault_points
    if fault_point_color is None:
        fault_point_color = point_color
    if rock_unit_opacity is None:
        rock_unit_opacity = scalar_opacity

    # ── scalar field volume ──────────────────────────────────────────────────
    if show_rock_unit and (model.grid_mesh is None or "scalar" not in model.grid_mesh.point_data):
        print("[warn] show_rock_unit=True but scalar field is missing on grid_mesh.")

    if show_rock_unit and model.grid_mesh is not None and "scalar" in model.grid_mesh.point_data:
        if not model.iso_surfaces:
            print("[warn] show_rock_unit=True but model.iso_surfaces is empty; fallback to scalar field.")
        else:
            unit_ids, thresholds = _build_rock_unit_ids(
                model.grid_mesh.point_data["scalar"],
                model.iso_surfaces,
            )
            model.grid_mesh.point_data[rock_unit_field_name] = unit_ids
            if getattr(model, "extra", None) is not None:
                model.extra["rock_unit_thresholds"] = thresholds.tolist()
                model.extra["rock_unit_count"] = int(unit_ids.max() + 1) if unit_ids.size > 0 else 0
                model.extra["rock_unit_field_name"] = rock_unit_field_name

            n_units = int(unit_ids.max() + 1) if unit_ids.size > 0 else 1
            annotations = _resolve_rock_unit_annotations(
                model,
                n_units=n_units,
                rock_unit_labels=rock_unit_labels,
            )
            if getattr(model, "extra", None) is not None:
                model.extra["rock_unit_labels"] = [annotations[float(i)] for i in range(n_units)]
            plotter.add_mesh(
                model.grid_mesh,
                scalars=rock_unit_field_name,
                cmap=rock_unit_cmap,
                opacity=rock_unit_opacity,
                categories=True,
                n_colors=max(1, n_units),
                annotations=annotations,
                clim=[-0.5, max(0.5, n_units - 0.5)],
                scalar_bar_args={"title": "rock_unit"},
                show_scalar_bar=True,
            )

    if (
        (not show_rock_unit or not model.iso_surfaces)
        and show_volume
        and model.grid_mesh is not None
        and "scalar" in model.grid_mesh.point_data
    ):
        plotter.add_mesh(
            model.grid_mesh,
            scalars="scalar",
            cmap=scalar_cmap,
            opacity=scalar_opacity,
            show_scalar_bar=True,
        )

    # ── iso-surfaces ─────────────────────────────────────────────────────────
    if show_isosurfaces and model.iso_surfaces and model.grid_mesh is not None:
        if "scalar" not in model.grid_mesh.point_data and model.predictions is not None:
            model.grid_mesh.point_data["scalar"] = np.asarray(model.predictions).ravel()

        if "scalar" not in model.grid_mesh.point_data:
            print("[warn] show_isosurfaces=True but no 'scalar' field exists on grid_mesh.")
        else:
            iso_mesh = model.grid_mesh.contour(
                isosurfaces=model.iso_surfaces,
                scalars="scalar",
            )
            if iso_mesh.n_points > 0:
                plotter.add_mesh(
                    iso_mesh,
                    cmap=scalar_cmap,
                    line_width=5,
                    show_scalar_bar=False,
                )

    # ── fault meshes ─────────────────────────────────────────────────────────
    if show_faults:
        for name, mesh in model.fault_meshes.items():
            if mesh is not None:
                plotter.add_mesh(
                    mesh,
                    color=fault_color,
                    opacity=fault_opacity,
                    show_edges=False,
                    line_width=2,
                )

    # ── training points ───────────────────────────────────────────────────────
    added_split_points = False
    if (
        model.surface_points_df is not None
        and "formation" in model.surface_points_df.columns
        and all(c in model.surface_points_df.columns for c in ("X", "Y", "Z"))
    ):
        df = model.surface_points_df
        formation_series = df["formation"].astype(str)
        fault_mask = formation_series.str.lower().str.startswith("fault")

        if show_interface_points:
            pts_interface = df.loc[~fault_mask, ["X", "Y", "Z"]].values
            if pts_interface.shape[0] > 0:
                plotter.add_points(
                    pts_interface,
                    render_points_as_spheres=True,
                    point_size=point_size,
                    color=point_color,
                )
                added_split_points = True

        if show_fault_points:
            pts_fault = df.loc[fault_mask, ["X", "Y", "Z"]].values
            if pts_fault.shape[0] > 0:
                plotter.add_points(
                    pts_fault,
                    render_points_as_spheres=True,
                    point_size=point_size,
                    color=fault_point_color,
                )
                added_split_points = True

    if not split_toggles_set and not added_split_points and show_points and model.surface_points is not None:
        plotter.add_points(
            model.surface_points,
            render_points_as_spheres=True,
            point_size=point_size,
            color=point_color,
        )

    # ── bounding box ──────────────────────────────────────────────────────────
    if show_box:
        plotter.add_mesh(
            pv.Box(bounds=[ext[0], ext[1], ext[2], ext[3], ext[4], ext[5]]),
            color="k",
            opacity=0.02,
        )

    plotter.add_axes()
    plotter.camera.parallel_projection = parallel_projection
    plotter.camera_position = camera_position

    if screenshot:
        plotter.screenshot(screenshot)

    plotter.show(interactive_update=not interactive)
    if interactive:
        plotter.close()
    return plotter


# ──────────────────────────────────────────────────────────────────────────────
# fault features on the grid
# ──────────────────────────────────────────────────────────────────────────────

def plot_fault_features(
    model,
    fault_name: Optional[str] = None,
    cmap: str = "RdBu",
    window_size: Tuple[int, int] = (1000, 700),
    interactive: bool = True,
    notebook: bool = False,
    camera_position: Any = "xz",
) -> Any:
    """
    Visualise the fault-feature scalar fields stored in model.grid_mesh.

    Parameters
    ----------
    fault_name : If given, show only that fault's feature field.
                 If None, open a separate window for every fault.
    notebook   : Forwarded to pyvista.Plotter(notebook=...).
    camera_position : Initial camera position (e.g. 'xz', 'xy', 'yz', or custom tuple/list).
    """
    pv = _require_pyvista()
    names = ([fault_name] if fault_name else model.fault_names) or []
    plotters = []
    for name in names:
        key = f"fault_feature_{name}"
        if model.grid_mesh is None or key not in model.grid_mesh.point_data:
            print(f"[warn] No grid data for fault '{name}'; skipping.")
            continue
        p = pv.Plotter(window_size=list(window_size), notebook=notebook)
        p.add_mesh(model.grid_mesh, scalars=key, cmap=cmap, show_scalar_bar=True)
        p.title = f"Fault feature: {name}"
        p.add_axes()
        p.camera_position = camera_position
        p.show(interactive_update=not interactive)
        if interactive:
            p.close()
        plotters.append(p)
    return plotters


# ──────────────────────────────────────────────────────────────────────────────
# loss curves
# ──────────────────────────────────────────────────────────────────────────────

def plot_fault_meshes(
    model,
    fault_names: Optional[Sequence[str]] = None,
    colors: Optional[Dict[str, str] | Sequence[str]] = None,
    opacity: float = 1.0,
    camera_position: str = "xz",
    background: str = "white",
    window_size: Tuple[int, int] = (1000, 700),
    interactive: bool = True,
    notebook: bool = False,
) -> Any:
    """
    Plot fault surface meshes only, with optional per-fault colours.

    Parameters
    ----------
    fault_names : Fault names to show. If None, plot all model.fault_names.
    colors      : Either {fault_name: color} or a list of colors used in order.
    opacity     : Mesh opacity.
    """
    pv = _require_pyvista()

    names = list(fault_names) if fault_names is not None else list(model.fault_names)
    if not names:
        print("[warn] No fault names found in model.")
        return None

    palette = ["red", "blue", "green", "yellow", "cyan", "magenta", "orange", "white"]
    plotter = pv.Plotter(window_size=list(window_size), notebook=notebook)
    plotter.background_color = background

    n_added = 0
    for i, name in enumerate(names):
        mesh = model.fault_meshes.get(name)
        if mesh is None:
            print(f"[warn] No mesh found for fault '{name}'; skipping.")
            continue

        if isinstance(colors, dict):
            color = colors.get(name, palette[i % len(palette)])
        elif colors is not None and len(colors) > 0:
            color = colors[i % len(colors)]
        else:
            color = palette[i % len(palette)]

        plotter.add_mesh(mesh, color=color, opacity=opacity)
        n_added += 1

    if n_added == 0:
        print("[warn] No fault meshes were added to the plot.")
        return plotter

    plotter.add_axes()
    plotter.camera_position = camera_position
    plotter.show(interactive_update=not interactive)

    if interactive:
        plotter.close()
    return plotter


def plot_input_data(
    model,
    show_interface_points: bool = True,
    show_fault_points: bool = True,
    show_orientations: bool = True,
    formation_colors: Optional[Dict[str, str]] = None,
    interface_color: str = "black",
    fault_colors: Optional[Dict[str, str] | Sequence[str]] = None,
    orientation_color: str = "red",
    interface_point_size: float = 12.0,
    fault_point_size: float = 12.0,
    orientation_scale: Optional[float] = None,
    orientation_scale_by_magnitude: bool = True,
    normalize_orientation_vectors: bool = False,
    camera_position: str = "xz",
    background: str = "white",
    window_size: Tuple[int, int] = (1000, 700),
    interactive: bool = True,
    notebook: bool = False,
    show_formation_legend: bool = True,
    legend_location: str = "lower right",
    legend_size: Tuple[float, float] = (0.22, 0.22),
    legend_font_size: int = 5,
) -> Any:
    """
    Visualize input constraints: interface points, fault points, and
    orientation vectors (as arrows).

    Notes
    -----
    - Interface/fault points are split by formation name. Any formation that
      starts with 'fault' (case-insensitive) is treated as a fault point set.
    - Orientation arrows use model.orientation_points and model.orientations.
    - Interface/orientation points with the same formation use the same color.
    """
    pv = _require_pyvista()

    plotter = pv.Plotter(window_size=list(window_size), notebook=notebook)
    plotter.background_color = background
    legend_colors: Dict[str, str] = {}

    palette = [
        "tab:blue",
        "tab:orange",
        "tab:green",
        "tab:red",
        "tab:purple",
        "tab:brown",
        "tab:pink",
        "tab:gray",
        "tab:olive",
        "tab:cyan",
    ]

    # Build a shared formation -> color map for interface/orientation datasets.
    formation_color_map: Dict[str, str] = {}
    if formation_colors:
        formation_color_map.update(formation_colors)

    ordered_formations: List[str] = []
    if model.surface_points_df is not None and "formation" in model.surface_points_df.columns:
        s_formations = (
            model.surface_points_df["formation"]
            .astype(str)
            .loc[~model.surface_points_df["formation"].astype(str).str.lower().str.startswith("fault")]
            .drop_duplicates()
            .tolist()
        )
        ordered_formations.extend(s_formations)
    if model.orientations_df is not None and "formation" in model.orientations_df.columns:
        o_formations = model.orientations_df["formation"].astype(str).drop_duplicates().tolist()
        ordered_formations.extend([f for f in o_formations if f not in ordered_formations])

    auto_idx = 0
    for formation in ordered_formations:
        if formation not in formation_color_map:
            formation_color_map[formation] = palette[auto_idx % len(palette)]
            auto_idx += 1

    if model.surface_points_df is not None and "formation" in model.surface_points_df.columns:
        df = model.surface_points_df
        x_col, y_col, z_col = "X", "Y", "Z"
        if all(c in df.columns for c in [x_col, y_col, z_col]):
            formation_series = df["formation"].astype(str)
            fault_mask = formation_series.str.lower().str.startswith("fault")

            if show_interface_points:
                interface_df = df.loc[~fault_mask, [x_col, y_col, z_col, "formation"]]
                if interface_df.shape[0] > 0:
                    formations = interface_df["formation"].astype(str).drop_duplicates().tolist()
                    for formation in formations:
                        pts = interface_df.loc[
                            interface_df["formation"].astype(str) == formation,
                            [x_col, y_col, z_col],
                        ].values
                        if pts.shape[0] == 0:
                            continue
                        color = formation_color_map.get(formation, interface_color)
                        legend_colors.setdefault(formation, color)
                        plotter.add_points(
                            pts,
                            render_points_as_spheres=True,
                            point_size=interface_point_size,
                            color=color,
                        )

            if show_fault_points:
                fault_df = df.loc[fault_mask, [x_col, y_col, z_col, "formation"]]
                if fault_df.shape[0] > 0:
                    fault_names = fault_df["formation"].astype(str).unique().tolist()
                    for i, fault_name in enumerate(fault_names):
                        pts = fault_df.loc[fault_df["formation"] == fault_name, [x_col, y_col, z_col]].values
                        if pts.shape[0] == 0:
                            continue
                        if isinstance(fault_colors, dict):
                            color = fault_colors.get(fault_name, palette[i % len(palette)])
                        elif fault_colors is not None and len(fault_colors) > 0:
                            color = fault_colors[i % len(fault_colors)]
                        else:
                            color = palette[i % len(palette)]
                        legend_colors.setdefault(fault_name, color)
                        plotter.add_points(
                            pts,
                            render_points_as_spheres=False,
                            point_size=fault_point_size,
                            color=color,
                        )
        else:
            print("[warn] surface_points_df is missing X/Y/Z columns; point visualization skipped.")
    else:
        print("[warn] surface_points_df or 'formation' column missing; point visualization skipped.")

    if show_orientations:
        orientation_points = getattr(model, "orientation_points", None)
        orientation_vectors = getattr(model, "orientations", None)
        if orientation_points is None and model.orientations_df is not None:
            df_orie = model.orientations_df
            if all(c in df_orie.columns for c in ["X", "Y", "Z"]):
                orientation_points = df_orie[["X", "Y", "Z"]].values
        if orientation_vectors is None and model.orientations_df is not None:
            df_orie = model.orientations_df
            if all(c in df_orie.columns for c in ["G_x", "G_y", "G_z"]):
                orientation_vectors = df_orie[["G_x", "G_y", "G_z"]].values

        if orientation_points is not None and orientation_vectors is not None and len(orientation_points) > 0:
            pts = np.asarray(orientation_points, dtype=np.float32)
            vecs = np.asarray(orientation_vectors, dtype=np.float32)
            if normalize_orientation_vectors:
                norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-8
                vecs = vecs / norms

            if orientation_scale is None:
                ext = model.extent
                spans = [ext[1] - ext[0], ext[3] - ext[2], ext[5] - ext[4]]
                orientation_scale = 0.06 * max(spans)

            # If formation exists in orientations_df, color arrows by formation
            # using the same mapping as interface points.
            can_group_by_formation = (
                model.orientations_df is not None
                and "formation" in model.orientations_df.columns
                and len(model.orientations_df) == pts.shape[0]
            )
            if can_group_by_formation:
                df_orie = model.orientations_df
                formations = df_orie["formation"].astype(str).drop_duplicates().tolist()
                for formation in formations:
                    idx = np.where(df_orie["formation"].astype(str).values == formation)[0]
                    if idx.size == 0:
                        continue
                    pdata = pv.PolyData(pts[idx])
                    pdata["vectors"] = vecs[idx]
                    if orientation_scale_by_magnitude:
                        pdata["vector_mag"] = np.linalg.norm(vecs[idx], axis=1)
                        arrows = pdata.glyph(orient="vectors", scale="vector_mag", factor=orientation_scale)
                    else:
                        arrows = pdata.glyph(orient="vectors", scale=False, factor=orientation_scale)
                    color = formation_color_map.get(formation, orientation_color)
                    legend_colors.setdefault(formation, color)
                    plotter.add_mesh(arrows, color=color)
            else:
                pdata = pv.PolyData(pts)
                pdata["vectors"] = vecs

                if orientation_scale_by_magnitude:
                    pdata["vector_mag"] = np.linalg.norm(vecs, axis=1)
                    arrows = pdata.glyph(orient="vectors", scale="vector_mag", factor=orientation_scale)
                else:
                    arrows = pdata.glyph(orient="vectors", scale=False, factor=orientation_scale)

                plotter.add_mesh(arrows, color=orientation_color)
                legend_colors.setdefault("orientation", orientation_color)
        else:
            print("[warn] No orientation points/vectors found; orientation visualization skipped.")

    if show_formation_legend and legend_colors:
        labels = [[name, color] for name, color in legend_colors.items()]
        legend_actor = plotter.add_legend(
            labels=labels,
            bcolor="white",
            border=True,
            size=list(legend_size),
            loc=legend_location,
        )
        if legend_actor is not None and hasattr(legend_actor, "GetEntryTextProperty"):
            legend_actor.GetEntryTextProperty().SetFontSize(int(legend_font_size))

    plotter.add_axes()
    plotter.camera_position = camera_position
    plotter.show(interactive_update=not interactive)

    if interactive:
        plotter.close()
    return plotter


def plot_loss(
    model,
    components: Optional[List[str]] = None,
    log_scale: bool = True,
    figsize: Tuple[float, float] = (10, 4),
    save_path: Optional[str] = None,
    show: bool = True,
    return_fig: bool = False,
) -> Any:
    """
    Plot training loss curves stored in model.loss_history.

    Parameters
    ----------
    components : List of keys to plot (default: all non-zero components).
    log_scale  : Use log-scale on the y-axis.
    save_path  : If given, save figure to this path.
    show       : If True, render immediately via matplotlib.
    return_fig : If True, return the Figure object.
    """
    plt = _require_matplotlib()

    if not model.loss_history:
        print("[warn] No loss history found – did you call mg.compute_model()?")
        return None

    history = model.loss_history
    if components is None:
        components = [k for k, v in history.items()
                      if v and any(x != 0.0 for x in v)]

    fig, ax = plt.subplots(figsize=figsize)
    for key in components:
        if key in history and history[key]:
            ax.plot(history[key], label=key)

    if log_scale:
        ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"Training Loss – {model.project_name}")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)

    if show:
        plt.show()
        # Avoid notebook auto-rendering the same open figure a second time.
        plt.close(fig)

    if return_fig:
        return fig
    return None


# ──────────────────────────────────────────────────────────────────────────────
# 2-D cross-section
# ──────────────────────────────────────────────────────────────────────────────

def plot_section(
    model,
    axis: str = "y",
    position: Optional[float] = None,
    cmap: str = "viridis",
    figsize: Tuple[float, float] = (10, 4.5),
    save_path: Optional[str] = None,
    show: bool = True,
    return_fig: bool = False,
) -> Any:
    """
    Plot a 2-D cross-section of the scalar field.

    Parameters
    ----------
    axis     : 'x', 'y', or 'z' – the axis to slice along.
    position : Value along *axis* at which to slice.
               Defaults to the midpoint of the model extent.
    """
    plt = _require_matplotlib()

    if model.predictions is None or model.grid_mesh is None:
        print("[warn] No predictions found – call mg.compute_model() first.")
        return None

    ext = model.extent
    res = model.resolution
    # VTK ImageData point ids vary fastest along X.
    # Use Fortran-order reshape to recover (nx, ny, nz) correctly from the
    # flattened prediction vector.
    scalar = model.predictions.reshape(res, order="F")

    axis_map = {"x": 0, "y": 1, "z": 2}
    ax_i = axis_map[axis.lower()]
    lo = ext[ax_i * 2]
    hi = ext[ax_i * 2 + 1]
    if position is None:
        position = (lo + hi) / 2

    # find nearest slice index
    idx = int((position - lo) / (hi - lo) * (res[ax_i] - 1))
    idx = max(0, min(idx, res[ax_i] - 1))

    if ax_i == 0:
        slice_data = scalar[idx, :, :]
        xlab, ylab = "Y", "Z"
        extent_2d = [ext[2], ext[3], ext[4], ext[5]]
    elif ax_i == 1:
        slice_data = scalar[:, idx, :]
        xlab, ylab = "X", "Z"
        extent_2d = [ext[0], ext[1], ext[4], ext[5]]
    else:
        slice_data = scalar[:, :, idx]
        xlab, ylab = "X", "Y"
        extent_2d = [ext[0], ext[1], ext[2], ext[3]]

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(
        slice_data.T,
        origin="lower",
        aspect="auto",
        cmap=cmap,
        extent=extent_2d,
    )
    plt.colorbar(im, ax=ax, label="Scalar field")
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_title(f"{model.project_name}  –  {axis.upper()} = {position:.3f}")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
    
    if show:
        plt.show()
        # Avoid notebook auto-rendering the same open figure a second time.
        plt.close(fig)

    if return_fig:
        return fig
    return None


# ──────────────────────────────────────────────────────────────────────────────
# convenience: all-in-one summary
# ──────────────────────────────────────────────────────────────────────────────

def plot_section_rock_unit(
    model,
    axis: str = "y",
    position: Optional[float] = None,
    cmap: str = "tab20",
    rock_unit_field_name: str = "rock_unit",
    rock_unit_labels: Optional[Dict[int, str] | Sequence[str]] = None,
    figsize: Tuple[float, float] = (5.5, 4),
    save_path: Optional[str] = None,
    show: bool = True,
    return_fig: bool = False,
) -> Any:
    """
    Plot a 2-D cross-section of the discretized rock-unit scalar field.

    Rock-unit IDs are built from the continuous scalar field using model.iso_surfaces.
    """
    plt = _require_matplotlib()
    from matplotlib import colors as mcolors

    if model.grid_mesh is None:
        print("[warn] No grid mesh found – call compute_model() first.")
        return None
    if "scalar" not in model.grid_mesh.point_data:
        print("[warn] No scalar field found on grid_mesh – call compute_model() first.")
        return None
    if not model.iso_surfaces:
        print("[warn] No iso_surfaces found – cannot derive rock units.")
        return None

    unit_ids, thresholds = _build_rock_unit_ids(
        model.grid_mesh.point_data["scalar"],
        model.iso_surfaces,
    )
    model.grid_mesh.point_data[rock_unit_field_name] = unit_ids

    n_units = int(unit_ids.max() + 1) if unit_ids.size > 0 else 1
    annotations = _resolve_rock_unit_annotations(
        model,
        n_units=n_units,
        rock_unit_labels=rock_unit_labels,
    )
    if getattr(model, "extra", None) is not None:
        model.extra["rock_unit_thresholds"] = thresholds.tolist()
        model.extra["rock_unit_count"] = int(n_units)
        model.extra["rock_unit_field_name"] = rock_unit_field_name
        model.extra["rock_unit_labels"] = [annotations[float(i)] for i in range(n_units)]

    ext = model.extent
    res = model.resolution
    unit_3d = unit_ids.reshape(res, order="F")

    axis_map = {"x": 0, "y": 1, "z": 2}
    if axis.lower() not in axis_map:
        raise ValueError("axis must be one of {'x', 'y', 'z'}.")
    ax_i = axis_map[axis.lower()]
    lo = ext[ax_i * 2]
    hi = ext[ax_i * 2 + 1]
    if position is None:
        position = (lo + hi) / 2

    idx = int((position - lo) / (hi - lo) * (res[ax_i] - 1))
    idx = max(0, min(idx, res[ax_i] - 1))

    if ax_i == 0:
        slice_data = unit_3d[idx, :, :]
        xlab, ylab = "Y", "Z"
        extent_2d = [ext[2], ext[3], ext[4], ext[5]]
    elif ax_i == 1:
        slice_data = unit_3d[:, idx, :]
        xlab, ylab = "X", "Z"
        extent_2d = [ext[0], ext[1], ext[4], ext[5]]
    else:
        slice_data = unit_3d[:, :, idx]
        xlab, ylab = "X", "Y"
        extent_2d = [ext[0], ext[1], ext[2], ext[3]]

    sampled_colors = plt.get_cmap(cmap)(np.linspace(0.0, 1.0, n_units))
    discrete_cmap = mcolors.ListedColormap(sampled_colors, name=f"{cmap}_{n_units}")
    boundaries = np.arange(-0.5, n_units + 0.5, 1.0)
    norm = mcolors.BoundaryNorm(boundaries, discrete_cmap.N)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(
        slice_data.T,
        origin="lower",
        aspect="auto",
        cmap=discrete_cmap,
        norm=norm,
        extent=extent_2d,
        interpolation="nearest",
    )
    tick_ids = np.arange(n_units, dtype=np.int32)
    cbar = plt.colorbar(
        im,
        ax=ax,
        boundaries=boundaries,
        ticks=tick_ids,
        spacing="proportional",
        label="Rock unit",
    )
    cbar.ax.set_yticklabels([annotations[float(i)] for i in tick_ids])

    #ax.set_xlabel(xlab)
    #ax.set_ylabel(ylab)
    #ax.set_title(f"{model.project_name}  –  {axis.upper()} = {position:.3f} (Rock units)")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)

    if show:
        plt.show()
        plt.close(fig)

    if return_fig:
        return fig
    return None


def plot_summary(
    model,
    sections: bool = True,
    loss_curve: bool = True,
    **plot_model_kwargs,
) -> None:
    """
    Show loss curve + optional cross-sections, then open 3-D viewer.
    """
    if loss_curve:
        plot_loss(model)

    if sections:
        for axis in ("x", "y", "z"):
            plot_section(model, axis=axis)

    plot_model(model, **plot_model_kwargs)
