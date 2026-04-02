"""
geoinr_faults/features/fault_features.py
─────────────────────────────────
Wraps FiniteFault and utils to compute fault features for the full domain
grid and the training points, storing results in a GeoModel.

A "FaultConfig" dataclass holds every per-fault parameter so callers can
pass a plain dict (which is auto-promoted) or a typed object.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from geoinr_faults.core.geomodel import GeoModel


# ──────────────────────────────────────────────────────────────────────────────
# FaultConfig  – one entry per fault
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class FaultConfig:
    """
    All parameters needed to fully describe a single finite fault.

    Parameters
    ----------
    name            : Identifier, must match the 'formation' column in the CSV.
    lx, ly, lz      : Ellipsoid semi-axes (influence radius along strike,
                      dip, and normal directions).
    dmax            : Maximum displacement scalar.
    lam             : RBF decay parameter (λ).
    rbf_kwargs      : kwargs forwarded to scipy RBF interpolator.
    Normal_fault    : True = normal fault; False = reverse / strike-slip.
    truncation      : Whether to apply truncation at another fault.
    trunc_fault_name: Name of the truncating fault (must be computed first).
    hanging_trunc   : Which side of the truncating fault to remove.
    """
    name: str = ""
    lx: float = 10.0
    ly: float = 10.0
    lz: float = 10.0
    dmax: float = 1.0
    lam: float = 0.5
    theta: Optional[float] = None
    phi: Optional[float] = None
    X0: Optional[float] = None
    Y0: Optional[float] = None
    Z0: Optional[float] = None
    rbf_kwargs: Dict[str, Any] = field(
        default_factory=lambda: {"kernel": "cubic", "epsilon": 1, "smoothing": 0.0}
    )
    Normal_fault: bool = False
    truncation: bool = False
    trunc_fault_name: Optional[str] = None   # name of the truncating fault
    hanging_trunc: bool = True

    @classmethod
    def from_dict(cls, d: dict) -> "FaultConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ──────────────────────────────────────────────────────────────────────────────
# main pipeline function
# ──────────────────────────────────────────────────────────────────────────────

def compute_fault_features(
    model: GeoModel,
    fault_configs: List[FaultConfig | dict],
    scale_value: float = 0.1,
) -> GeoModel:
    """
    Compute finite-fault features for every fault in *fault_configs* and
    append them to the model.

    The faults are processed **in the order given**, so a fault that
    truncates another must appear earlier in the list.

    Parameters
    ----------
    model         : GeoModel (must already contain surface_points_df,
                    test_xyz, surface_points).
    fault_configs : List of FaultConfig objects (or plain dicts).
    scale_value   : Multiplier applied to raw fault features before training
                    (paper recommendation: 0.1).

    Side-effects
    ------------
    Populates model.fault_names, model.fault_configs,
    model.domain_fault_feats, model.train_fault_feats, model.fault_meshes.
    Also stores scale_value in model.extra['fault_scale'].
    """
    # lazy imports so the rest of the framework works without faults installed
    # (Importing as `faults.*` assumes a top-level module named `faults`, which
    # breaks when GeoINR is used as a package. Import from our package instead.)
    from geoinr_faults.features.faults.Fault_feature_encoder import FiniteFault
    import geoinr_faults.features.faults.utils as utils

    model.fault_names = []
    model.fault_configs = {}
    model.domain_fault_feats = {}
    model.train_fault_feats = {}
    model.orientation_fault_feats = {}
    model.fault_meshes = {}
    model.extra["fault_scale"] = scale_value

    df = model.surface_points_df
    if model.surface_points is None or model.test_xyz is None:
        raise ValueError(
            "surface_points/test_xyz are not set. Call add_stratis() (and create_geomodel()) before add_faults()."
        )

    for cfg in fault_configs:
        if isinstance(cfg, dict):
            cfg = FaultConfig.from_dict(cfg)

        name = cfg.name
        fault_pts = df.loc[df["formation"] == name, ["X", "Y", "Z"]].values.astype(np.float32)

        # ── geometry ──────────────────────────────────────────────────────────
        centroid, normal_vec, _ = utils.ellipsoid_parameters(fault_pts)
        phi_deg, theta_deg = utils.normal_to_azimuth_dip(normal_vec)
        theta_auto = np.deg2rad(phi_deg)
        phi_auto = np.deg2rad(theta_deg)
        theta_rad = float(cfg.theta) if cfg.theta is not None else float(theta_auto)
        phi_rad = float(cfg.phi) if cfg.phi is not None else float(phi_auto)
        X0 = float(cfg.X0) if cfg.X0 is not None else float(centroid[0])
        Y0 = float(cfg.Y0) if cfg.Y0 is not None else float(centroid[1])
        Z0 = float(cfg.Z0) if cfg.Z0 is not None else float(centroid[2])

        # ── build truncation kwargs if needed ─────────────────────────────────
        trunc_kwargs = {}
        trunc_mesh_arg = None
        if cfg.truncation and cfg.trunc_fault_name:
            tf_name = cfg.trunc_fault_name
            tf_cfg_raw = model.fault_configs[tf_name]
            tf_pts = df.loc[df["formation"] == tf_name, ["X", "Y", "Z"]].values.astype(np.float32)
            tf_centroid, tf_normal, _ = utils.ellipsoid_parameters(tf_pts)
            tf_phi, tf_theta = utils.normal_to_azimuth_dip(tf_normal)
            tf_theta_rad = float(tf_cfg_raw.theta) if tf_cfg_raw.theta is not None else float(np.deg2rad(tf_phi))
            tf_phi_rad = float(tf_cfg_raw.phi) if tf_cfg_raw.phi is not None else float(np.deg2rad(tf_theta))
            tf_X0 = float(tf_cfg_raw.X0) if tf_cfg_raw.X0 is not None else float(tf_centroid[0])
            tf_Y0 = float(tf_cfg_raw.Y0) if tf_cfg_raw.Y0 is not None else float(tf_centroid[1])
            tf_Z0 = float(tf_cfg_raw.Z0) if tf_cfg_raw.Z0 is not None else float(tf_centroid[2])

            trunc_kwargs = {
                "truncation": True,
                "trunc_fault": {
                    "fault_points": tf_pts,
                    "theta": tf_theta_rad,
                    "phi": tf_phi_rad,
                    "X0": tf_X0,
                    "Y0": tf_Y0,
                    "Z0": tf_Z0,
                    "rbf_kwargs": tf_cfg_raw.rbf_kwargs,
                },
                "hanging_trunc": cfg.hanging_trunc,
            }
            trunc_mesh_arg = model.fault_meshes.get(tf_name)

        # ── domain (grid) features ────────────────────────────────────────────
        domain_feat = FiniteFault.fault_features(
            points=model.test_xyz,
            fault_points=fault_pts,
            theta=theta_rad, phi=phi_rad,
            X0=X0, Y0=Y0, Z0=Z0,
            lx=cfg.lx, ly=cfg.ly, lz=cfg.lz,
            dmax=cfg.dmax, lam=cfg.lam,
            rbf_kwargs=cfg.rbf_kwargs,
            Normal_fault=cfg.Normal_fault,
            **trunc_kwargs,
        )

        # ── training-point features ───────────────────────────────────────────
        train_feat = FiniteFault.fault_features(
            points=model.surface_points,
            fault_points=fault_pts,
            theta=theta_rad, phi=phi_rad,
            X0=X0, Y0=Y0, Z0=Z0,
            lx=cfg.lx, ly=cfg.ly, lz=cfg.lz,
            dmax=cfg.dmax, lam=cfg.lam,
            rbf_kwargs=cfg.rbf_kwargs,
            Normal_fault=cfg.Normal_fault,
            **trunc_kwargs,
        )

        orientation_feat = None
        if model.orientation_points is not None and model.orientation_points.shape[0] > 0:
            orientation_feat = FiniteFault.fault_features(
                points=model.orientation_points,
                fault_points=fault_pts,
                theta=theta_rad,
                phi=phi_rad,
                X0=X0,
                Y0=Y0,
                Z0=Z0,
                lx=cfg.lx,
                ly=cfg.ly,
                lz=cfg.lz,
                dmax=cfg.dmax,
                lam=cfg.lam,
                rbf_kwargs=cfg.rbf_kwargs,
                Normal_fault=cfg.Normal_fault,
                **trunc_kwargs,
            )

        # ── fault surface mesh ────────────────────────────────────────────────
        mesh_trunc_kwargs = {}
        if cfg.truncation and trunc_mesh_arg is not None:
            mesh_trunc_kwargs = {
                "truncation": True,
                "trunc_fault_mesh": trunc_mesh_arg,
                "hanging_trunc": cfg.hanging_trunc,
            }
        fault_mesh = FiniteFault.fault_mesh(
            points=model.test_xyz,
            fault_points=fault_pts,
            theta=theta_rad, phi=phi_rad,
            X0=X0, Y0=Y0, Z0=Z0,
            lx=cfg.lx, ly=cfg.ly, lz=cfg.lz,
            rbf_kwargs=cfg.rbf_kwargs,
            resolution=200,
            **mesh_trunc_kwargs,
        )

        # ── store ─────────────────────────────────────────────────────────────
        model.fault_names.append(name)
        model.fault_configs[name] = cfg
        model.domain_fault_feats[name] = domain_feat
        model.train_fault_feats[name] = train_feat
        if orientation_feat is not None:
            model.orientation_fault_feats[name] = orientation_feat
        model.fault_meshes[name] = fault_mesh

        # also attach to PyVista grid for inspection
        if model.grid_mesh is not None:
            model.grid_mesh.point_data[f"fault_feature_{name}"] = domain_feat.ravel()

    return model

