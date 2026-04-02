"""
geoinr_faults
=====
Convenience imports for the public API.

This package previously relied on implicit namespace packaging (no
`geoinr_faults/__init__.py`). In that mode, `import geoinr_faults` does not automatically
expose functions living in submodules, so calls like `geoinr_faults.create_geomodel(...)`
raise `AttributeError`.

Keeping a small `__init__` fixes that by re-exporting the high-level entrypoints.
"""

from __future__ import annotations
import warnings

warnings.filterwarnings("ignore")

from geoinr_faults.api.create import add_faults, add_stratis, create_geomodel
from geoinr_faults.features.fault_features import FaultConfig
from geoinr_faults.features.strati import StratiConfig

__all__ = [
    "create_geomodel",
    "add_stratis",
    "add_faults",
    "StratiConfig",
    "FaultConfig",
    "compute_model",
    "plot_model",
    "plot_input_data",
    "plot_fault_meshes",
    "plot_fault_features",
    "plot_loss",
    "plot_section",
    "plot_section_rock_unit",
    "plot_summary",
]

__version__ = "0.1.0"


def __getattr__(name: str):
    # Lazy imports for optional / heavier dependencies (torch, matplotlib, etc.)
    if name == "compute_model":
        from geoinr_faults.api.compute import compute_model as _compute_model

        return _compute_model

    if name in {
        "plot_model",
        "plot_input_data",
        "plot_fault_meshes",
        "plot_fault_features",
        "plot_loss",
        "plot_section",
        "plot_section_rock_unit",
        "plot_summary",
    }:
        from geoinr_faults.visualization.visualize import (
            plot_fault_meshes as _plot_fault_meshes,
            plot_fault_features as _plot_fault_features,
            plot_input_data as _plot_input_data,
            plot_loss as _plot_loss,
            plot_model as _plot_model,
            plot_section as _plot_section,
            plot_section_rock_unit as _plot_section_rock_unit,
            plot_summary as _plot_summary,
        )

        return {
            "plot_model": _plot_model,
            "plot_input_data": _plot_input_data,
            "plot_fault_meshes": _plot_fault_meshes,
            "plot_fault_features": _plot_fault_features,
            "plot_loss": _plot_loss,
            "plot_section": _plot_section,
            "plot_section_rock_unit": _plot_section_rock_unit,
            "plot_summary": _plot_summary,
        }[name]

    raise AttributeError(f"module 'geoinr_faults' has no attribute {name!r}")

