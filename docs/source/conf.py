from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

project = "GeoINR"
author = "GeoINR contributors"
copyright = "2026, GeoINR contributors"

try:
    from geoinr_faults import __version__ as release
except Exception:
    release = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

autosummary_generate = True
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
    "member-order": "bysource",
}
autodoc_member_order = "bysource"
autoclass_content = "both"
autodoc_typehints = "description"
napoleon_google_docstring = False
napoleon_numpy_docstring = True

# Allow API docs to build in a lightweight docs-only environment.
autodoc_mock_imports = [
    "numpy",
    "pandas",
    "torch",
    "pyvista",
    "scipy",
    "sklearn",
    "matplotlib",
    "matplotlib.pyplot",
]

html_theme = "alabaster"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_js_files = ["custom.js"]
html_logo = "_static/images/logo.png"
html_theme_options = {
    "description": "",
    "fixed_sidebar": True,
    "github_button": True,
    "github_type": "star",
    "logo_name": False,
    "sidebar_collapse": False,
    "page_width": "1080px",
    "body_max_width": "760px",
}


def _run_apidoc() -> None:
    from sphinx.ext.apidoc import main

    module_dir = ROOT / "geoinr_faults"
    api_dir = SOURCE / "api"
    api_dir.mkdir(parents=True, exist_ok=True)

    main(
        [
            "-f",
            "-e",
            "-M",
            "-o",
            str(api_dir),
            str(module_dir),
            str(module_dir / "__pycache__"),
        ]
    )


def _prepare_docs(_app) -> None:
    try:
        _run_apidoc()
    except Exception as exc:
        print(f"[docs] Warning: API docs generation failed: {exc}")


def setup(app):
    app.connect("builder-inited", _prepare_docs)
